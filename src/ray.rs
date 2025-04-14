use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::sync::{mpsc, Arc, RwLock};

use anyhow::{anyhow, Result};
use ash::{vk, RawPtr};
use vk_mem;
use winit::dpi::PhysicalSize;

use crate::primitives::{Addressable, Extrema, ObjectType, Objectionable, Primitive, AABB};
use crate::uniforms::{self, UniformBufferObject};
use crate::vulkan::Destructor;
use crate::{
    core, material, physics, primitives, structures, tools, transfer, vulkan, MAX_FRAMES_IN_FLIGHT,
};

pub enum TransferCommand {
    Pause(u64),
}

pub fn thread(
    device: ash::Device,

    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,

    queue_family_indices: structures::QueueFamilyIndices,
    uniform_buffers: [vk::Buffer; 2],

    should_threads_die: Arc<RwLock<bool>>,
    transfer_semaphore: vk::Semaphore,
    transfer_sender: mpsc::Sender<transfer::ResizedSource>,
    transfer_commands: mpsc::Receiver<TransferCommand>,
    mut resize: single_value_channel::Receiver<PhysicalSize<u32>>,
    notify_complete_frame: single_value_channel::Updater<u8>,
    update_as: mpsc::Receiver<physics::UpdateScene>,
    add_material: mpsc::Receiver<material::Material>,
    uniform: mpsc::Sender<uniforms::Event>,
    latest_frame_index: single_value_channel::Updater<usize>,
) {
    log::trace!("Creating thread");
    let mut allocator_create_info =
        vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
    allocator_create_info.vulkan_api_version = vk::API_VERSION_1_3;
    allocator_create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

    let allocator = match unsafe { vk_mem::Allocator::new(allocator_create_info) } {
        Ok(v) => Arc::new(v),
        Err(e) => {
            panic!("Failed to create an allocator: {:?}", e);
        }
    };

    let rt_device = ash::khr::ray_tracing_pipeline::Device::new(&instance, &device);
    let as_device = ash::khr::acceleration_structure::Device::new(&instance, &device);

    let mut ray_tracing_pipeline_properties =
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut device_properties =
        vk::PhysicalDeviceProperties2::default().push(&mut ray_tracing_pipeline_properties);

    unsafe { instance.get_physical_device_properties2(physical_device, &mut device_properties) };

    let mut ray = match Ray::new(
        device,
        instance,
        physical_device,
        rt_device,
        as_device,
        ray_tracing_pipeline_properties,
        allocator,
        queue_family_indices,
        uniform_buffers,
        update_as,
        add_material,
        uniform,
    ) {
        Ok(v) => v,
        Err(e) => {
            log::error!("Failed to create ray tracing object: {}", e);
            return;
        }
    };

    log::trace!("Ray tracing object has been constructed");

    // The latest value used by the timeline semaphore
    let mut current_timestamp = 0;
    // The latest value we have been geiven by Transfer that we cannot work until complete
    let mut current_transfer_timestamp = 0;
    // The index in the frames in flight that we are currently writing to
    let mut current_frame_index = 0;

    let mut first_run = true;

    let mut is_minimised = false;

    loop {
        // Check if we should end the thread
        match should_threads_die.read() {
            Ok(should_die) => {
                if *should_die == true {
                    break;
                }
            }
            Err(e) => {
                log::error!("rwlock is poisoned, ending thread: {}", e)
            }
        }

        let mut size = resize.latest();

        // TODO: Figure out why Ray sometimes doesn't always start until a manual resize/draw occurs
        if first_run {
            let new_images = match ray.resize(*size) {
                Err(e) => {
                    log::error!("Failed to create new output images: {e}");
                    break;
                }
                Ok(v) => v,
            };
            log::warn!(
                "First images {:?} are now {}x{}",
                new_images,
                size.width,
                size.height
            );
            match transfer_sender.send(transfer::ResizedSource::Ray((*size, new_images))) {
                Err(e) => log::error!("Failed to update transfer with new images: {e}"),
                Ok(()) => {}
            };
        } else {
            // Wait for previous frame to finish
            match core::wait_on_semaphore(
                &ray.device,
                ray.semaphore.get(),
                current_timestamp,
                10_000_000, // 10ms
            ) {
                Err(vk::Result::TIMEOUT) => continue,
                Err(e) => {
                    log::error!("Failed to wait for previous render to finish: {e}");
                    break;
                }
                Ok(()) => {}
            }

            // Send current frame to transfer thread so that it knows where it can get a valid frame from
            match notify_complete_frame.update(current_frame_index as u8) {
                Err(e) => {
                    log::error!("Failed to notify transfer channel: {e}");
                    break;
                }
                Ok(()) => {}
            }
            current_frame_index += 1;
            current_frame_index %= MAX_FRAMES_IN_FLIGHT;
        }

        while ray.need_resize(*size) {
            if size.width == 0 || size.height == 0 {
                is_minimised = true;
                break;
            }
            is_minimised = false;
            // log::debug!("Received resize to {}x{}", size.width, size.height);
            // We have been told to not do any work until a new timeline has been reached
            let new_images = match ray.resize(*size) {
                Err(e) => {
                    log::error!("Failed to create new output images: {e}");
                    break;
                }
                Ok(v) => v,
            };
            match transfer_sender.send(transfer::ResizedSource::Ray((*size, new_images))) {
                Err(e) => log::error!("Failed to update transfer with new images: {e}"),
                Ok(()) => {}
            };
            log::debug!(
                "Images {:?} are now {}x{}",
                new_images,
                size.width,
                size.height
            );
            size = resize.latest();
        }
        // std::thread::sleep(std::time::Duration::from_millis(1000));

        // Check if transfer is actively using our images and we need to wait for it to finish
        // or if the window has been resized and we need to change our images
        match transfer_commands.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {
                // Nothing to wait for, we can go on as planned
            }
            Ok(TransferCommand::Pause(v)) => {
                // We have been told to not do any work until a new timeline has been reached
                current_transfer_timestamp = v;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Failed to receive a notification from transfer");
                break;
            }
        }

        if let Err(e) = ray.update() {
            log::error!("Failed to check for updates: {e}");
            break;
        }

        if is_minimised {
            continue;
        }

        // Perform a render
        // TODO: Measure time taken
        match ray.ray_trace(
            current_frame_index,
            transfer_semaphore,
            current_transfer_timestamp,
            current_timestamp,
            current_timestamp + 1,
        ) {
            Err(e) => {
                log::error!("Failed to perform compute: {e}")
            }
            Ok(()) => {
                if let Err(e) = latest_frame_index.update(current_frame_index) {
                    log::error!("Failed to send latest frame index to uniforms: {e}")
                }
            }
        };
        if first_run {
            log::info!("No longer first run");
        }
        first_run = false;
        current_timestamp += 1;
    }
    log::warn!("Ending thread");
}

struct Ray<'a> {
    ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,
    raygen_shader_binding_table: vulkan::Buffer,
    miss_shader_binding_table: vulkan::Buffer,
    hit_shader_binding_table: vulkan::Buffer,

    update_as: mpsc::Receiver<physics::UpdateScene>,
    add_material: mpsc::Receiver<material::Material>,
    uniform: mpsc::Sender<uniforms::Event>,
    semaphore: Destructor<vk::Semaphore>,

    images: Option<[vulkan::Image<'a>; 2]>,

    uniform_buffers: [vk::Buffer; 2],

    queue: vk::Queue,
    descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    pipeline_layout: Destructor<vk::PipelineLayout>,
    pipeline: Destructor<vk::Pipeline>,
    command_pool: Destructor<vk::CommandPool>,
    commands: [vk::CommandBuffer; 2],
    descriptor_pool: Destructor<vk::DescriptorPool>,
    descriptor_sets: Option<[vk::DescriptorSet; 2]>,

    primitive_id_location_map: HashMap<u64, (Primitive, usize)>,
    materials_buffer: vulkan::Buffer,
    aabbs_buffers: Vec<vulkan::Buffer>,

    buffer_references: vulkan::Buffer,
    blass: Vec<AccelerationStructure>,
    top_level_as: AccelerationStructure,

    allocator: Arc<vk_mem::Allocator>,
    physical_device: vk::PhysicalDevice,
    rt_device: ash::khr::ray_tracing_pipeline::Device,
    as_device: ash::khr::acceleration_structure::Device,
    instance: ash::Instance,
    device: ash::Device,
}

impl<'a> Ray<'a> {
    fn new(
        device: ash::Device,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        rt_device: ash::khr::ray_tracing_pipeline::Device,
        as_device: ash::khr::acceleration_structure::Device,
        ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,
        allocator: Arc<vk_mem::Allocator>,
        queue_family_indices: structures::QueueFamilyIndices,
        uniform_buffers: [vk::Buffer; 2],

        update_as: mpsc::Receiver<physics::UpdateScene>,
        add_material: mpsc::Receiver<material::Material>,
        uniform: mpsc::Sender<uniforms::Event>,
    ) -> Result<Self> {
        log::trace!("Creating object");
        // Populates queue
        let queue = core::create_queue(&device, queue_family_indices.compute_family.unwrap());

        let (command_pool, commands) = create_empty_commands(&device, queue_family_indices)?;

        let materials_buffer = create_materials(
            &device,
            command_pool.get(),
            queue,
            &allocator,
            &add_material,
        )?;

        let (primitive_id_location_map, blass, blas_instances, primitive_addresses, aabbs_buffers) =
            create_blas(
                &allocator,
                &device,
                &as_device,
                command_pool.get(),
                queue,
                &update_as,
            )?;

        // Get a reference to where all the parts of each model are on the GPU
        let buffer_references = create_buffer_references(
            &device,
            command_pool.get(),
            queue,
            &allocator,
            &primitive_addresses,
        )?;

        let top_level_as = create_top_level_acceleration_structure(
            &device,
            &allocator,
            &as_device,
            &blas_instances,
            queue,
            command_pool.get(),
        )?;

        log::trace!("Creating pipeline");
        let (descriptor_set_layout, shader_groups, pipeline_layout, pipeline) =
            create_pipeline(&device, &rt_device)?;

        log::trace!("Creating SBT");
        let (raygen_shader_binding_table, miss_shader_binding_table, hit_shader_binding_table) =
            create_shader_binding_tables(
                &allocator,
                &rt_device,
                pipeline.get(),
                &ray_tracing_pipeline_properties,
                &shader_groups,
            )?;
        let descriptor_pool = create_descriptor_pool(&device)?;

        log::trace!("Building semaphore");
        let semaphore = Destructor::new(
            &device,
            core::create_semaphore(&device)?,
            device.fp_v1_0().destroy_semaphore,
        );

        Ok(Self {
            device,
            instance,
            physical_device,
            allocator,
            semaphore,
            images: None,
            queue,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            command_pool,
            commands,
            descriptor_pool,
            descriptor_sets: None,
            rt_device,
            as_device,
            ray_tracing_pipeline_properties,
            raygen_shader_binding_table,
            miss_shader_binding_table,
            hit_shader_binding_table,
            uniform_buffers,
            top_level_as,
            buffer_references,
            blass,
            materials_buffer,
            aabbs_buffers,
            primitive_id_location_map,
            update_as,
            add_material,
            uniform,
        })
    }
    pub fn need_resize(&self, size: PhysicalSize<u32>) -> bool {
        if let Some(images) = &self.images {
            // Return true if either image size is incorrect
            !images[0].is_correct_size(size.width, size.height)
                || !images[1].is_correct_size(size.width, size.height)
        } else {
            false
        }
    }
    pub fn initial_size(&mut self, size: PhysicalSize<u32>) -> Result<[vk::Image; 2]> {
        log::trace!("Creating initial images and commands");
        self.images = Some(core::create_storage_image_pair(
            &self.device,
            &self.instance,
            &self.allocator,
            self.physical_device,
            self.command_pool.get(),
            self.queue,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            size,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        )?);
        self.descriptor_sets = Some(create_descriptor_sets(
            &self.device,
            self.descriptor_set_layout.get(),
            self.descriptor_pool.get(),
            &self.top_level_as,
            self.images.as_ref().unwrap(),
            &self.uniform_buffers,
            &self.buffer_references,
            &self.materials_buffer,
        )?);
        build_command_buffers(
            &self.device,
            &self.rt_device,
            size.width,
            size.height,
            self.descriptor_sets.unwrap(),
            self.commands,
            &self.ray_tracing_pipeline_properties,
            &self.raygen_shader_binding_table,
            &self.miss_shader_binding_table,
            &self.hit_shader_binding_table,
            self.pipeline.get(),
            self.pipeline_layout.get(),
        )?;
        let mut raw_images = [vk::Image::null(); 2];
        for i in 0..self.images.as_ref().unwrap().len() {
            raw_images[i] = self.images.as_ref().unwrap()[i].get();
        }
        Ok(raw_images)
    }
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) -> Result<[vk::Image; 2]> {
        log::trace!("Resizing to {}x{}", new_size.width, new_size.height);
        if self.images.is_none() {
            return self.initial_size(new_size);
        }

        let mut resized = false;
        // Check if either storage image needed a resize and perform it
        for i in 0..2 {
            if self.images.as_mut().unwrap()[i].resize(new_size.width, new_size.height)? {
                resized = true;
                core::transition_image_layout(
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                    self.images.as_mut().unwrap()[i].get(),
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                )?;
            }
        }

        if resized {
            for (i, &descriptor_set) in self.descriptor_sets.unwrap().iter().enumerate() {
                // Now need to update the descriptor to reference the new image
                let read_image_descriptor = [vk::DescriptorImageInfo {
                    image_view: self.images.as_ref().unwrap()[i].view(),
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                }];
                let write_image_descriptor = [vk::DescriptorImageInfo {
                    image_view: self.images.as_ref().unwrap()[(i + 1) % 2].view(),
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                }];

                let descriptor_writes = [
                    vk::WriteDescriptorSet {
                        dst_set: descriptor_set,
                        dst_binding: 1,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        ..Default::default()
                    }
                    .image_info(&read_image_descriptor),
                    vk::WriteDescriptorSet {
                        dst_set: descriptor_set,
                        dst_binding: 2,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        ..Default::default()
                    }
                    .image_info(&write_image_descriptor),
                ];
                unsafe { self.device.update_descriptor_sets(&descriptor_writes, &[]) };
            }
        }
        build_command_buffers(
            &self.device,
            &self.rt_device,
            new_size.width,
            new_size.height,
            self.descriptor_sets.unwrap(),
            self.commands,
            &self.ray_tracing_pipeline_properties,
            &self.raygen_shader_binding_table,
            &self.miss_shader_binding_table,
            &self.hit_shader_binding_table,
            self.pipeline.get(),
            self.pipeline_layout.get(),
        )?;
        let mut raw_images = [vk::Image::null(); 2];
        for i in 0..self.images.as_ref().unwrap().len() {
            raw_images[i] = self.images.as_ref().unwrap()[i].get();
        }
        log::trace!("Resize complete: have commands {:?}", self.commands);
        Ok(raw_images)
    }
    pub fn update(&mut self) -> Result<()> {
        let mut changed = false;

        loop {
            match self.add_material.try_recv() {
                Ok(m) => {
                    log::warn!("TODO: Implement adding materials")
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Nothing to wait for, we can go on as planned
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(anyhow!(
                        "Failed to receive an acceleration structure update"
                    ));
                }
            }
        }

        loop {
            match self.update_as.try_recv() {
                Ok(physics::UpdateScene::Add(id, obj)) => {
                    changed = true;
                    log::warn!("TODO: Implement adding objects")
                }
                Ok(physics::UpdateScene::Remove(id)) => {
                    changed = true;
                    log::warn!("TODO: Implement removing objects")
                }
                Ok(physics::UpdateScene::AddInstance(
                    instance_id,
                    primitive_id,
                    transformation,
                )) => {
                    changed = true;
                    log::warn!("TODO: Implement adding instances")
                }
                Ok(physics::UpdateScene::RemoveInstance(instance_id)) => {
                    changed = true;
                    log::warn!("TODO: Implement removing instances")
                }
                Ok(physics::UpdateScene::MoveInstance(instance_id, transformation)) => {
                    changed = true;
                    log::warn!("TODO: Implement moving instances")
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Nothing to wait for, we can go on as planned
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(anyhow!(
                        "Failed to receive an acceleration structure update"
                    ));
                }
            }
        }
        if changed {
            self.uniform.send(uniforms::Event::ResetSamples)?;
        }
        Ok(())
    }
    pub fn ray_trace(
        &self,
        current_frame_index: usize,
        transfer_semaphore: vk::Semaphore,
        transfer_timestamp_to_wait: u64,
        ray_timestamp_to_wait: u64,
        timestamp_to_signal: u64,
    ) -> Result<()> {
        // log::trace!(
        //     "Ray tracing into {current_frame_index} when timestamp reaches {}",
        //     transfer_timestamp_to_wait
        // );
        let wait_timestamps = [transfer_timestamp_to_wait, ray_timestamp_to_wait];
        let wait_semaphores = [transfer_semaphore, self.semaphore.get()];
        let wait_stages = [
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        ];

        if self.images.is_none() {
            // We only want to start ray tracing once the output images are set up
            return Ok(());
        }

        let signal_timestamps = [timestamp_to_signal];
        let signal_semaphores = [self.semaphore.get()];

        let command_buffer = [self.commands[current_frame_index]];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_timestamps)
            .wait_semaphore_values(&wait_timestamps);
        let submit_info = [vk::SubmitInfo::default()
            .command_buffers(&command_buffer)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .push(&mut timeline_info)];

        unsafe {
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
        }?;
        self.uniform.send(uniforms::Event::RayTick)?;
        Ok(())
    }
}

impl<'a> Drop for Ray<'a> {
    fn drop(&mut self) {
        unsafe {
            match self.device.queue_wait_idle(self.queue) {
                Err(e) => log::error!("Ray failed to wait for its queue to finish: {e}"),
                _ => {}
            }
        }
        log::trace!("Cleaned up Ray");
    }
}

struct ScratchBuffer {
    pub device_address: vk::DeviceAddress,
    pub _buffer: vulkan::Buffer,
}
impl ScratchBuffer {
    fn new(
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        log::trace!("Creating Scratch buffer");
        let _buffer = vulkan::Buffer::new_gpu(
            allocator,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        let device_address = get_buffer_device_address(device, _buffer.get());

        Ok(Self {
            device_address,
            _buffer,
        })
    }
}

struct AccelerationStructure {
    device: vk::Device,
    destructor: vk::PFN_vkDestroyAccelerationStructureKHR,

    pub handle: vk::AccelerationStructureKHR,
    pub device_address: vk::DeviceAddress,
    pub buffer: Option<vulkan::Buffer>,
}

impl AccelerationStructure {
    pub fn new(device: &ash::khr::acceleration_structure::Device) -> Self {
        Self {
            device: device.device(),
            destructor: device.fp().destroy_acceleration_structure_khr,

            handle: vk::AccelerationStructureKHR::null(),
            device_address: 0,
            buffer: None,
        }
    }
    pub fn create_buffer(
        &mut self,
        allocator: &vk_mem::Allocator,
        build_size_info: vk::AccelerationStructureBuildSizesInfoKHR,
    ) -> Result<()> {
        log::trace!("Creating AS buffer");
        self.buffer = Some(vulkan::Buffer::new_gpu(
            allocator,
            build_size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?);
        Ok(())
    }
    pub fn get_buffer(&self) -> vk::Buffer {
        match &self.buffer {
            Some(v) => v.get(),
            None => {
                log::error!("Acceleration Structure Buffer was not initialised prior to calling");
                vk::Buffer::null()
            }
        }
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        log::trace!("Dropping Acceleration Structure");
        unsafe { (self.destructor)(self.device, self.handle, None.as_raw_ptr()) };
    }
}

fn get_buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    let buffer_device_ai = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    unsafe { device.get_buffer_device_address(&buffer_device_ai) }
}

///  Create the bottom level acceleration structure that contains the scene's geometry (triangles)
fn create_bottom_level_acceleration_structure(
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    as_device: &ash::khr::acceleration_structure::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    geometry_type: vk::GeometryTypeKHR,
    // Either Model (triangles) or AABB buffer and number of primitives
    obj: Option<&primitives::model::Model>,
    aabb: Option<&vulkan::Buffer>,
    prim_count: Option<u32>,
) -> Result<AccelerationStructure> {
    #[rustfmt::skip]
    let transform_matrix = vk::TransformMatrixKHR { matrix: [
		1.0, 0.0, 0.0, 0.0, 
		0.0, 1.0, 0.0, 0.0, 
		0.0, 0.0, 1.0, 0.0],
	};

    // TODO: Add a staging buffer so these buffers can live on the GPU
    let transform_buffer = vulkan::Buffer::new_populated(
        allocator,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        &transform_matrix,
        1,
    )?;

    // Build
    let mut as_geometries = [vk::AccelerationStructureGeometryKHR::default()
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry_type(geometry_type)];

    match geometry_type {
        vk::GeometryTypeKHR::TRIANGLES => {
            assert!(obj.is_some());
            let obj = obj.unwrap();

            let vertex_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
                device_address: obj.primitive_data.as_ref().unwrap().vertices,
            };
            let index_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
                device_address: obj.primitive_data.as_ref().unwrap().indices,
            };
            let transform_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
                device_address: get_buffer_device_address(device, transform_buffer.get()),
            };

            as_geometries[0].geometry.triangles =
                vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vertex_buffer_device_address)
                    .max_vertex(obj.vertices.len() as u32)
                    .vertex_stride(size_of::<primitives::model::Vertex>() as u64)
                    .index_type(vk::IndexType::UINT32)
                    .index_data(index_buffer_device_address)
                    .transform_data(transform_buffer_device_address);
        }
        vk::GeometryTypeKHR::AABBS => {
            assert!(aabb.is_some());
            assert!(prim_count.is_some());
            let aabb = aabb.unwrap();
            as_geometries[0].geometry.aabbs =
                vk::AccelerationStructureGeometryAabbsDataKHR::default()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: aabb.get_device_address(device),
                    })
                    .stride(size_of::<primitives::AABB>() as u64);
        }
        _ => {
            return Err(anyhow!(
                "Trying to construct a BLAS from unsupported primitive {:?}",
                geometry_type
            ))
        }
    }
    let as_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(&as_geometries);

    let num_prim = if let Some(o) = obj {
        [o.indices.len() as u32 / 3] // Number of triangle
    } else if let Some(v) = prim_count {
        [v] // Number of primitives in AABB
    } else {
        return Err(anyhow!(
            "Trying to construct a BLAS without an object or a count of primitives"
        ));
    };
    let mut as_build_sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

    unsafe {
        as_device.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &as_build_geometry_info,
            &num_prim,
            &mut as_build_sizes_info,
        );
    }

    // Create a buffer to hold the acceleration structure
    let mut bottom_level_as = AccelerationStructure::new(as_device);
    bottom_level_as.create_buffer(allocator, as_build_sizes_info)?;

    // Create the Acceleration Structure
    let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(bottom_level_as.get_buffer())
        .size(as_build_sizes_info.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
    bottom_level_as.handle =
        unsafe { as_device.create_acceleration_structure(&as_create_info, None) }?;

    // The actual build process starts here

    // Create a scratch buffer as a temporary storage for the acceleration structure build
    // TODO: Share the scratch buffer between all BLAS constructions
    let scratch_buffer =
        ScratchBuffer::new(device, allocator, as_build_sizes_info.build_scratch_size)?;

    // TODO: Compaction (https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/#accelerationstructure/bottom-levelaccelerationstructure/helperdetails:raytracingbuilder::buildblas())
    let mut as_build_geometry_infos = [vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .dst_acceleration_structure(bottom_level_as.handle)
        .geometries(&as_geometries)];
    as_build_geometry_infos[0].scratch_data.device_address = scratch_buffer.device_address;

    let as_build_range_infos =
        [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(num_prim[0])];

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
    let command_buffer = core::begin_single_time_commands(device, command_pool)?;
    unsafe {
        as_device.cmd_build_acceleration_structures(
            command_buffer,
            &as_build_geometry_infos,
            &[&as_build_range_infos],
        );
    }
    core::end_single_time_command(device, command_pool, queue, command_buffer)?;

    // Get the bottom acceleration structure's handle, which will be used during the top level acceleration build
    let as_device_address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
        .acceleration_structure(bottom_level_as.handle);
    bottom_level_as.device_address =
        unsafe { as_device.get_acceleration_structure_device_address(&as_device_address_info) };

    Ok(bottom_level_as)
}

fn create_top_level_acceleration_structure(
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    as_device: &ash::khr::acceleration_structure::Device,
    blas_instances: &Vec<vk::AccelerationStructureInstanceKHR>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
) -> Result<AccelerationStructure> {
    log::debug!("Making a TLAS with {} instances", blas_instances.len());
    let instances_buffer = vulkan::Buffer::new_populated(
        allocator,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        blas_instances.as_ptr(),
        blas_instances.len(),
    )?;

    let instance_data_device_address = vk::DeviceOrHostAddressConstKHR {
        device_address: get_buffer_device_address(device, instances_buffer.get()),
    };

    // The top level acceleration structure contains (bottom level) instances as the input geometry
    let as_geometries = [vk::AccelerationStructureGeometryKHR {
        geometry_type: vk::GeometryTypeKHR::INSTANCES,
        flags: vk::GeometryFlagsKHR::OPAQUE,
        geometry: vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                array_of_pointers: vk::FALSE,
                data: instance_data_device_address,
                ..Default::default()
            },
        },
        ..Default::default()
    }];

    // Get the size requirements for buffers involved in the acceleration structure build process
    let mut as_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        ..Default::default()
    };
    as_build_geometry_info = as_build_geometry_info.geometries(&as_geometries);

    let primitive_count = [blas_instances.len() as u32];

    let mut as_build_sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        as_device.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &as_build_geometry_info,
            &primitive_count,
            &mut as_build_sizes_info,
        )
    };

    let mut top_level_as = AccelerationStructure::new(as_device);
    // Create a buffer to hold the acceleration structure
    top_level_as.create_buffer(allocator, as_build_sizes_info)?;

    // Create the acceleration structure
    let as_create_info = vk::AccelerationStructureCreateInfoKHR {
        buffer: top_level_as.get_buffer(),
        size: as_build_sizes_info.acceleration_structure_size,
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        ..Default::default()
    };
    top_level_as.handle =
        unsafe { as_device.create_acceleration_structure(&as_create_info, None) }?;

    // The actual build process starts here

    // Create a scratch buffer as a temporary storage for the acceleration structure build
    let scratch_buffer =
        ScratchBuffer::new(device, allocator, as_build_sizes_info.build_scratch_size)?;

    let as_build_geometry_infos = [vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        dst_acceleration_structure: top_level_as.handle,
        scratch_data: vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address,
        },
        ..Default::default()
    }
    .geometries(&as_geometries)];

    let as_build_range_infos = [
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count[0])
    ];

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
    let command_buffer = core::begin_single_time_commands(device, command_pool)?;
    unsafe {
        as_device.cmd_build_acceleration_structures(
            command_buffer,
            &as_build_geometry_infos,
            &[&as_build_range_infos],
        );
    }
    core::end_single_time_command(device, command_pool, queue, command_buffer)?;

    Ok(top_level_as)
}

fn create_blas_instance(
    as_device: &ash::khr::acceleration_structure::Device,
    blass: &mut Vec<AccelerationStructure>,
    blas_id: usize,
    transform: cgmath::Matrix4<f32>,
    object_type: primitives::ObjectType,
) -> Result<vk::AccelerationStructureInstanceKHR> {
    #[rustfmt::skip]
    let transform_matrix = vk::TransformMatrixKHR {matrix:[
        transform.x[0], transform.y[0], transform.z[0], transform.w[0],
        transform.x[1], transform.y[1], transform.z[1], transform.w[1],
        transform.x[2], transform.y[2], transform.z[2], transform.w[2]],
    };

    let blas = &mut blass[blas_id];

    // Get the bottom acceleration structure's handle, which will be used during the top level acceleration build
    let as_device_address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
        .acceleration_structure(blas.handle);
    blas.device_address =
        unsafe { as_device.get_acceleration_structure_device_address(&as_device_address_info) };

    Ok(vk::AccelerationStructureInstanceKHR {
        transform: transform_matrix,
        instance_custom_index_and_mask: vk::Packed24_8::new(blas_id as u32, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            if object_type == primitives::ObjectType::Triangle {
                0
            } else {
                1
            },
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas.device_address,
        },
    })
}

fn aligned_size(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

fn create_shader_binding_tables(
    allocator: &vk_mem::Allocator,
    rt_device: &ash::khr::ray_tracing_pipeline::Device,
    pipeline: vk::Pipeline,
    ray_tracing_pipeline_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    shader_groups: &Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
) -> Result<(vulkan::Buffer, vulkan::Buffer, vulkan::Buffer)> {
    let handle_size = ray_tracing_pipeline_properties.shader_group_handle_size;
    let base_size = ray_tracing_pipeline_properties.shader_group_base_alignment as u64;

    let group_count = shader_groups.len() as u32;
    let sbt_size = (group_count * handle_size) as usize;
    let sbt_buffer_usage_flags = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

    // Raygen
    // Create binding table buffers for each shader type
    let mut raygen_shader_binding_table = vulkan::Buffer::new_generic(
        allocator,
        base_size,
        vk_mem::MemoryUsage::Auto,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        sbt_buffer_usage_flags,
    )?;
    let mut miss_shader_binding_table = vulkan::Buffer::new_generic(
        allocator,
        base_size,
        vk_mem::MemoryUsage::Auto,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        sbt_buffer_usage_flags,
    )?;
    let mut hit_shader_binding_table = vulkan::Buffer::new_generic(
        allocator,
        base_size,
        vk_mem::MemoryUsage::Auto,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        sbt_buffer_usage_flags,
    )?;

    // Copy the pipeline's shader handles into a host buffer
    let shader_handle_storage = unsafe {
        rt_device.get_ray_tracing_shader_group_handles(pipeline, 0, group_count, sbt_size)?
    };

    // Copy the shader handles from the host buffer to the binding tables
    unsafe {
        raygen_shader_binding_table
            .populate(shader_handle_storage.as_ptr(), handle_size as usize)?;
        miss_shader_binding_table.populate(
            shader_handle_storage.as_ptr().offset(handle_size as isize),
            handle_size as usize,
        )?;
        hit_shader_binding_table.populate(
            shader_handle_storage
                .as_ptr()
                .offset((handle_size * 2) as isize),
            handle_size as usize * 2,
        )?;
    }
    Ok((
        raygen_shader_binding_table,
        miss_shader_binding_table,
        hit_shader_binding_table,
    ))
}

fn create_descriptor_pool(device: &ash::Device) -> Result<Destructor<vk::DescriptorPool>> {
    let pool_sizes = vec![
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        },
    ];
    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32);
    Ok(Destructor::new(
        device,
        unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None)? },
        device.fp_v1_0().destroy_descriptor_pool,
    ))
}

fn create_descriptor_sets(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    top_level_as: &AccelerationStructure,
    storage_images: &[vulkan::Image; 2],
    ubo: &[vk::Buffer; 2],
    scene: &vulkan::Buffer,
    materials: &vulkan::Buffer,
) -> Result<[vk::DescriptorSet; 2]> {
    let set_layouts = [set_layout, set_layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);

    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&allocate_info)? };

    let structures = [top_level_as.handle];
    for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
        let mut descriptor_acceleration_structure_info =
            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&structures);

        let read_image_descriptor = [vk::DescriptorImageInfo {
            image_view: storage_images[i].view(),
            image_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        }];
        let write_image_descriptor = [vk::DescriptorImageInfo {
            image_view: storage_images[i].view(),
            image_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        }];

        let buffer_descriptor = [vk::DescriptorBufferInfo {
            buffer: ubo[i],
            offset: 0,
            range: size_of::<UniformBufferObject>() as u64,
        }];
        let scene_descriptor = [scene.create_descriptor()];
        let material_descriptor = [materials.create_descriptor()];

        let descriptor_writes = [
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                ..Default::default()
            }
            .push(&mut descriptor_acceleration_structure_info), // Chain the AS descriptor
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                ..Default::default()
            }
            .image_info(&read_image_descriptor),
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                ..Default::default()
            }
            .image_info(&write_image_descriptor),
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 3,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                ..Default::default()
            }
            .buffer_info(&buffer_descriptor),
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 4,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&scene_descriptor),
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 5,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            }
            .buffer_info(&material_descriptor),
        ];
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }
    Ok([descriptor_sets[0], descriptor_sets[1]])
}

fn create_pipeline<'a>(
    device: &ash::Device,
    rt_device: &ash::khr::ray_tracing_pipeline::Device,
) -> Result<(
    Destructor<vk::DescriptorSetLayout>,
    Vec<vk::RayTracingShaderGroupCreateInfoKHR<'a>>,
    Destructor<vk::PipelineLayout>,
    Destructor<vk::Pipeline>,
)> {
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            // UBO
            binding: 3,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR
                | vk::ShaderStageFlags::INTERSECTION_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            // Scene
            binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::INTERSECTION_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            // Materials
            binding: 5,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ..Default::default()
        },
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let descriptor_set_layout = Destructor::new(
        device,
        unsafe { device.create_descriptor_set_layout(&layout_info, None) }?,
        device.fp_v1_0().destroy_descriptor_set_layout,
    );

    let set_layouts = [descriptor_set_layout.get()];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

    let pipeline_layout = Destructor::new(
        device,
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? },
        device.fp_v1_0().destroy_pipeline_layout,
    );

    let mut shader_stages = vec![];
    let mut shader_groups = vec![];

    // TODO: Replace with slang api
    let main_name = CString::new("main").unwrap();
    let raygen_code = tools::read_shader_code(Path::new("shaders/spv/raygen.slang.spv"))?;
    let miss_code = tools::read_shader_code(Path::new("shaders/spv/miss.slang.spv"))?;
    let intersection_code =
        tools::read_shader_code(Path::new("shaders/spv/intersection.slang.spv"))?;
    let closest_hit_code =
        tools::read_shader_code(Path::new("shaders/spv/closest_hit_triangle.slang.spv"))?;
    let sphere_closest_hit_code =
        tools::read_shader_code(Path::new("shaders/spv/closest_hit_generic.slang.spv"))?;

    let raygen_module = core::create_shader_module(device, &raygen_code)?;
    let miss_module = core::create_shader_module(device, &miss_code)?;
    let intersection_module = core::create_shader_module(device, &intersection_code)?;
    let closest_hit_module = core::create_shader_module(device, &closest_hit_code)?;
    let sphere_closest_hit_module = core::create_shader_module(device, &sphere_closest_hit_code)?;
    log::trace!("All Shader modules created");
    {
        // Ray generation group
        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(raygen_module.get())
                .name(&main_name),
        );
        let group_ci = vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
            general_shader: shader_stages.len() as u32 - 1,
            closest_hit_shader: vk::SHADER_UNUSED_KHR,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        };
        shader_groups.push(group_ci);
    }
    {
        // Ray miss group
        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(miss_module.get())
                .name(&main_name),
        );
        let group_ci = vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
            general_shader: shader_stages.len() as u32 - 1,
            closest_hit_shader: vk::SHADER_UNUSED_KHR,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        };
        shader_groups.push(group_ci);
    }
    {
        // Ray closest hit group
        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(closest_hit_module.get())
                .name(&main_name),
        );
        let group_ci = vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
            general_shader: vk::SHADER_UNUSED_KHR,
            closest_hit_shader: shader_stages.len() as u32 - 1,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        };
        shader_groups.push(group_ci);
    }
    {
        // Procedural hit group
        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(sphere_closest_hit_module.get())
                .name(&main_name),
        );
        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
                .module(intersection_module.get())
                .name(&main_name),
        );
        let group_ci = vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP,
            general_shader: vk::SHADER_UNUSED_KHR,
            closest_hit_shader: shader_stages.len() as u32 - 2,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: shader_stages.len() as u32 - 1,
            ..Default::default()
        };
        shader_groups.push(group_ci);
    }

    let ray_tracing_pipeline_create_infos = [vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(shader_stages.as_slice())
        .groups(&shader_groups)
        .max_pipeline_ray_recursion_depth(1)
        .layout(pipeline_layout.get())];

    log::trace!("Building pipeline {:#?}", rt_device.device());
    let pipeline = unsafe {
        rt_device.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &ray_tracing_pipeline_create_infos,
            None,
        )
    };
    log::trace!("Pipeline created - checking");
    match pipeline {
        Ok(v) => Ok((
            descriptor_set_layout,
            shader_groups,
            pipeline_layout,
            Destructor::new(device, v[0], device.fp_v1_0().destroy_pipeline),
        )),
        Err(e) => Err(anyhow!("Failed to create graphics pipeline: {:?}", e)),
    }
}

fn build_command_buffers(
    device: &ash::Device,
    rt_device: &ash::khr::ray_tracing_pipeline::Device,
    width: u32,
    height: u32,
    descriptor_sets: [vk::DescriptorSet; 2],
    command_buffers: [vk::CommandBuffer; 2],
    ray_tracing_pipeline_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    raygen_shader_binding_table: &vulkan::Buffer,
    miss_shader_binding_table: &vulkan::Buffer,
    hit_shader_binding_table: &vulkan::Buffer,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
) -> Result<()> {
    let begin_info = vk::CommandBufferBeginInfo::default();

    // for &command_buffer in command_buffers {
    for (i, &command_buffer) in command_buffers.iter().enumerate() {
        unsafe {
            device.begin_command_buffer(command_buffer, &begin_info)?;
            let single_handle_size_aligned = aligned_size(
                ray_tracing_pipeline_properties.shader_group_handle_size,
                ray_tracing_pipeline_properties.shader_group_handle_alignment,
            ) as u64;
            let base_handle_size_aligned = aligned_size(
                ray_tracing_pipeline_properties.shader_group_handle_size,
                ray_tracing_pipeline_properties.shader_group_base_alignment,
            ) as u64;

            let raygen_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
                device_address: raygen_shader_binding_table.get_device_address(device),
                stride: base_handle_size_aligned,
                size: base_handle_size_aligned,
            };
            let miss_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
                device_address: miss_shader_binding_table.get_device_address(device),
                stride: single_handle_size_aligned,
                size: base_handle_size_aligned,
            };
            let hit_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
                device_address: hit_shader_binding_table.get_device_address(device),
                stride: single_handle_size_aligned,
                size: base_handle_size_aligned,
            };

            let callable_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR::default();

            // Dispatch the ray tracing commands
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline,
            );
            let descriptor_sets_to_bind = [descriptor_sets[i]];
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline_layout,
                0,
                &descriptor_sets_to_bind,
                &[],
            );
            rt_device.cmd_trace_rays(
                command_buffer,
                &raygen_shader_sbt_entry,
                &miss_shader_sbt_entry,
                &hit_shader_sbt_entry,
                &callable_shader_sbt_entry,
                width,
                height,
                1,
            );

            device.end_command_buffer(command_buffer)?;
        }
    }
    log::trace!("Recorded commands {:?}", command_buffers);
    Ok(())
}

fn create_empty_commands(
    device: &ash::Device,
    queue_family_indices: structures::QueueFamilyIndices,
) -> Result<(Destructor<vk::CommandPool>, [vk::CommandBuffer; 2])> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_indices.compute_family.unwrap().0);
    let command_pool = Destructor::new(
        device,
        unsafe { device.create_command_pool(&pool_info, None)? },
        device.fp_v1_0().destroy_command_pool,
    );

    let command_buffers = core::create_command_buffers(device, command_pool.get(), 2)?;
    let commands = [command_buffers[0], command_buffers[1]];
    Ok((command_pool, commands))
}

fn create_buffer_references(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    allocator: &vk_mem::Allocator,
    data: &Vec<primitives::PrimitiveAddresses>,
) -> Result<vulkan::Buffer> {
    log::trace!("Creating buffer references");
    vulkan::Buffer::new_populated_staged(
        device,
        command_pool,
        queue,
        allocator,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        data.as_ptr(),
        data.len(),
    )
}

fn create_materials(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    allocator: &vk_mem::Allocator,
    add_material: &mpsc::Receiver<material::Material>,
) -> Result<vulkan::Buffer> {
    log::trace!("Creating materials");

    let mut materials = Vec::with_capacity(100);

    // materials.push(material::Material::new_basic(Vec3::new(1.0, 0.0, 0.0), 0.));

    loop {
        match add_material.try_recv() {
            // TODO: Material IDs so we can modify or remove materials
            Ok(v) => materials.push(v),
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(anyhow!("Material channel disconnected"))
            }
        }
    }

    let material_buffer = vulkan::Buffer::new_populated_staged(
        device,
        command_pool,
        queue,
        allocator,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        materials.as_ptr(),
        materials.len(),
    )?;
    log::trace!("Materials created");
    Ok(material_buffer)
}

fn add_aabb<T: Addressable + Extrema>(
    allocator: &vk_mem::Allocator,
    device: &ash::Device,
    as_device: &ash::khr::acceleration_structure::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    blass: &mut Vec<AccelerationStructure>,
    primitive_addresses: &mut Vec<primitives::PrimitiveAddresses>,
    aabbs: &mut Vec<AABB>,
    aabbs_buffers: &mut Vec<vulkan::Buffer>,
    obj: &T,
) -> Result<()> {
    aabbs.push(primitives::AABB::new(obj));
    primitive_addresses.push(obj.get_addresses(device)?);
    aabbs_buffers.push(vulkan::Buffer::new_populated_staged(
        device,
        command_pool,
        queue,
        allocator,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        aabbs.last().unwrap(),
        1,
    )?);
    blass.push(create_bottom_level_acceleration_structure(
        device,
        allocator,
        as_device,
        command_pool,
        queue,
        vk::GeometryTypeKHR::AABBS,
        None,
        Some(&aabbs_buffers.last().unwrap()),
        Some(1),
    )?);
    Ok(())
}

fn create_blas(
    allocator: &vk_mem::Allocator,
    device: &ash::Device,
    as_device: &ash::khr::acceleration_structure::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    update_as: &mpsc::Receiver<physics::UpdateScene>,
) -> Result<(
    HashMap<u64, (Primitive, usize)>,
    Vec<AccelerationStructure>,
    Vec<vk::AccelerationStructureInstanceKHR>,
    Vec<primitives::PrimitiveAddresses>,
    Vec<vulkan::Buffer>, // AABBs
)> {
    log::trace!("Creating BLASs and objects");
    let mut blass = vec![];
    let mut blas_instances = vec![];
    let mut primitive_addresses = vec![];

    let mut aabbs = vec![];
    let mut aabbs_buffers = vec![];

    // Map the IDs of objects to their position in the blass vector and store the object
    let mut primitive_map: HashMap<u64, (Primitive, usize)> = HashMap::new();

    // TODO: Somehow use ray.update instead?
    loop {
        // Create objects that the instances will refer to
        match update_as.try_recv() {
            Ok(physics::UpdateScene::Add(id, primitives::Primitive::Model(mut o))) => {
                log::trace!("{} - Adding a model", line!());
                o.allocate(allocator, device, command_pool, queue)?;
                primitive_addresses.push(o.get_addresses(device)?);
                blass.push(create_bottom_level_acceleration_structure(
                    device,
                    allocator,
                    as_device,
                    command_pool,
                    queue,
                    vk::GeometryTypeKHR::TRIANGLES,
                    Some(&o),
                    None,
                    None,
                )?);
                primitive_map.insert(id, (primitives::Primitive::Model(o), blass.len() - 1));
            }
            Ok(physics::UpdateScene::Add(id, primitives::Primitive::Lentil(mut o))) => {
                log::trace!("{} - Adding a lentil", line!());
                o.allocate(allocator, device, command_pool, queue)?;
                primitive_map.insert(id, (primitives::Primitive::Lentil(o), blass.len() - 1));
                add_aabb(
                    allocator,
                    device,
                    as_device,
                    command_pool,
                    queue,
                    &mut blass,
                    &mut primitive_addresses,
                    &mut aabbs,
                    &mut aabbs_buffers,
                    &primitive_map.get(&id).unwrap().0,
                )?;
            }
            Ok(physics::UpdateScene::Add(id, primitives::Primitive::Sphere(mut o))) => {
                log::trace!("{} - Adding a sphere", line!());
                o.allocate(allocator, device, command_pool, queue)?;
                primitive_map.insert(id, (primitives::Primitive::Sphere(o), blass.len() - 1));
                add_aabb(
                    allocator,
                    device,
                    as_device,
                    command_pool,
                    queue,
                    &mut blass,
                    &mut primitive_addresses,
                    &mut aabbs,
                    &mut aabbs_buffers,
                    &primitive_map.get(&id).unwrap().0,
                )?;
            }
            Ok(physics::UpdateScene::AddInstance(instance_id, primitive_id, transformation)) => {
                log::trace!("{} - Adding an instance", line!());
                // TODO: How are we keeping track of instance_id?
                let (primitive, location_in_blas) = match primitive_map.get(&primitive_id) {
                    Some(v) => v,
                    None => {
                        return Err(anyhow!(
                            "Failed to find primitive {} in the blas",
                            primitive_id
                        ))
                    }
                };
                let primitive_type = match primitive {
                    &Primitive::Model(_) => ObjectType::Triangle,
                    &Primitive::Sphere(_) => ObjectType::Sphere,
                    &Primitive::Lentil(_) => ObjectType::Lentil,
                };
                // Create a Bottom Level Acceleration structure for each instance
                blas_instances.push(create_blas_instance(
                    as_device,
                    &mut blass,
                    *location_in_blas,
                    transformation,
                    primitive_type,
                )?);
            }
            Ok(physics::UpdateScene::Remove(_)) => log::warn!(
                "Asking to remove a primitive before we initial state is built - ignoring"
            ),
            Ok(physics::UpdateScene::RemoveInstance(_)) => log::warn!(
                "Asking to remove an instance before we initial state is built - ignoring"
            ),
            Ok(physics::UpdateScene::MoveInstance(_, _)) => {
                log::warn!("Asking to move an instance before we initial state is built - ignoring")
            }
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(anyhow!(
                    "Update Acceleration Structure channel disconnected"
                ))
            }
        }
    }

    Ok((
        primitive_map,
        blass,
        blas_instances,
        primitive_addresses,
        aabbs_buffers,
    ))
}
