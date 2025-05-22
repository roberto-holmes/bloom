use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::sync::{mpsc, Arc, RwLock};

use anyhow::{anyhow, Result};
use ash::vk::AccelerationStructureInstanceKHR;
use ash::{vk, RawPtr};
use hecs::{Entity, World};
use vk_mem;
use winit::dpi::PhysicalSize;

use crate::material::Material;
use crate::primitives::{
    Addressable, ObjectType, Objectionable, Primitive, PrimitiveAddresses, AABB,
};
use crate::uniforms::{self, UniformBufferObject};
use crate::vec::Vec3;
use crate::vulkan::Destructor;
use crate::{api, core, primitives, structures, tools, transfer, vulkan, MAX_FRAMES_IN_FLIGHT};

const RESERVED_SIZE: usize = 100;

pub enum TransferCommand {
    Pause(u64),
}

#[derive(Debug, Default)]
pub struct Update {
    pub current_frame_index: u8,
    pub accumulated_frames: u32,
}

#[derive(Debug, Clone, Copy)]
enum TlasProcess {
    Build,
    Update,
}

pub fn thread(
    world: Arc<RwLock<World>>,

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
    notify_complete_frame: Arc<RwLock<Update>>,
    uniform: mpsc::Sender<uniforms::Event>,
    latest_frame_index: Arc<RwLock<usize>>,
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

    let mut is_minimised;

    let mut accumulated_frames = 0;
    let mut final_accumulation = false;

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

        // We will consider either axis being zero as being minimised because we don't need to render anything
        is_minimised = size.width == 0 || size.height == 0;

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

            if final_accumulation {
                accumulated_frames = 0;
                final_accumulation = false;
            }

            accumulated_frames += 1;

            // Send current frame to transfer thread so that it knows where it can get a valid frame from
            match notify_complete_frame.write() {
                Ok(mut v) => {
                    *v = Update {
                        current_frame_index,
                        accumulated_frames,
                    }
                }
                Err(e) => {
                    log::error!("Ray failed to notify transfer channel: {e}");
                    break;
                }
            }
            if let Err(e) = ray.uniform.send(uniforms::Event::RayTick) {
                log::error!("Ray failed to update Uniform's frame count: {e}");
                break;
            };

            current_frame_index += 1;
            current_frame_index %= MAX_FRAMES_IN_FLIGHT as u8;
        }

        if !is_minimised {
            while ray.need_resize(*size) {
                log::debug!("Received resize to {}x{}", size.width, size.height);
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
        }

        // Check if transfer is actively using our images and we need to wait for it to finish
        // or if the window has been resized and we need to change our images
        match transfer_commands.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {
                // Nothing to wait for, we can go on as planned
                // log::debug!("Can go for another accumulation");
            }
            Ok(TransferCommand::Pause(v)) => {
                // log::debug!("Pausing until {v}");
                // We have been told to not do any work until a new timeline has been reached
                current_transfer_timestamp = v;
                // Schedule frame count for resetting as we only accumulate between viewport frames
                final_accumulation = true;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Failed to receive a notification from transfer");
                break;
            }
        }

        if let Err(e) = ray.update(&world) {
            log::error!("Failed to check for updates: {e}");
            break;
        }

        if is_minimised {
            continue;
        }

        // Perform a render
        // TODO: Measure time taken
        match ray.ray_trace(
            current_frame_index as usize,
            transfer_semaphore,
            current_transfer_timestamp,
            current_timestamp,
            current_timestamp + 1,
        ) {
            Err(e) => {
                log::error!("Failed to perform compute: {e}")
            }
            Ok(()) => match latest_frame_index.write() {
                Ok(mut v) => *v = current_frame_index as usize,
                Err(e) => log::error!("Failed to send latest frame index to uniforms: {e}"),
            },
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

    material_location_map: HashMap<Entity, usize>,
    instance_location_map: HashMap<Entity, usize>,
    /// Map the IDs of objects to their position in the blass vector and store the object
    primitive_location_map: HashMap<Entity, (ObjectType, usize)>,
    primitive_addresses: Vec<PrimitiveAddresses>,
    materials_buffer: vulkan::Buffer,
    aabbs_buffers: Vec<vulkan::Buffer>,
    buffer_references: vulkan::Buffer,
    instances_buffer: vulkan::Buffer,

    materials: Vec<Material>,
    aabbs: Vec<AABB>,
    blass: Vec<AccelerationStructure>,
    blas_instances: Vec<AccelerationStructureInstanceKHR>,
    tlas: AccelerationStructure,

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

        uniform: mpsc::Sender<uniforms::Event>,
    ) -> Result<Self> {
        log::trace!("Creating object");
        // Populates queue
        let queue = core::create_queue(&device, queue_family_indices.compute_family.unwrap());

        let (command_pool, commands) = create_empty_commands(&device, queue_family_indices)?;

        let materials_buffer = vulkan::Buffer::new_gpu(
            &allocator,
            (size_of::<Material>() * RESERVED_SIZE) as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        // Get a reference to where all the parts of each model are on the GPU
        let buffer_references = vulkan::Buffer::new_gpu(
            &allocator,
            (size_of::<primitives::PrimitiveAddresses>() * RESERVED_SIZE) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let instances_buffer = vulkan::Buffer::new_generic(
            &allocator,
            (size_of::<vk::AccelerationStructureInstanceKHR>() * RESERVED_SIZE) as u64,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        let blas_instances = Vec::with_capacity(RESERVED_SIZE);

        let tlas = AccelerationStructure::new(&as_device);

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

        let mut materials = Vec::with_capacity(RESERVED_SIZE);
        // Set a default material for primitives that are missing one or have not been properly set up
        materials.push(Material::new_basic(Vec3::new(1.0, 0.1, 0.7), 0.));

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
            tlas,
            buffer_references,
            materials,
            aabbs: Vec::with_capacity(RESERVED_SIZE),
            blass: Vec::with_capacity(RESERVED_SIZE),
            blas_instances,
            materials_buffer,
            primitive_addresses: Vec::with_capacity(RESERVED_SIZE),
            aabbs_buffers: Vec::with_capacity(RESERVED_SIZE),
            instance_location_map: HashMap::with_capacity(RESERVED_SIZE),
            primitive_location_map: HashMap::with_capacity(RESERVED_SIZE),
            material_location_map: HashMap::with_capacity(RESERVED_SIZE),
            uniform,
            instances_buffer,
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
        self.create_top_level_acceleration_structure(TlasProcess::Build)?;
        self.images = Some(core::create_storage_image_pair(
            &self.device,
            &self.instance,
            &self.allocator,
            self.physical_device,
            self.command_pool.get(),
            self.queue,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            size,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        )?);
        self.descriptor_sets = Some(create_descriptor_sets(
            &self.device,
            self.descriptor_set_layout.get(),
            self.descriptor_pool.get(),
            &self.tlas,
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
                    image_view: self.images.as_ref().unwrap()[(i + 1) % 2].view(),
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                }];
                let write_image_descriptor = [vk::DescriptorImageInfo {
                    image_view: self.images.as_ref().unwrap()[i].view(),
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

    fn add_primitive(&mut self, entity: Entity, primitive: &mut Primitive) -> Result<bool> {
        let mut primitives_changed = false;
        match primitive {
            Primitive::Model(ref mut o) if !self.primitive_location_map.contains_key(&entity) => {
                log::info!("Adding model {:?}", entity);
                // TODO: Find the index of each material in the material vector
                o.set_materials(&self.material_location_map);
                o.allocate(
                    &self.allocator,
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                )?;
                self.primitive_addresses
                    .push(o.get_addresses(&self.device)?);
                self.blass.push(create_bottom_level_acceleration_structure(
                    &self.device,
                    &self.allocator,
                    &self.as_device,
                    self.command_pool.get(),
                    self.queue,
                    vk::GeometryTypeKHR::TRIANGLES,
                    Some(&o),
                    None,
                    None,
                )?);
                if self
                    .primitive_location_map
                    .insert(entity, (ObjectType::Triangle, self.blass.len() - 1))
                    != None
                {
                    log::warn!("Failed to add model to primitive map");
                }
                primitives_changed = true;
            }
            Primitive::Lentil(ref mut o) if !self.primitive_location_map.contains_key(&entity) => {
                log::info!("Adding lentil {:?}: {:?}", entity, o);
                o.set_materials(&self.material_location_map);
                o.allocate(
                    &self.allocator,
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                )?;
                self.primitive_location_map
                    .insert(entity, (ObjectType::Lentil, self.blass.len()));
                self.add_aabb(AABB::new(o), o.get_addresses(&self.device)?)?;
                primitives_changed = true;
            }
            Primitive::Sphere(ref mut o) if !self.primitive_location_map.contains_key(&entity) => {
                log::info!("Adding sphere {:?}: {:?}", entity, o);
                o.set_materials(&self.material_location_map);
                o.allocate(
                    &self.allocator,
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                )?;
                self.primitive_location_map
                    .insert(entity, (ObjectType::Sphere, self.blass.len()));
                self.add_aabb(AABB::new(o), o.get_addresses(&self.device)?)?;
                primitives_changed = true;
            }
            _ => {
                // Primitive is already saved
                // TODO: Update primitive if it has changed (Hash the state somehow?)
            }
        }
        Ok(primitives_changed)
    }

    pub fn update(&mut self, world: &Arc<RwLock<World>>) -> Result<()> {
        let mut need_new_references = false;
        let mut primitives_changed = false;
        let mut need_rebuild: Option<TlasProcess> = None;

        {
            let mut w = world.write().unwrap();
            // TODO: Use some sort of `new` component to only add primitives/materials that have changed
            for (entity, material) in w.query_mut::<&Material>() {
                if self.material_location_map.contains_key(&entity) {
                    continue;
                }
                // If the material is new, remake the buffer with all the materials
                self.materials.push(*material);
                self.material_location_map
                    .insert(entity, self.materials.len() - 1);
                // Create a new buffer with the old data
                self.materials_buffer = vulkan::Buffer::new_populated_staged(
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                    &self.allocator,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    self.materials.as_ptr(),
                    self.materials.len(),
                    self.materials.len(),
                )?;
                // Update reference that shader sees
                need_new_references = true;
                // TODO: Remove materials that no longer exist
            }

            let mut primitives = Vec::with_capacity(RESERVED_SIZE);
            for (entity, primitive) in w.query::<&mut Primitive>().iter() {
                // log::debug!("Initialising primitive {:?}: {:?}", entity, primitive);
                primitives.push(entity);
                if self.add_primitive(entity, primitive)? {
                    primitives_changed = true;
                }
            }
            // Look for primitives that we have instantiated but are no longer in the world.
            let mut orphaned_primitives = primitives.clone();
            // Reverse loop to avoid copies as we remove items
            for (i, p) in primitives.iter().enumerate().rev() {
                if self.primitive_location_map.contains_key(p) {
                    orphaned_primitives.remove(i);
                }
            }

            // TODO: Do something with `orphaned_primitives` (Remove from primitive addresses and blass)

            for (entity, (instance, ori)) in w.query_mut::<(&api::Instance, &api::Orientation)>() {
                let transform = ori.transformation * instance.base_transform;
                match self.instance_location_map.get(&entity) {
                    None => {
                        // We need to create a new BLAS
                        log::trace!("Adding an instance: {:?}", instance);
                        // TODO: How are we keeping track of instance_id?
                        let (primitive_type, location_in_blas) =
                            match self.primitive_location_map.get(&instance.primitive) {
                                Some(v) => v,
                                None => {
                                    return Err(anyhow!(
                                        "Failed to find primitive {:?} in the blas",
                                        instance.primitive
                                    ))
                                }
                            };

                        self.instance_location_map
                            .insert(entity, self.blas_instances.len());
                        // Create a Bottom Level Acceleration structure for each instance
                        self.blas_instances.push(create_blas_instance(
                            &self.as_device,
                            &mut self.blass,
                            *location_in_blas,
                            transform,
                            *primitive_type,
                        )?);
                        need_rebuild = Some(TlasProcess::Build);
                    }
                    Some(&instance_index) => {
                        need_rebuild = Some(TlasProcess::Update);
                        #[rustfmt::skip]
                    let transform_matrix = vk::TransformMatrixKHR {matrix:[
                        transform.x[0], transform.y[0], transform.z[0], transform.w[0],
                        transform.x[1], transform.y[1], transform.z[1], transform.w[1],
                        transform.x[2], transform.y[2], transform.z[2], transform.w[2]],
                    };
                        self.blas_instances[instance_index].transform = transform_matrix;
                    }
                }
            }
        }

        if primitives_changed {
            if self
                .buffer_references
                .check_available_space::<PrimitiveAddresses>(self.primitive_addresses.len())
            {
                self.buffer_references.populate_staged(
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                    &self.allocator,
                    self.primitive_addresses.as_ptr(),
                    self.primitive_addresses.len(),
                )?;
            } else {
                // Grow the buffer by creating a new one (overwriting the old one and causing it to drop and cleanup)
                self.buffer_references = vulkan::Buffer::new_populated_staged(
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                    &self.allocator,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    self.primitive_addresses.as_ptr(),
                    self.primitive_addresses.len(),
                    self.primitive_addresses.len() + RESERVED_SIZE,
                )?;
                need_new_references = true;
            }
        }
        match need_rebuild {
            None => {}
            Some(v) => {
                self.create_top_level_acceleration_structure(v)?;
            }
        }

        if let Some(TlasProcess::Build) = need_rebuild {
            self.rebuild_references()?;
        } else if need_new_references {
            self.rebuild_references()?;
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
        Ok(())
    }

    fn create_top_level_acceleration_structure(&mut self, process: TlasProcess) -> Result<()> {
        match process {
            TlasProcess::Build => {
                log::trace!(
                    "Building a TLAS with {} instances",
                    self.blas_instances.len()
                );
                if !self
                    .instances_buffer
                    .check_available_space::<AccelerationStructureInstanceKHR>(
                        self.blas_instances.len(),
                    )
                {
                    self.instances_buffer = vulkan::Buffer::new_generic(
                        &self.allocator,
                        (size_of::<vk::AccelerationStructureInstanceKHR>()
                            * (self.blas_instances.len() + RESERVED_SIZE))
                            as u64,
                        vk_mem::MemoryUsage::Auto,
                        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )?;
                }
            }
            TlasProcess::Update => {
                log::trace!(
                    "Updating a TLAS with {} instances",
                    self.blas_instances.len()
                );
                if self.instances_buffer.get_populated_bytes()
                    != (size_of::<vk::AccelerationStructureInstanceKHR>()
                        * self.blas_instances.len()) as u64
                {
                    log::warn!(
                        "Updating the TLAS with a different number of BLAS {} != {}x{}={}",
                        self.instances_buffer.get_populated_bytes(),
                        size_of::<vk::AccelerationStructureInstanceKHR>(),
                        self.blas_instances.len(),
                        (size_of::<vk::AccelerationStructureInstanceKHR>()
                            * self.blas_instances.len())
                    );
                    return Ok(());
                }
            }
        }
        log::trace!(
            "Populating instances with {}x{}={} Bytes",
            size_of::<vk::AccelerationStructureInstanceKHR>(),
            self.blas_instances.len(),
            (size_of::<vk::AccelerationStructureInstanceKHR>() * self.blas_instances.len())
        );
        self.instances_buffer
            .populate(self.blas_instances.as_ptr(), self.blas_instances.len())?;

        let instance_data_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: get_buffer_device_address(&self.device, self.instances_buffer.get()),
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
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            ..Default::default()
        };
        as_build_geometry_info = as_build_geometry_info.geometries(&as_geometries);

        let primitive_count = [self.blas_instances.len() as u32];

        let mut as_build_sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.as_device.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &as_build_geometry_info,
                &primitive_count,
                &mut as_build_sizes_info,
            )
        };

        match process {
            TlasProcess::Build => {
                self.tlas
                    .grow_buffer(&self.allocator, as_build_sizes_info)?;
                let as_create_info = vk::AccelerationStructureCreateInfoKHR {
                    buffer: self.tlas.get_buffer(),
                    size: as_build_sizes_info.acceleration_structure_size,
                    ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                    ..Default::default()
                };
                let old_tlas = self.tlas.handle;
                self.tlas.handle = unsafe {
                    self.as_device
                        .create_acceleration_structure(&as_create_info, None)
                }?;
                unsafe {
                    self.as_device
                        .destroy_acceleration_structure(old_tlas, None)
                };
            }
            TlasProcess::Update => {}
        }
        // The actual build process starts here

        // Create a scratch buffer as a temporary storage for the acceleration structure build
        log::trace!(
            "Scratch buffer needs to be {} B",
            as_build_sizes_info.build_scratch_size
        );
        let scratch_buffer = ScratchBuffer::new(
            &self.device,
            &self.allocator,
            as_build_sizes_info.build_scratch_size,
        )?;

        let as_build_geometry_infos = [vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            mode: match process {
                TlasProcess::Build => vk::BuildAccelerationStructureModeKHR::BUILD,
                TlasProcess::Update => vk::BuildAccelerationStructureModeKHR::UPDATE,
            },
            src_acceleration_structure: match process {
                TlasProcess::Build => vk::AccelerationStructureKHR::default(),
                TlasProcess::Update => self.tlas.handle,
            },
            dst_acceleration_structure: self.tlas.handle,
            scratch_data: vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.device_address,
            },
            ..Default::default()
        }
        .geometries(&as_geometries)];

        let as_build_range_infos = [vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(primitive_count[0])];

        // Build the acceleration structure on the device via a one-time command buffer submission
        // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
        let command_buffer =
            core::begin_single_time_commands(&self.device, self.command_pool.get())?;
        log::trace!("Actually building TLAS");
        unsafe {
            self.as_device.cmd_build_acceleration_structures(
                command_buffer,
                &as_build_geometry_infos,
                &[&as_build_range_infos],
            );
        }
        core::end_single_time_command(
            &self.device,
            self.command_pool.get(),
            self.queue,
            command_buffer,
        )?;
        log::trace!("TLAS built");

        Ok(())
    }

    fn add_aabb(&mut self, aabb: AABB, addresses: PrimitiveAddresses) -> Result<()> {
        self.aabbs.push(aabb);
        self.primitive_addresses.push(addresses);
        self.aabbs_buffers
            .push(vulkan::Buffer::new_populated_staged(
                &self.device,
                self.command_pool.get(),
                self.queue,
                &self.allocator,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                self.aabbs.last().unwrap(),
                1,
                1,
            )?);
        self.blass.push(create_bottom_level_acceleration_structure(
            &self.device,
            &self.allocator,
            &self.as_device,
            self.command_pool.get(),
            self.queue,
            vk::GeometryTypeKHR::AABBS,
            None,
            Some(&self.aabbs_buffers.last().unwrap()),
            Some(1),
        )?);
        Ok(())
    }

    fn rebuild_references(&mut self) -> Result<()> {
        // Update descriptor sets with new TLAS and potentially new buffers
        let structures = [self.tlas.handle];
        for descriptor_set in self.descriptor_sets.unwrap() {
            let mut descriptor_acceleration_structure_info =
                vk::WriteDescriptorSetAccelerationStructureKHR::default()
                    .acceleration_structures(&structures);
            let scene_descriptor = [self.buffer_references.create_descriptor()];
            let material_descriptor = [self.materials_buffer.create_descriptor()];

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
            unsafe { self.device.update_descriptor_sets(&descriptor_writes, &[]) };
        }
        // Create new commands with new descriptor sets
        build_command_buffers(
            &self.device,
            &self.rt_device,
            self.images.as_ref().unwrap()[0].width,
            self.images.as_ref().unwrap()[0].height,
            self.descriptor_sets.unwrap(),
            self.commands,
            &self.ray_tracing_pipeline_properties,
            &self.raygen_shader_binding_table,
            &self.miss_shader_binding_table,
            &self.hit_shader_binding_table,
            self.pipeline.get(),
            self.pipeline_layout.get(),
        )?;
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
        let _buffer = vulkan::Buffer::new_aligned(
            allocator,
            size,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::empty(),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            128, // TODO: Get programatically from VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
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
            build_size_info.acceleration_structure_size * 2,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?);
        Ok(())
    }
    pub fn grow_buffer(
        &mut self,
        allocator: &vk_mem::Allocator,
        build_size_info: vk::AccelerationStructureBuildSizesInfoKHR,
    ) -> Result<bool> {
        match self.buffer.as_mut() {
            None => {
                self.create_buffer(allocator, build_size_info)?;
                return Ok(true);
            }
            Some(b) => {
                // If the current buffer doesn't have enough space for the new acceleration structure,
                // create a new buffer to overwrite it
                if !b.check_available_space::<u8>(
                    build_size_info.acceleration_structure_size as usize,
                ) {
                    self.create_buffer(allocator, build_size_info)?;
                    return Ok(true);
                }
            }
        }
        // The current buffer is good enough
        Ok(false)
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
        log::trace!("Dropping Acceleration Structure {:?}", self.handle);
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
    // TODO: Replace options with enum?
    obj: Option<&primitives::model::Model>,
    aabb: Option<&vulkan::Buffer>,
    prim_count: Option<u32>,
) -> Result<AccelerationStructure> {
    log::trace!("Creating a BLAS");
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
    // TODO: Set flags to vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE | PREFER_FAST_BUILD if this object will be modified
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
    // TODO: Set flags to vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE | PREFER_FAST_BUILD if this object will be modified
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
    tlas: &AccelerationStructure,
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

    let structures = [tlas.handle];
    for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
        let mut descriptor_acceleration_structure_info =
            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&structures);

        let read_image_descriptor = [vk::DescriptorImageInfo {
            image_view: storage_images[(i + 1) % 2].view(),
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
