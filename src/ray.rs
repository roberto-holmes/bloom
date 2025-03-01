use std::ffi::CString;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex, RwLock};

use anyhow::{anyhow, Result};
use ash::{vk, RawPtr};
use vk_mem::{self};

use crate::api::BloomAPI;
use crate::core::UniformBufferObject;
use crate::vec::Vec3;
use crate::vulkan::Destructor;
use crate::{core, structures, tools, vulkan, MAX_FRAMES_IN_FLIGHT};

pub fn thread(
    device: ash::Device,

    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,

    queue_family_indices: structures::QueueFamilyIndices,
    images: [vulkan::Image; 2],

    should_threads_die: Arc<RwLock<bool>>,
    notify_transfer_wait: mpsc::Receiver<u64>,
    notify_complete_frame: mpsc::Sender<u8>,
    transfer_semaphore: vk::Semaphore,

    width: u32,
    height: u32,

    api: Arc<Mutex<BloomAPI>>,
) {
    log::trace!("Creating thread");
    let mut allocator_create_info =
        vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
    allocator_create_info.vulkan_api_version = vk::API_VERSION_1_3;
    allocator_create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

    let allocator = match unsafe { vk_mem::Allocator::new(allocator_create_info) } {
        Ok(v) => v,
        Err(e) => {
            panic!("Failed to create an allocator: {:?}", e);
        }
    };

    let rt_device = ash::khr::ray_tracing_pipeline::Device::new(&instance, &device);
    let as_device = ash::khr::acceleration_structure::Device::new(&instance, &device);

    // TODO: Which of these do we want? (or both?)
    let mut ray_tracing_pipeline_properties =
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut device_properties =
        vk::PhysicalDeviceProperties2::default().push(&mut ray_tracing_pipeline_properties);

    unsafe { instance.get_physical_device_properties2(physical_device, &mut device_properties) };

    let mut ray = match Ray::new(
        device,
        rt_device,
        as_device,
        ray_tracing_pipeline_properties,
        allocator,
        queue_family_indices,
        images,
        width,
        height,
        api,
    ) {
        Ok(v) => v,
        Err(e) => panic!("Failed to create ray tracing object: {}", e),
    };

    log::trace!("Ray tracing object has been constructed");

    // The latest value used by the timeline semaphore
    let mut current_timestamp = 0;
    // The latest value we have been geiven by Transfer that we cannot work until complete
    let mut current_transfer_timestamp = 0;
    // The index in the frames in flight that we are currently writing to
    let mut current_frame_index = 0;

    let mut first_run = true;

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

        if !first_run {
            match core::block_on_semaphore(
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

            // Compute sets the MSb of the frame so that transfer knows where it came from
            let channel_frame = current_frame_index as u8 | 0x80;
            // Send current frame to transfer thread so that it knows where it can get a valid frame from
            match notify_complete_frame.send(channel_frame) {
                Err(e) => {
                    log::error!("Failed to notify transfer channel: {e}")
                }
                Ok(()) => {}
            }
            current_frame_index += 1;
            current_frame_index %= MAX_FRAMES_IN_FLIGHT;
        }

        // Check if we need to wait for a transfer to complete
        match notify_transfer_wait.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {
                // Nothing to wait for, we can go on as planned
            }
            Ok(v) => {
                // We have been told to not do any work until a new timeline has been reached
                current_transfer_timestamp = v;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Failed to receive a notification from transfer");
                break;
            }
        }

        // TODO: Check for a resize event
        if ray.need_resize(width, height, current_frame_index) {
            match ray.resize(width, height, current_frame_index) {
                Err(e) => {
                    log::error!("Failed to resize output image: {e}")
                }
                _ => {}
            };
        }

        match ray.update_uniforms(current_frame_index) {
            Err(e) => log::error!("Failed to update uniform buffers: {e}"),
            Ok(()) => {}
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
            _ => {}
        };
        first_run = false;
        current_timestamp += 1;
    }
    log::warn!("Ending thread");
}

struct Ray<'a> {
    device: ash::Device,
    rt_device: ash::khr::ray_tracing_pipeline::Device,
    as_device: ash::khr::acceleration_structure::Device,

    ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,
    raygen_shader_binding_table: vulkan::Buffer,
    miss_shader_binding_table: vulkan::Buffer,
    hit_shader_binding_table: vulkan::Buffer,

    semaphore: Destructor<vk::Semaphore>,

    images: [vulkan::Image<'a>; 2],

    uniform_buffers: [vulkan::Buffer; 2],

    queue: vk::Queue,
    descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    pipeline_layout: Destructor<vk::PipelineLayout>,
    pipeline: Destructor<vk::Pipeline>,
    command_pool: Destructor<vk::CommandPool>,
    commands: [vk::CommandBuffer; 2],
    descriptor_pool: Destructor<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    bottom_level_as: AccelerationStructure,
    top_level_as: AccelerationStructure,

    allocator: vk_mem::Allocator,

    api: Arc<Mutex<BloomAPI>>,
}

impl<'a> Ray<'a> {
    fn new(
        device: ash::Device,
        rt_device: ash::khr::ray_tracing_pipeline::Device,
        as_device: ash::khr::acceleration_structure::Device,
        ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,
        allocator: vk_mem::Allocator,
        queue_family_indices: structures::QueueFamilyIndices,
        mut images: [vulkan::Image<'a>; 2],
        width: u32,
        height: u32,
        api: Arc<Mutex<BloomAPI>>,
    ) -> Result<Self> {
        log::trace!("Creating object");
        // Populates queue
        let queue = core::create_queue(&device, queue_family_indices.compute_family.unwrap());

        let (command_pool, commands) = create_empty_commands(&device, queue_family_indices)?;

        let uniform_buffers = [
            vulkan::Buffer::new_mapped(
                &allocator,
                size_of::<UniformBufferObject>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?,
            vulkan::Buffer::new_mapped(
                &allocator,
                size_of::<UniformBufferObject>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?,
        ];

        log::trace!("Creating bottom level acceleration structure");
        let bottom_level_as = create_bottom_level_acceleration_structure(
            &device,
            &allocator,
            &as_device,
            command_pool.get(),
            queue,
        )?;

        log::trace!("Creating top level Acceleration Structure");
        let top_level_as = create_top_level_acceleration_structure(
            &device,
            &allocator,
            &as_device,
            &bottom_level_as,
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

        log::trace!("Creating Descriptor Sets");
        let (descriptor_pool, descriptor_sets) = create_descriptor_sets(
            &device,
            descriptor_set_layout.get(),
            &top_level_as,
            &images,
            &uniform_buffers,
        )?;

        log::trace!("Building command buffers");
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            build_command_buffer(
                &device,
                &rt_device,
                &mut images[i],
                width,
                height,
                descriptor_sets[i],
                commands[i],
                &ray_tracing_pipeline_properties,
                raygen_shader_binding_table.get(),
                miss_shader_binding_table.get(),
                hit_shader_binding_table.get(),
                pipeline.get(),
                pipeline_layout.get(),
            )?;
        }

        log::trace!("Building semaphore");
        let semaphore = Destructor::new(
            &device,
            core::create_semaphore(&device)?,
            device.fp_v1_0().destroy_semaphore,
        );

        Ok(Self {
            device,
            allocator,
            semaphore,
            images,
            queue,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            command_pool,
            commands,
            descriptor_pool,
            descriptor_sets,
            rt_device,
            as_device,
            ray_tracing_pipeline_properties,
            raygen_shader_binding_table,
            miss_shader_binding_table,
            hit_shader_binding_table,
            api,
            uniform_buffers,
            bottom_level_as,
            top_level_as,
        })
    }
    pub fn need_resize(&self, width: u32, height: u32, frame_index: usize) -> bool {
        self.images[frame_index].is_correct_size(width, height)
    }
    pub fn resize(&mut self, width: u32, height: u32, frame_index: usize) -> Result<()> {
        build_command_buffer(
            &self.device,
            &self.rt_device,
            &mut self.images[frame_index],
            width,
            height,
            self.descriptor_sets[frame_index],
            self.commands[frame_index],
            &self.ray_tracing_pipeline_properties,
            self.raygen_shader_binding_table.get(),
            self.miss_shader_binding_table.get(),
            self.hit_shader_binding_table.get(),
            self.pipeline.get(),
            self.pipeline_layout.get(),
        )?;
        Ok(())
    }
    pub fn ray_trace(
        &self,
        current_frame_index: usize,
        transfer_semaphore: vk::Semaphore,
        transfer_timestamp_to_wait: u64,
        compute_timestamp_to_wait: u64,
        timestamp_to_signal: u64,
    ) -> Result<()> {
        let wait_timestamps = [transfer_timestamp_to_wait, compute_timestamp_to_wait];
        let wait_semaphores = [transfer_semaphore, self.semaphore.get()];
        let wait_stages = [
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        ];

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
    fn update_uniforms(&mut self, frame_index: usize) -> Result<()> {
        // let api = self.api.take().unwrap();
        let ubos = [self.api.lock().unwrap().uniform.clone()];
        // let ubos = [api.uniform.clone()];
        self.uniform_buffers[frame_index].populate_mapped(ubos.as_ptr(), ubos.len())
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
) -> Result<AccelerationStructure> {
    let vertices = vec![
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
    ];

    let indices: Vec<u32> = vec![0, 1, 2];

    #[rustfmt::skip]
    let transform_matrix = vk::TransformMatrixKHR { matrix: [
		1.0, 0.0, 0.0, 0.0, 
		0.0, 1.0, 0.0, 0.0, 
		0.0, 0.0, 1.0, 0.0],
	};

    // TODO: Replace host visibility with a staging buffer
    log::debug!("Creating vertex buffer");
    let vertex_buffer = vulkan::Buffer::new_populated(
        allocator,
        (vertices.len() * size_of::<Vec3>()) as vk::DeviceSize,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vertices.as_ptr(),
        vertices.len(),
    )?;
    log::debug!("Creating index buffer");
    let index_buffer = vulkan::Buffer::new_populated(
        allocator,
        (indices.len() * size_of::<u32>()) as vk::DeviceSize,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        indices.as_ptr(),
        indices.len(),
    )?;
    log::debug!("Creating transformation buffer");
    let transform_buffer = vulkan::Buffer::new_populated(
        allocator,
        size_of::<vk::TransformMatrixKHR>() as vk::DeviceSize,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        &transform_matrix,
        1,
    )?;

    let mut vertex_buffer_device_address = vk::DeviceOrHostAddressConstKHR::default();
    let mut index_buffer_device_address = vk::DeviceOrHostAddressConstKHR::default();
    let mut transform_buffer_device_address = vk::DeviceOrHostAddressConstKHR::default();

    vertex_buffer_device_address.device_address =
        get_buffer_device_address(device, vertex_buffer.get());
    index_buffer_device_address.device_address =
        get_buffer_device_address(device, index_buffer.get());
    transform_buffer_device_address.device_address =
        get_buffer_device_address(device, transform_buffer.get());

    // Build
    let mut as_geometries = [vk::AccelerationStructureGeometryKHR::default()
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)];
    as_geometries[0].geometry.triangles =
        vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vertex_buffer_device_address)
            .max_vertex(vertices.len() as u32)
            .vertex_stride(size_of::<Vec3>() as u64)
            .index_type(vk::IndexType::UINT32)
            .index_data(index_buffer_device_address)
            .transform_data(transform_buffer_device_address);

    let as_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(&as_geometries);

    let num_triangles = [1];
    let mut as_build_sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

    unsafe {
        as_device.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &as_build_geometry_info,
            &num_triangles,
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
    let scratch_buffer =
        ScratchBuffer::new(device, allocator, as_build_sizes_info.build_scratch_size)?;

    let mut as_build_geometry_infos = [vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .dst_acceleration_structure(bottom_level_as.handle)
        .geometries(&as_geometries)];
    as_build_geometry_infos[0].scratch_data.device_address = scratch_buffer.device_address;

    let as_build_range_infos =
        [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(num_triangles[0])];

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
    bottom_level_as: &AccelerationStructure,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
) -> Result<AccelerationStructure> {
    #[rustfmt::skip]
    let transform_matrix = vk::TransformMatrixKHR { matrix: [
		1.0, 0.0, 0.0, 0.0, 
		0.0, 1.0, 0.0, 0.0, 
		0.0, 0.0, 1.0, 0.0],
	};

    let as_instance = vk::AccelerationStructureInstanceKHR {
        transform: transform_matrix,
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: bottom_level_as.device_address,
        },
    };

    let instances_buffer = vulkan::Buffer::new_populated(
        allocator,
        size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        &as_instance,
        size_of::<vk::AccelerationStructureInstanceKHR>(),
    )?;

    let instance_data_device_address = vk::DeviceOrHostAddressConstKHR {
        device_address: get_buffer_device_address(device, instances_buffer.get()),
    };

    // The top level acceleration structure contains (bottom level) instance as the input geometry
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

    let primitive_count = [1];

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

    // Get the top acceleration structure's handle, which will be usedto setup its descriptor
    let as_device_address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
        .acceleration_structure(bottom_level_as.handle);
    top_level_as.device_address =
        unsafe { as_device.get_acceleration_structure_device_address(&as_device_address_info) };

    Ok(top_level_as)
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
    // let handle_size = ray_tracing_pipeline_properties.shader_group_handle_size as u64;
    let handle_size = ray_tracing_pipeline_properties.shader_group_base_alignment as u64; // TODO: Potentially revisit. This should be using the line above but it fails
    let handle_size_aligned = aligned_size(
        ray_tracing_pipeline_properties.shader_group_handle_size,
        ray_tracing_pipeline_properties.shader_group_handle_alignment,
    );
    let group_count = shader_groups.len() as u32;
    let sbt_size = (group_count * handle_size_aligned) as usize;
    let sbt_buffer_usage_flags = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

    // Raygen
    // Create binding table buffers for each shader type
    let raygen_shader_binding_table = vulkan::Buffer::new_generic(
        allocator,
        handle_size,
        vk_mem::MemoryUsage::Auto,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        sbt_buffer_usage_flags,
    )?;
    let miss_shader_binding_table = vulkan::Buffer::new_generic(
        allocator,
        handle_size,
        vk_mem::MemoryUsage::Auto,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        sbt_buffer_usage_flags,
    )?;
    let hit_shader_binding_table = vulkan::Buffer::new_generic(
        allocator,
        handle_size,
        vk_mem::MemoryUsage::Auto,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        sbt_buffer_usage_flags,
    )?;

    log::info!("Buffers created");
    // Copy the pipeline's shader handles into a host buffer
    let shader_handle_storage = unsafe {
        rt_device.get_ray_tracing_shader_group_handles(pipeline, 0, group_count, sbt_size)?
    };

    log::info!("Have ray tracing shader group handles");
    // Copy the shader handles from the host buffer to the binding tables
    unsafe {
        raygen_shader_binding_table
            .populate(shader_handle_storage.as_ptr(), handle_size as usize)?;
        log::info!("Populated raygen");
        miss_shader_binding_table.populate(
            shader_handle_storage
                .as_ptr()
                .offset(handle_size_aligned as isize),
            handle_size as usize,
        )?;
        log::info!("Populated miss");
        hit_shader_binding_table.populate(
            shader_handle_storage
                .as_ptr()
                .offset((handle_size_aligned * 2) as isize),
            handle_size as usize,
        )?;
        log::info!("Populated hit");
    }
    Ok((
        raygen_shader_binding_table,
        miss_shader_binding_table,
        hit_shader_binding_table,
    ))
}

fn create_descriptor_sets(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
    top_level_as: &AccelerationStructure,
    storage_images: &[vulkan::Image; 2],
    ubo: &[vulkan::Buffer; 2],
) -> Result<(Destructor<vk::DescriptorPool>, Vec<vk::DescriptorSet>)> {
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
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        },
    ];
    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32);
    let descriptor_pool = Destructor::new(
        device,
        unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None)? },
        device.fp_v1_0().destroy_descriptor_pool,
    );

    let set_layouts = [set_layout, set_layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool.get())
        .set_layouts(&set_layouts);

    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&allocate_info)? };

    let structures = [top_level_as.handle];
    for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
        let mut descriptor_acceleration_structure_info =
            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&structures);

        let image_descriptor = [vk::DescriptorImageInfo {
            image_view: storage_images[i].view(),
            image_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        }];

        let buffer_descriptor = [ubo[i].create_descriptor()];

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
            .image_info(&image_descriptor),
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                ..Default::default()
            }
            .buffer_info(&buffer_descriptor),
        ];
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }
    Ok((descriptor_pool, descriptor_sets))
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
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
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
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
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
    let main_function_name = CString::new("main").unwrap();
    let raygen_code = tools::read_shader_code(Path::new("shaders/spv/raygen.slang.spv"))?;
    let miss_code = tools::read_shader_code(Path::new("shaders/spv/miss.slang.spv"))?;
    let closest_hit_code = tools::read_shader_code(Path::new("shaders/spv/closest_hit.slang.spv"))?;

    let raygen_module = core::create_shader_module(device, &raygen_code)?;
    let miss_module = core::create_shader_module(device, &miss_code)?;
    let closest_hit_module = core::create_shader_module(device, &closest_hit_code)?;

    {
        // Ray generation group
        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(raygen_module.get())
                .name(&main_function_name),
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
                .name(&main_function_name),
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
                .name(&main_function_name),
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

    let ray_tracing_pipeline_create_infos = [vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(shader_stages.as_slice())
        .groups(&shader_groups)
        .max_pipeline_ray_recursion_depth(1)
        .layout(pipeline_layout.get())];

    let pipeline = unsafe {
        rt_device.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &ray_tracing_pipeline_create_infos,
            None,
        )
    };

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

fn build_command_buffer(
    device: &ash::Device,
    rt_device: &ash::khr::ray_tracing_pipeline::Device,
    storage_image: &mut vulkan::Image,
    width: u32,
    height: u32,
    descriptor_set: vk::DescriptorSet,
    command_buffer: vk::CommandBuffer,
    ray_tracing_pipeline_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    raygen_shader_binding_table: vk::Buffer,
    miss_shader_binding_table: vk::Buffer,
    hit_shader_binding_table: vk::Buffer,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
) -> Result<()> {
    if !storage_image.is_correct_size(width, height) {
        storage_image.resize(width, height)?;
        // Now need to update the descriptor to reference the new image
        let image_descriptor = [vk::DescriptorImageInfo {
            image_view: storage_image.view(),
            image_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        }];

        let descriptor_writes = [vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 1,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ..Default::default()
        }
        .image_info(&image_descriptor)];
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    let begin_info = vk::CommandBufferBeginInfo::default();

    // for &command_buffer in command_buffers {
    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;
        let handle_size_aligned = aligned_size(
            ray_tracing_pipeline_properties.shader_group_handle_size,
            ray_tracing_pipeline_properties.shader_group_base_alignment,
        ) as u64;

        let raygen_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
            device_address: get_buffer_device_address(device, raygen_shader_binding_table),
            stride: handle_size_aligned,
            size: handle_size_aligned,
        };
        let miss_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
            device_address: get_buffer_device_address(device, miss_shader_binding_table),
            stride: handle_size_aligned,
            size: handle_size_aligned,
        };
        let hit_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
            device_address: get_buffer_device_address(device, hit_shader_binding_table),
            stride: handle_size_aligned,
            size: handle_size_aligned,
        };

        let callable_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR::default();

        // Dispatch the ray tracing commands
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline,
        );
        let descriptor_sets = [descriptor_set];
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline_layout,
            0,
            &descriptor_sets,
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

        // TODO: Channel stuff from compute
        device.end_command_buffer(command_buffer)?;
    }
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
