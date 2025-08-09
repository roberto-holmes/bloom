use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::path::Path;
use std::sync::{mpsc, Arc, RwLock};

use anyhow::{anyhow, Result};
use ash::vk;
use hecs::{Entity, World};
use vk_mem;
use winit::dpi::PhysicalSize;

use crate::core::create_commands_flight_frames;
use crate::material::{Material, MaterialData};
use crate::oceans::{Ocean, FFT_IMAGES};
use crate::physics::Physics;
use crate::primitives::{
    Addressable, ObjectType, Objectionable, Primitive, PrimitiveAddresses, AABB,
};
use crate::ray::acceleration_structure::AccelerationStructure;
use crate::ray::instance_buffer::InstanceBuffer;
// use crate::ray::texture_buffer::TextureBuffer;
use crate::uniforms;
use crate::vec::Vec3;
use crate::vulkan::Destructor;
use crate::{api, core, primitives, structures, sync, tools, vulkan, MAX_FRAMES_IN_FLIGHT};

mod acceleration_structure;
pub mod instance_buffer;
// mod texture_buffer;

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

#[derive(Debug, Default)]
#[repr(C)]
struct PushConstants {
    pub ubo: vk::DeviceAddress,
    pub scene: vk::DeviceAddress,
    pub materials: vk::DeviceAddress,
    pub frame_index: u32,
}

impl PushConstants {
    pub fn as_slice(&self) -> &[u8; size_of::<Self>()] {
        unsafe { &*(self as *const Self as *const [u8; size_of::<Self>()]) }
    }
}

struct DescriptorData {
    layout: Destructor<vk::DescriptorSetLayout>,
    buffer: vulkan::Buffer,
    offset: vk::DeviceSize,
}

impl DescriptorData {
    fn new(
        binding: u32,
        allocator: &vk_mem::Allocator,
        device: &ash::Device,
        db_device: &ash::ext::descriptor_buffer::Device,
        descriptor_type: vk::DescriptorType,
        descriptor_count: u32,
        stage_flags: vk::ShaderStageFlags,
        descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    ) -> Result<Self> {
        let layout = core::create_bindless_descriptor_set_layout(
            device,
            binding,
            descriptor_type,
            descriptor_count,
            stage_flags,
        )?;
        // Compute and align the sizes of the sets so that we can compute the offsets
        let size = core::aligned_size(
            unsafe { db_device.get_descriptor_set_layout_size(layout.get()) },
            descriptor_buffer_properties.descriptor_buffer_offset_alignment,
        );
        // Get the offsets of the descriptor bindings of each set layout as they don't necessarily start at
        // the beginning of the buffer (driver could add metadata at the start or something)
        let offset = unsafe { db_device.get_descriptor_set_layout_binding_offset(layout.get(), 0) };

        let buffer = vulkan::Buffer::new_aligned(
            allocator,
            size,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::MAPPED //? Does it need to be mapped or just HOST_VISIBLE?
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM, // TODO: Random, sequential, or neither?
            // | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE, // https://www.khronos.org/blog/vk-ext-descriptor-buffer says we can use this flag if we "ensure multiple writes are written contiguously"
            vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            descriptor_buffer_properties.descriptor_buffer_offset_alignment, // TODO: Does this buffer need to be aligned?
            "Descriptor Data",
        )?;
        Ok(Self {
            layout,
            buffer,
            offset,
        })
    }

    fn get_descriptor_image(
        &mut self,
        db_device: &ash::ext::descriptor_buffer::Device,
        image_view: vk::ImageView,
        offset: usize,
        descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    ) -> Result<()> {
        // Create a new descriptor pointing at the new image
        let image_descriptor = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: image_view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        let image_descriptor_info = vk::DescriptorGetInfoEXT {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            data: vk::DescriptorDataEXT {
                p_storage_image: &image_descriptor,
            },
            ..Default::default()
        };
        log::info!(
            "Offsetting data by {}",
            self.offset
                + (descriptor_buffer_properties.storage_image_descriptor_size * offset)
                    as vk::DeviceSize
        );
        // Copy the desciptor into the descriptor buffer
        unsafe {
            let mapped_data = self.buffer.get_mapped_data(
                descriptor_buffer_properties.storage_image_descriptor_size as vk::DeviceSize,
                self.offset
                    + (descriptor_buffer_properties.storage_image_descriptor_size * offset)
                        as vk::DeviceSize,
            )?;

            db_device.get_descriptor(&image_descriptor_info, mapped_data);
        };
        Ok(())
    }

    fn get_descriptor_tlas(
        &mut self,
        db_device: &ash::ext::descriptor_buffer::Device,
        tlas_handle: vk::DeviceAddress,
        offset: usize,
        descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    ) -> Result<()> {
        let image_descriptor_info = vk::DescriptorGetInfoEXT {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            data: vk::DescriptorDataEXT {
                acceleration_structure: tlas_handle,
            },
            ..Default::default()
        };
        // Copy the desciptor into the descriptor buffer
        unsafe {
            let mapped_data = self.buffer.get_mapped_data(
                descriptor_buffer_properties.acceleration_structure_descriptor_size
                    as vk::DeviceSize,
                self.offset
                    + (descriptor_buffer_properties.acceleration_structure_descriptor_size * offset)
                        as vk::DeviceSize,
            )?;

            db_device.get_descriptor(&image_descriptor_info, mapped_data);
        };
        Ok(())
    }
}

pub fn thread(
    world: Arc<RwLock<World>>,

    device: ash::Device,

    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,

    queue_family_indices: structures::QueueFamilyIndices,
    uniform_buffers: [vk::DeviceAddress; 2],

    should_threads_die: Arc<RwLock<bool>>,
    transfer_semaphore: vk::Semaphore,
    transfer_sender: mpsc::Sender<sync::ResizedSource>,
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
    let db_device = ash::ext::descriptor_buffer::Device::new(&instance, &device);

    let mut descriptor_buffer_properties =
        vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
    let mut acceleration_structure_properties =
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut ray_tracing_pipeline_properties =
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut device_properties = vk::PhysicalDeviceProperties2::default()
        .push(&mut acceleration_structure_properties)
        .push(&mut ray_tracing_pipeline_properties)
        .push(&mut descriptor_buffer_properties);

    unsafe { instance.get_physical_device_properties2(physical_device, &mut device_properties) };

    let mut ray = match Ray::new(
        device,
        instance,
        physical_device,
        rt_device,
        as_device,
        db_device,
        descriptor_buffer_properties,
        ray_tracing_pipeline_properties,
        acceleration_structure_properties,
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
                log::error!("rwlock is poisoned, ending thread: {}", e);
                break;
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
            log::trace!(
                "First images {:?} are now {}x{}",
                new_images,
                size.width,
                size.height
            );
            match transfer_sender.send(sync::ResizedSource::Ray((*size, new_images))) {
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
            log::trace!("Previous frame complete");

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
                match transfer_sender.send(sync::ResizedSource::Ray((*size, new_images))) {
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
    descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'a>,
    ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,
    acceleration_structure_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'a>,

    raygen_shader_binding_table: vulkan::Buffer,
    miss_shader_binding_table: vulkan::Buffer,
    hit_shader_binding_table: vulkan::Buffer,

    uniform: mpsc::Sender<uniforms::Event>,
    semaphore: Destructor<vk::Semaphore>,

    images: Option<[vulkan::Image<'a>; MAX_FRAMES_IN_FLIGHT]>,
    uniform_buffers: [vk::DeviceAddress; MAX_FRAMES_IN_FLIGHT],

    queue: vk::Queue,
    pipeline_layout: Destructor<vk::PipelineLayout>,
    pipeline: Destructor<vk::Pipeline>,
    command_pool: Destructor<vk::CommandPool>,
    commands: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    // descriptor_sets: Option<[vk::DescriptorSet; 2]>,
    material_location_map: HashMap<Entity, usize>,
    /// Map the IDs of objects to their position in the blass vector and store the object
    primitive_location_map: HashMap<Entity, (ObjectType, usize)>,
    primitive_addresses: Vec<PrimitiveAddresses>,
    materials_buffer: vulkan::Buffer,
    aabbs_buffers: Vec<vulkan::Buffer>,
    buffer_references: vulkan::Buffer,
    instances_buffer: Arc<RwLock<InstanceBuffer>>,

    output_image_descriptors: DescriptorData,
    ocean_descriptors: DescriptorData,
    tlas_descriptor: DescriptorData,
    // texture_descriptors: DescriptorData,
    push_constants: PushConstants,
    // textures: TextureBuffer,
    materials: Vec<MaterialData>,
    aabbs: Vec<AABB>,
    blass: Vec<AccelerationStructure>,
    tlas: AccelerationStructure,

    ocean: Ocean<'a>,
    physics: Physics,

    allocator: Arc<vk_mem::Allocator>,
    physical_device: vk::PhysicalDevice,
    rt_device: ash::khr::ray_tracing_pipeline::Device,
    as_device: ash::khr::acceleration_structure::Device,
    db_device: ash::ext::descriptor_buffer::Device,
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
        db_device: ash::ext::descriptor_buffer::Device,
        descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'a>,
        ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,
        acceleration_structure_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'a>,
        allocator: Arc<vk_mem::Allocator>,
        queue_family_indices: structures::QueueFamilyIndices,
        uniform_buffers: [vk::DeviceAddress; 2],

        uniform: mpsc::Sender<uniforms::Event>,
    ) -> Result<Self> {
        log::trace!("Creating object");

        // Populates queue
        let queue = core::create_queue(&device, queue_family_indices.compute_family.unwrap());

        let (command_pool, commands) =
            create_commands_flight_frames(&device, queue_family_indices.compute_family.unwrap().0)?;

        let materials_buffer = vulkan::Buffer::new_gpu(
            &allocator,
            (size_of::<MaterialData>() * RESERVED_SIZE) as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            "materials",
        )?;

        // Get a reference to where all the parts of each model are on the GPU
        let buffer_references = vulkan::Buffer::new_gpu(
            &allocator,
            (size_of::<primitives::PrimitiveAddresses>() * RESERVED_SIZE) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            "primitive addresses",
        )?;
        let instances_buffer = Arc::new(RwLock::new(InstanceBuffer::new(&allocator)?));

        let tlas = AccelerationStructure::new(&as_device, "tlas");

        let tlas_descriptor = DescriptorData::new(
            0,
            &allocator,
            &device,
            &db_device,
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            1,
            vk::ShaderStageFlags::RAYGEN_KHR,
            descriptor_buffer_properties,
        )?;
        let output_image_descriptors = DescriptorData::new(
            0,
            &allocator,
            &device,
            &db_device,
            vk::DescriptorType::STORAGE_IMAGE,
            MAX_FRAMES_IN_FLIGHT as u32,
            vk::ShaderStageFlags::RAYGEN_KHR,
            descriptor_buffer_properties,
        )?;
        let mut ocean_descriptors = DescriptorData::new(
            0,
            &allocator,
            &device,
            &db_device,
            vk::DescriptorType::STORAGE_IMAGE,
            FFT_IMAGES as u32,
            vk::ShaderStageFlags::RAYGEN_KHR // TODO: Remove Raygen (this is just for debug viewing the image directly)
                | vk::ShaderStageFlags::INTERSECTION_KHR // Needed for ray marching collisions
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR, // Needed for calculating normals
            descriptor_buffer_properties,
        )?;
        // let textures = DescriptorData::new(
        //     &allocator,
        //     &device,
        //     &db_device,
        //     vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        //     todo!("Figure out how many textures to support"),
        //     vk::ShaderStageFlags::RAYGEN_KHR // TODO: Remove Raygen (this is just for debug viewing the image directly)
        //             | vk::ShaderStageFlags::MISS_KHR // For the skybox
        //             | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        //     descriptor_buffer_properties,
        // )?;

        log::trace!("Creating pipeline");
        let (shader_groups, pipeline_layout, pipeline) = create_pipeline(
            &device,
            &rt_device,
            &tlas_descriptor,
            &output_image_descriptors,
            &ocean_descriptors,
        )?;

        log::trace!("Creating SBT");
        let (raygen_shader_binding_table, miss_shader_binding_table, hit_shader_binding_table) =
            create_shader_binding_tables(
                &allocator,
                &rt_device,
                pipeline.get(),
                &ray_tracing_pipeline_properties,
                &shader_groups,
            )?;

        log::trace!("Building semaphore");
        let semaphore = Destructor::new(
            &device,
            core::create_semaphore(&device)?,
            device.fp_v1_0().destroy_semaphore,
        );

        let mut materials = Vec::with_capacity(RESERVED_SIZE);
        // Set a default material for primitives that are missing one or have not been properly set up
        materials.push(Material::new_basic(Vec3::new(1.0, 0.1, 0.7), 0.).get_data());

        let ocean = Ocean::new(&device, &allocator, queue_family_indices)?;

        log::info!("Ocean has provided {} images", ocean.images.len());
        // Create ocean descriptor
        // for (i, image) in ocean.images.iter().enumerate()
        {
            let i = 0;
            let image = &ocean.images[0];
            log::info!("Building ocean image {i}");
            // Create a new descriptor pointing at the new image
            ocean_descriptors.get_descriptor_image(
                &db_device,
                image.view(),
                i,
                descriptor_buffer_properties,
            )?;
        }

        let physics = Physics::new(
            &device,
            Arc::clone(&allocator),
            queue_family_indices,
            Arc::clone(&instances_buffer),
            ocean.images[0].view(),
        )?;

        let push_constants = PushConstants {
            frame_index: 0,
            ubo: uniform_buffers[0],
            scene: buffer_references.get_device_address(&device),
            materials: materials_buffer.get_device_address(&device),
        };

        Ok(Self {
            device,
            instance,
            physical_device,
            allocator,
            semaphore,
            images: None,
            queue,
            pipeline_layout,
            pipeline,
            command_pool,
            commands,
            rt_device,
            as_device,
            db_device,
            ray_tracing_pipeline_properties,
            acceleration_structure_properties,
            raygen_shader_binding_table,
            miss_shader_binding_table,
            hit_shader_binding_table,
            tlas,
            buffer_references,
            materials,
            aabbs: Vec::with_capacity(RESERVED_SIZE),
            blass: Vec::with_capacity(RESERVED_SIZE),
            materials_buffer,
            primitive_addresses: Vec::with_capacity(RESERVED_SIZE),
            aabbs_buffers: Vec::with_capacity(RESERVED_SIZE),
            primitive_location_map: HashMap::with_capacity(RESERVED_SIZE),
            material_location_map: HashMap::with_capacity(RESERVED_SIZE),
            uniform,
            uniform_buffers,
            instances_buffer,
            ocean,
            physics,
            descriptor_buffer_properties,
            tlas_descriptor,
            output_image_descriptors,
            ocean_descriptors,
            push_constants,
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
            "Ray Output",
        )?);

        for (i, image) in self.images.as_ref().unwrap().iter().enumerate() {
            self.output_image_descriptors.get_descriptor_image(
                &self.db_device,
                image.view(),
                i,
                self.descriptor_buffer_properties,
            )?;
        }

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
            for (i, image) in self.images.as_ref().unwrap().iter().enumerate() {
                self.output_image_descriptors.get_descriptor_image(
                    &self.db_device,
                    image.view(),
                    i,
                    self.descriptor_buffer_properties,
                )?;
            }
        }
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
                    self.acceleration_structure_properties,
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
            Primitive::Ocean(ref mut o) if !self.primitive_location_map.contains_key(&entity) => {
                log::info!("Adding Ocean {:?}: {:?}", entity, o);
                o.set_materials(&self.material_location_map);
                o.allocate(
                    &self.allocator,
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                )?;
                self.primitive_location_map
                    .insert(entity, (ObjectType::Ocean, self.blass.len()));
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
        let mut primitives_changed = false;
        let mut need_rebuild = false;

        {
            let mut w = world.write().unwrap();
            // TODO: Use some sort of `new` component to only add primitives/materials that have changed
            for (entity, material) in w.query_mut::<&Material>() {
                if self.material_location_map.contains_key(&entity) {
                    continue;
                }
                // If the material is new, remake the buffer with all the materials
                self.materials.push(material.get_data());
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
                    "materials",
                )?;
                // Update reference that shader sees
                self.push_constants.materials =
                    self.materials_buffer.get_device_address(&self.device);
                // TODO: Remove materials that no longer exist
            }

            let mut primitives = Vec::with_capacity(RESERVED_SIZE);
            for (entity, primitive) in w.query::<&mut Primitive>().iter() {
                if let Primitive::Ocean(o) = primitive {
                    if !self.ocean.update(entity, o) {
                        // We don't want to include the ocean primitive if the ocean struct is ignoring it
                        continue;
                    }
                }

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

            let mut instances = HashSet::with_capacity(RESERVED_SIZE);
            for (entity, instance) in w.query_mut::<&api::Instance>() {
                // Get the primitive that this is an instance of
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

                // We only add the instance if it isn't already there
                if self.instances_buffer.write().unwrap().try_add(
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                    &self.allocator,
                    entity,
                    create_blas_instance(
                        &self.as_device,
                        &mut self.blass,
                        *location_in_blas,
                        instance.initial_transform * instance.base_transform,
                        *primitive_type,
                    )?,
                    instance.base_transform,
                    instance.initial_transform,
                )? {
                    log::trace!("Adding an instance: {:?}", instance);
                    need_rebuild = true;
                }

                // Keep track of the indices that were found so we can drop dead ones
                instances.insert(entity);
            }
            if self
                .instances_buffer
                .write()
                .unwrap()
                .remove_orphans(&instances)
                > 0
            {
                need_rebuild = true;
            }
        }
        log::trace!("Instances processed");
        if primitives_changed {
            if self
                .buffer_references
                .check_total_space::<PrimitiveAddresses>(self.primitive_addresses.len())
            {
                log::trace!(
                    "Using old buffer reference buffer to store {} addresses",
                    self.primitive_addresses.len()
                );
                self.buffer_references.populate_staged(
                    &self.device,
                    self.command_pool.get(),
                    self.queue,
                    &self.allocator,
                    self.primitive_addresses.as_ptr(),
                    self.primitive_addresses.len(),
                )?;
            } else {
                log::trace!("Growing buffer reference buffer");
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
                    "primitive addresses",
                )?;
                self.push_constants.scene = self.buffer_references.get_device_address(&self.device);
            }
        }
        self.create_top_level_acceleration_structure(if need_rebuild {
            TlasProcess::Build
        } else {
            TlasProcess::Update
        })?;

        Ok(())
    }
    pub fn ray_trace(
        &mut self,
        current_frame_index: usize,
        transfer_semaphore: vk::Semaphore,
        transfer_timestamp_to_wait: u64,
        ray_timestamp_to_wait: u64,
        timestamp_to_signal: u64,
    ) -> Result<()> {
        if self.images.is_none() {
            // We only want to start ray tracing once the output images are set up
            return Ok(());
        }

        let (ocean_semaphore, ocean_timestamp) =
            self.ocean.dispatch(&self.device, current_frame_index)?;

        let (physics_semaphore, physics_timestamp) =
            self.physics.dispatch(&self.device, current_frame_index)?;

        let height = self.images.as_ref().unwrap()[current_frame_index].height;
        let width = self.images.as_ref().unwrap()[current_frame_index].width;

        self.push_constants.frame_index = current_frame_index as u32;
        self.push_constants.ubo = self.uniform_buffers[current_frame_index];

        self.build_command_buffers(width, height, current_frame_index)?;

        // log::trace!(
        //     "Ray tracing into {current_frame_index} when timestamp reaches {}",
        //     transfer_timestamp_to_wait
        // );
        let wait_timestamps = [
            transfer_timestamp_to_wait,
            ray_timestamp_to_wait,
            ocean_timestamp,
            physics_timestamp,
        ];
        let wait_semaphores = [
            transfer_semaphore,
            self.semaphore.get(),
            ocean_semaphore,
            physics_semaphore,
        ];
        let wait_stages = [
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
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

    fn create_top_level_acceleration_structure(&mut self, process: TlasProcess) -> Result<()> {
        log::trace!(
            "Populating instances with {}x{}={} Bytes",
            size_of::<vk::AccelerationStructureInstanceKHR>(),
            self.instances_buffer.read().unwrap().instance_count,
            (size_of::<vk::AccelerationStructureInstanceKHR>()
                * self.instances_buffer.read().unwrap().instance_count)
        );
        let instance_data_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: self.instances_buffer.write().unwrap().get_address_array(
                &self.device,
                self.command_pool.get(),
                self.queue,
                &self.allocator,
            )?,
        };

        // The top level acceleration structure contains (bottom level) instances as the input geometry
        let as_geometries = [vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                    array_of_pointers: vk::TRUE,
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

        let primitive_count = [self.instances_buffer.read().unwrap().instance_count as u32];

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
                log::trace!("TLAS needs rebuilding");
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

                // Update the stored device address for the TLAS
                let as_device_address_info =
                    vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(self.tlas.handle);
                self.tlas.device_address = unsafe {
                    self.as_device
                        .get_acceleration_structure_device_address(&as_device_address_info)
                };

                // Update TLAS descriptor
                self.tlas_descriptor.get_descriptor_tlas(
                    &self.db_device,
                    self.tlas.device_address,
                    0,
                    self.descriptor_buffer_properties,
                )?;
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
            self.acceleration_structure_properties,
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
        // TODO: I wonder if we should double down on rebuilding the TLAS every time and incorporate this command into the main ray pipeline
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
                "aabbs",
            )?);
        self.blass.push(create_bottom_level_acceleration_structure(
            &self.device,
            &self.allocator,
            &self.as_device,
            self.acceleration_structure_properties,
            self.command_pool.get(),
            self.queue,
            vk::GeometryTypeKHR::AABBS,
            None,
            Some(&self.aabbs_buffers.last().unwrap()),
            Some(1),
        )?);
        Ok(())
    }

    fn build_command_buffers(&self, width: u32, height: u32, frame_index: usize) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        let command_buffer = self.commands[frame_index];
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let single_handle_size_aligned = core::aligned_size(
                self.ray_tracing_pipeline_properties
                    .shader_group_handle_size,
                self.ray_tracing_pipeline_properties
                    .shader_group_handle_alignment,
            ) as u64;
            let base_handle_size_aligned = core::aligned_size(
                self.ray_tracing_pipeline_properties
                    .shader_group_handle_size,
                self.ray_tracing_pipeline_properties
                    .shader_group_base_alignment,
            ) as u64;

            let raygen_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
                device_address: self
                    .raygen_shader_binding_table
                    .get_device_address(&self.device),
                stride: base_handle_size_aligned,
                size: base_handle_size_aligned,
            };
            let miss_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
                device_address: self
                    .miss_shader_binding_table
                    .get_device_address(&self.device),
                stride: single_handle_size_aligned,
                size: base_handle_size_aligned,
            };
            let hit_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR {
                device_address: self
                    .hit_shader_binding_table
                    .get_device_address(&self.device),
                stride: single_handle_size_aligned,
                size: base_handle_size_aligned,
            };

            let callable_shader_sbt_entry = vk::StridedDeviceAddressRegionKHR::default();

            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout.get(),
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::INTERSECTION_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                0,
                self.push_constants.as_slice(),
            );

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline.get(),
            );

            // How do we include the rest of the descriptors we need in the regime
            // Where does the acceleration structure go?
            // UBO, Scene buffer, and materials can probably go in via push constants
            // https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/descriptor_buffer_basic
            let descriptor_buffer_binding_info = [
                //  output images + Ocean map(s)
                vk::DescriptorBufferBindingInfoEXT {
                    address: self.tlas_descriptor.buffer.get_device_address(&self.device),
                    usage: vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
                    ..Default::default()
                },
                vk::DescriptorBufferBindingInfoEXT {
                    address: self
                        .output_image_descriptors
                        .buffer
                        .get_device_address(&self.device),
                    usage: vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
                    ..Default::default()
                },
                vk::DescriptorBufferBindingInfoEXT {
                    address: self
                        .ocean_descriptors
                        .buffer
                        .get_device_address(&self.device),
                    usage: vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
                    ..Default::default()
                },
                // Textures
                // vk::DescriptorBufferBindingInfoEXT {
                //     address: self
                //         .texture_descriptors
                //         .buffer
                //         .get_device_address(&self.device),
                //     usage: vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
                //         | vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
                //     ..Default::default()
                // },
            ];
            self.db_device
                .cmd_bind_descriptor_buffers(command_buffer, &descriptor_buffer_binding_info);

            // These are the indices of the respective fields in the descriptor_buffer_binding_info array
            let buffer_index_tlas = [0];
            let buffer_index_images = [1];
            let buffer_index_ocean = [2];
            let buffer_offsets = [0];

            // Top Level Acceleration Structure
            self.db_device.cmd_set_descriptor_buffer_offsets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout.get(),
                0, // TLAS goes into set 0
                &buffer_index_tlas,
                &buffer_offsets,
            );

            // Output Images
            self.db_device.cmd_set_descriptor_buffer_offsets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout.get(),
                1, // Output images go into set 1
                &buffer_index_images,
                &buffer_offsets,
            );

            // Ocean image(s)
            self.db_device.cmd_set_descriptor_buffer_offsets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout.get(),
                2, // Ocean map(s) go into set 2
                &buffer_index_ocean,
                &buffer_offsets,
            );

            // Textures
            // buffer_offsets[0] = 0;
            // for _ in textures.len() {
            //     buffer_offsets[0] += self.texture_descriptors.size;
            //     self.db_device.cmd_set_descriptor_buffer_offsets(
            //         command_buffer,
            //         vk::PipelineBindPoint::RAY_TRACING_KHR,
            //         self.pipeline_layout.get(),
            //         1, // Textures go into set 1
            //         &buffer_index_textures,
            //         &buffer_offsets,
            //     );
            // }

            // Dispatch the ray tracing commands
            self.rt_device.cmd_trace_rays(
                command_buffer,
                &raygen_shader_sbt_entry,
                &miss_shader_sbt_entry,
                &hit_shader_sbt_entry,
                &callable_shader_sbt_entry,
                width,
                height,
                1,
            );

            self.device.end_command_buffer(command_buffer)?;
        }
        // log::info!("Recorded commands {:?}", self.commands);
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
        properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    ) -> Result<Self> {
        log::trace!("Creating Scratch buffer");
        let buffer = vulkan::Buffer::new_aligned(
            allocator,
            size,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::empty(),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            properties.min_acceleration_structure_scratch_offset_alignment as u64,
            "Scratch",
        )?;

        let device_address = buffer.get_device_address(device);

        Ok(Self {
            device_address,
            _buffer: buffer,
        })
    }
}

///  Create the bottom level acceleration structure that contains the scene's geometry (triangles)
fn create_bottom_level_acceleration_structure(
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    as_device: &ash::khr::acceleration_structure::Device,
    acceleration_structure_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
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
        "blas transform staging",
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
                device_address: transform_buffer.get_device_address(device),
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
    let mut bottom_level_as = AccelerationStructure::new(as_device, "blas");
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
    let scratch_buffer = ScratchBuffer::new(
        device,
        allocator,
        as_build_sizes_info.build_scratch_size,
        acceleration_structure_properties,
    )?;

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
        "raygen sbt",
    )?;
    let mut miss_shader_binding_table = vulkan::Buffer::new_generic(
        allocator,
        base_size,
        vk_mem::MemoryUsage::Auto,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        sbt_buffer_usage_flags,
        "miss sbt",
    )?;
    let mut hit_shader_binding_table = vulkan::Buffer::new_generic(
        allocator,
        base_size,
        vk_mem::MemoryUsage::Auto,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        sbt_buffer_usage_flags,
        "hit sbt",
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

fn create_pipeline<'a>(
    device: &ash::Device,
    rt_device: &ash::khr::ray_tracing_pipeline::Device,
    tlas_descriptor: &DescriptorData,
    output_image_descriptor: &DescriptorData,
    ocean_descriptor: &DescriptorData,
    // textures: &mut DescriptorData,
) -> Result<(
    Vec<vk::RayTracingShaderGroupCreateInfoKHR<'a>>,
    Destructor<vk::PipelineLayout>,
    Destructor<vk::Pipeline>,
)> {
    let push_constants = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR
            | vk::ShaderStageFlags::INTERSECTION_KHR
            | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        size: size_of::<PushConstants>() as u32,
        offset: 0,
    }];

    let set_layouts = [
        tlas_descriptor.layout.get(),
        output_image_descriptor.layout.get(),
        ocean_descriptor.layout.get(),
        // textures.layout.get(),
    ];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constants);

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
        .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
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
            shader_groups,
            pipeline_layout,
            Destructor::new(device, v[0], device.fp_v1_0().destroy_pipeline),
        )),
        Err(e) => Err(anyhow!("Failed to create graphics pipeline: {:?}", e)),
    }
}
