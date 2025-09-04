use std::{
    self,
    ffi::CString,
    path::Path,
    sync::{Arc, RwLock},
    time::Instant,
};

use anyhow::{Result, anyhow};
use ash::vk;

use crate::{
    MAX_FRAMES_IN_FLIGHT,
    core::{self, create_shader_module},
    ray::instance_buffer::InstanceBuffer,
    structures,
    tools::read_shader_code,
    vulkan::Destructor,
};

#[derive(Debug, Default)]
#[repr(C)]
struct PushConstants {
    pub timestamp_ns: u64, // Nanoseconds from start of running
    pub instances: vk::DeviceAddress,
    pub instance_count: u32,
}

impl PushConstants {
    pub fn new() -> Self {
        Self {
            timestamp_ns: 0,
            instances: 0,
            instance_count: 0,
        }
    }

    pub fn as_slice(&self) -> &[u8; size_of::<Self>()] {
        unsafe { &*(self as *const Self as *const [u8; size_of::<Self>()]) }
    }
}

#[repr(C)]
pub struct EntityData {
    pub base_transform: cgmath::Matrix4<f32>,
    pub world_transform: cgmath::Matrix4<f32>,
    pub entity: u32,
    pub pad: [u32; 3],
}

pub struct Physics {
    start_time: Instant,
    instances: Arc<RwLock<InstanceBuffer>>, // TODO: Make 2 so that ray can use it at the same time

    _descriptor_pool: Destructor<vk::DescriptorPool>,
    _descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    pipeline_layout: Destructor<vk::PipelineLayout>,
    pipeline: Destructor<vk::Pipeline>,
    commands: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    command_pool: Destructor<vk::CommandPool>,

    semaphores: [Destructor<vk::Semaphore>; MAX_FRAMES_IN_FLIGHT],
    semaphore_values: [u64; MAX_FRAMES_IN_FLIGHT],

    push_constants: PushConstants,

    queue: vk::Queue,
    allocator: Arc<vk_mem::Allocator>,
}

impl Physics {
    pub fn new(
        device: &ash::Device,
        allocator: Arc<vk_mem::Allocator>,
        queue_family_indices: structures::QueueFamilyIndices,
        instances: Arc<RwLock<InstanceBuffer>>,
        ocean: vk::ImageView,
    ) -> Result<Self> {
        let queue = core::create_queue(&device, queue_family_indices.compute_family.unwrap());
        let (command_pool, commands) = core::create_commands_flight_frames(
            &device,
            queue_family_indices.compute_family.unwrap().0,
        )?;

        let (descriptor_pool, descriptor_set_layout, descriptor_sets, pipeline_layout, pipeline) =
            create_pipeline(device, ocean)?;

        let semaphores = std::array::from_fn(|_| {
            Destructor::new(
                &device,
                core::create_semaphore(&device).unwrap(),
                device.fp_v1_0().destroy_semaphore,
            )
        });

        Ok(Self {
            start_time: Instant::now(),
            instances,
            push_constants: PushConstants::new(),
            _descriptor_pool: descriptor_pool,
            _descriptor_set_layout: descriptor_set_layout,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            commands,
            command_pool,
            semaphore_values: [0; 2],
            semaphores,
            queue,
            allocator,
        })
    }
    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        frame_index: usize,
    ) -> Result<(vk::Semaphore, u64)> {
        // Update Push constants
        self.push_constants.timestamp_ns = self.start_time.elapsed().as_nanos() as u64;
        self.push_constants.instances = self.instances.write().unwrap().get_address_array(
            device,
            self.command_pool.get(),
            self.queue,
            &self.allocator,
        )?;
        self.push_constants.instance_count = self.instances.read().unwrap().instance_count as u32;

        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device.begin_command_buffer(self.commands[frame_index], &begin_info)? };
        unsafe {
            // Update the push constants
            device.cmd_push_constants(
                self.commands[frame_index],
                self.pipeline_layout.get(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                self.push_constants.as_slice(),
            );
            device.cmd_bind_pipeline(
                self.commands[frame_index],
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.get(),
            );
            let descriptor_sets_to_bind = [self.descriptor_sets[frame_index]];
            device.cmd_bind_descriptor_sets(
                self.commands[frame_index],
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout.get(),
                0,
                &descriptor_sets_to_bind,
                &[],
            );
            // Dispatch the generation of the ocean wave spectra

            let group_count = 1.max(self.push_constants.instance_count / 8);
            device.cmd_dispatch(self.commands[frame_index], group_count, 1, 1);
            device.end_command_buffer(self.commands[frame_index])?;
        }

        let command_buffer_infos = [vk::CommandBufferSubmitInfo {
            command_buffer: self.commands[frame_index],
            ..Default::default()
        }];
        // Wait for the last run to finish
        let wait_semaphore_infos = [vk::SemaphoreSubmitInfo {
            semaphore: self.semaphores[frame_index].get(),
            value: self.semaphore_values[frame_index],
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            ..Default::default()
        }];
        // Let ray know that we are finished
        let signal_semaphore_infos = [vk::SemaphoreSubmitInfo {
            semaphore: self.semaphores[frame_index].get(),
            value: self.semaphore_values[frame_index] + 1,
            stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            ..Default::default()
        }];
        let submits = [vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .wait_semaphore_infos(&wait_semaphore_infos)];

        unsafe {
            device.queue_submit2(self.queue, &submits, vk::Fence::null())?;
        };

        self.semaphore_values[frame_index] += 1;
        Ok((
            self.semaphores[frame_index].get(),
            self.semaphore_values[frame_index],
        ))
    }
}

fn create_pipeline(
    device: &ash::Device,
    ocean: vk::ImageView,
) -> Result<(
    Destructor<vk::DescriptorPool>,
    Destructor<vk::DescriptorSetLayout>,
    [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    Destructor<vk::PipelineLayout>,
    Destructor<vk::Pipeline>,
)> {
    let set_layout_bindings = [vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        ..Default::default()
    }];

    let descriptor_layout =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_layout_bindings);
    let descriptor_set_layout = Destructor::new(
        device,
        unsafe { device.create_descriptor_set_layout(&descriptor_layout, None)? },
        device.fp_v1_0().destroy_descriptor_set_layout,
    );
    let set_layouts: [vk::DescriptorSetLayout; MAX_FRAMES_IN_FLIGHT] =
        std::array::from_fn(|_| descriptor_set_layout.get());

    let push_constants = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        size: size_of::<PushConstants>() as u32,
        offset: 0,
    }];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constants);
    let pipeline_layout = Destructor::new(
        device,
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? },
        device.fp_v1_0().destroy_pipeline_layout,
    );

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
        .ty(vk::DescriptorType::STORAGE_IMAGE)];

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

    let descriptor_pool = Destructor::new(
        device,
        unsafe { device.create_descriptor_pool(&pool_info, None)? },
        device.fp_v1_0().destroy_descriptor_pool,
    );

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool.get())
        .set_layouts(set_layouts.as_slice());
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

    for (_, &descriptor_set) in descriptor_sets.iter().enumerate() {
        // Swap image views around each frame
        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(ocean)];
        let descriptor_writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .image_info(&image_info)];
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    let shader_code = read_shader_code(Path::new("shaders/spv/physics.slang.spv"))?;
    let shader_module = create_shader_module(&device, &shader_code)?;
    log::info!("Processing shader {:?}", shader_module.get());
    let main_function_name = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module.get())
        .name(&main_function_name);

    let pipeline_create_infos = [vk::ComputePipelineCreateInfo::default()
        .layout(pipeline_layout.get())
        .stage(stage)];
    let pipeline = Destructor::new(
        device,
        match unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
        } {
            Ok(v) => v[0], // We are only creating one pipeline so we only want the first object in the vector
            Err(e) => return Err(anyhow!(e.1)),
        },
        device.fp_v1_0().destroy_pipeline,
    );

    let descriptor_sets = <Vec<vk::DescriptorSet> as TryInto<
        [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    >>::try_into(descriptor_sets)
    .unwrap();

    Ok((
        descriptor_pool,
        descriptor_set_layout,
        descriptor_sets,
        pipeline_layout,
        pipeline,
    ))
}
