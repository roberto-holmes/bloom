use std::{array, ffi::CString, path::Path};

use anyhow::{anyhow, Result};
use ash::vk;

use crate::{
    core, structures, tools::read_shader_code, vulkan::Destructor,
    IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT, IDEAL_RADIANCE_IMAGE_SIZE_WIDTH, MAX_FRAMES_IN_FLIGHT,
};

pub struct Compute {
    device: ash::Device,
    semaphore: Destructor<vk::Semaphore>,

    radiance_images: [vk::Image; 2],
    radiance_image_views: [vk::ImageView; 2],
    radiance_image_memories: [vk::DeviceMemory; 2],

    queue: vk::Queue,
    descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    pipeline_layout: Destructor<vk::PipelineLayout>,
    pipeline: Destructor<vk::Pipeline>,
    command_pool: Destructor<vk::CommandPool>,
    commands: [vk::CommandBuffer; 2],
    copy_commands: [vk::CommandBuffer; 4],
    descriptor_pool: Destructor<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    current_timestamp: u64,
    current_frame_index: usize,
}

impl Compute {
    pub fn new(
        device: ash::Device,
        queue_family_indices: structures::QueueFamilyIndices,
        radiance_images: [vk::Image; 2],
        radiance_image_views: [vk::ImageView; 2],
        radiance_image_memories: [vk::DeviceMemory; 2],
        output_images: [vk::Image; 2],
    ) -> Result<Self> {
        // Populates queue
        let queue = create_queue(&device, queue_family_indices);
        // Populates descriptor_set_layout, pipeline_layout, descriptor_pool, descriptor_sets, and pipeline
        let (descriptor_pool, descriptor_set_layout, descriptor_sets, pipeline_layout, pipeline) =
            create_pipeline(&device, &radiance_image_views)?;
        // Populates command_pool, commands, and copy_commands
        let (command_pool, commands, copy_commands) = create_commands(
            &device,
            queue_family_indices,
            &descriptor_sets,
            pipeline_layout.get(),
            pipeline.get(),
            radiance_images,
            output_images,
        )?;
        // populates semaphore
        let semaphore = Destructor::new(
            &device,
            create_semaphore(&device)?,
            device.fp_v1_0().destroy_semaphore,
        );

        Ok(Self {
            device,
            semaphore,
            radiance_images,
            radiance_image_views,
            radiance_image_memories,
            queue,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            command_pool,
            commands,
            copy_commands,
            descriptor_pool,
            descriptor_sets,
            current_timestamp: 0,
            current_frame_index: 0,
        })
    }
    pub fn get_semaphore(&self) -> vk::Semaphore {
        self.semaphore.get()
    }
    pub fn update(&mut self) -> Result<()> {
        // Perform a render
        self.perform_compute(self.current_timestamp, self.current_timestamp + 1)?;

        self.current_frame_index += 1;
        self.current_frame_index %= MAX_FRAMES_IN_FLIGHT;

        // Perform a copy
        self.perform_compute_copy(self.current_timestamp + 1, self.current_timestamp + 2)?;

        // Wait for the copy to complete
        self.wait_for_compute_operation(self.current_timestamp + 2)?;
        self.current_timestamp += 2;

        Ok(())
    }
    fn wait_for_compute_operation(&mut self, timestamp_to_wait: u64) -> Result<()> {
        let semaphores = [self.semaphore.get()];
        let wait_values = [timestamp_to_wait];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&wait_values);
        unsafe { self.device.wait_semaphores(&wait_info, u64::MAX) }?;
        Ok(())
    }
    fn perform_compute(&self, timestamp_to_wait: u64, timestamp_to_signal: u64) -> Result<()> {
        let wait_timestamps = [timestamp_to_wait];
        let signal_timestamps = [timestamp_to_signal];
        let semaphores = [self.semaphore.get()];

        let command_buffer = [self.commands[self.current_frame_index]];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_timestamps)
            .wait_semaphore_values(&wait_timestamps);
        let submit_info = [vk::SubmitInfo::default()
            .command_buffers(&command_buffer)
            .signal_semaphores(&semaphores)
            .push(&mut timeline_info)];

        unsafe {
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
        }?;
        Ok(())
    }
    fn perform_compute_copy(
        &mut self,
        timestamp_to_wait: u64,
        timestamp_to_signal: u64,
    ) -> Result<()> {
        let wait_timestamps = [timestamp_to_wait];
        let signal_timestamps = [timestamp_to_signal];
        let semaphores = [self.semaphore.get()];

        let command_index = 1 << self.current_frame_index | self.current_frame_index;
        let command_buffer = [self.copy_commands[command_index]];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_timestamps)
            .wait_semaphore_values(&wait_timestamps);
        let submit_info = [vk::SubmitInfo::default()
            .command_buffers(&command_buffer)
            .signal_semaphores(&semaphores)
            .push(&mut timeline_info)];

        match unsafe {
            self.device
                .queue_submit(self.queue, &submit_info, vk::Fence::null())
        } {
            Err(e) => return Err(anyhow!("Failed to submit compute commands: {}", e)),
            _ => {}
        }
        Ok(())
    }
    pub fn signal_end_compute(&self) -> Result<()> {
        let signal_info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore.get())
            .value(u64::MAX);
        match unsafe { self.device.signal_semaphore(&signal_info) } {
            Err(e) => {
                return Err(anyhow!("Failed to signal compute copy semaphore: {}", e));
            }
            _ => {}
        };
        Ok(())
    }
}

impl Drop for Compute {
    fn drop(&mut self) {
        unsafe {
            for image_view in self.radiance_image_views {
                self.device.destroy_image_view(image_view, None);
            }
            for image in self.radiance_images {
                self.device.destroy_image(image, None);
            }
            for image_memory in self.radiance_image_memories {
                self.device.free_memory(image_memory, None); // TODO: Make RAII wrapper
            }
        }
    }
}

fn create_queue(
    device: &ash::Device,
    queue_family_indices: structures::QueueFamilyIndices,
) -> vk::Queue {
    unsafe {
        device.get_device_queue(
            queue_family_indices.compute_family.unwrap(),
            if queue_family_indices.queue_count > 1 {
                1
            } else {
                0
            },
        )
    }
}
fn create_semaphore(device: &ash::Device) -> Result<vk::Semaphore> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);
    Ok(unsafe { device.create_semaphore(&semaphore_info, None) }?)
}

fn create_pipeline(
    device: &ash::Device,
    radiance_image_views: &[vk::ImageView; 2],
) -> Result<(
    Destructor<vk::DescriptorPool>,
    Destructor<vk::DescriptorSetLayout>,
    Vec<vk::DescriptorSet>,
    Destructor<vk::PipelineLayout>,
    Destructor<vk::Pipeline>,
)> {
    let set_layout_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let descriptor_layout =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_layout_bindings);
    let descriptor_set_layout = Destructor::new(
        device,
        unsafe { device.create_descriptor_set_layout(&descriptor_layout, None)? },
        device.fp_v1_0().destroy_descriptor_set_layout,
    );
    let mut set_layouts = vec![];
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        set_layouts.push(descriptor_set_layout.get());
    }

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let pipeline_layout = Destructor::new(
        device,
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? },
        device.fp_v1_0().destroy_pipeline_layout,
    );

    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_IMAGE),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_IMAGE),
    ];

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
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? }; // TODO: Create Destructor wrapper

    for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
        // Swap radiance image views around each frame
        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(radiance_image_views[i])];

        let storage_image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(radiance_image_views[(i + 1) % 2])];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .image_info(&image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .image_info(&storage_image_info),
        ];
        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    let shader_code = read_shader_code(Path::new("shaders/spv/ray.comp.spv"))?;
    let shader_module = core::create_shader_module(&device, &shader_code)?;
    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&main_function_name);

    let pipeline_create_infos = [vk::ComputePipelineCreateInfo::default()
        .layout(pipeline_layout.get())
        .stage(stage)];
    let pipeline = Destructor::new(
        device,
        match unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
        } {
            Ok(v) => v[0], // We are only creating one timeline so we only want the first object in the vector
            Err(e) => return Err(anyhow!(e.1)),
        },
        device.fp_v1_0().destroy_pipeline,
    );

    unsafe { device.destroy_shader_module(shader_module, None) }; // TODO: RAII
    Ok((
        descriptor_pool,
        descriptor_set_layout,
        descriptor_sets,
        pipeline_layout,
        pipeline,
    ))
}

fn create_commands(
    device: &ash::Device,
    queue_family_indices: structures::QueueFamilyIndices,
    descriptor_sets: &Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    radiance_images: [vk::Image; 2],
    output_images: [vk::Image; 2],
) -> Result<(
    Destructor<vk::CommandPool>,
    [vk::CommandBuffer; 2],
    [vk::CommandBuffer; 4],
)> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_indices.compute_family.unwrap());
    let command_pool = Destructor::new(
        device,
        unsafe { device.create_command_pool(&pool_info, None)? },
        device.fp_v1_0().destroy_command_pool,
    );

    let command_buffers = core::create_command_buffers(device, command_pool.get(), 2)?;

    let copy_command_buffers = core::create_command_buffers(device, command_pool.get(), 4)?;

    let commands = record_commands(
        device,
        pipeline_layout,
        pipeline,
        &descriptor_sets,
        &command_buffers,
    )?;
    let copy_commands = record_copy_commands(
        device,
        &copy_command_buffers,
        &radiance_images,
        &output_images,
    )?;

    Ok((command_pool, commands, copy_commands))
}

fn compute_commands(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffers: &Vec<vk::CommandBuffer>,
    descriptor_set: vk::DescriptorSet,
    current_frame_index: usize,
) -> Result<vk::CommandBuffer> {
    assert_eq!(command_buffers.len(), 2);
    let begin_info = vk::CommandBufferBeginInfo::default();
    let command_buffer = command_buffers[current_frame_index];

    let descriptor_sets_to_bind = [descriptor_set];

    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &descriptor_sets_to_bind,
            &[],
        );
        device.cmd_dispatch(command_buffer, 16, 16, 1);
        device.end_command_buffer(command_buffer)?;
    }
    Ok(command_buffer)
}

pub fn record_commands(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_sets: &Vec<vk::DescriptorSet>,
    command_buffers: &Vec<vk::CommandBuffer>,
) -> Result<[vk::CommandBuffer; 2]> {
    assert_eq!(command_buffers.len(), 2);
    assert_eq!(descriptor_sets.len(), 2);
    // let mut commands = [vk::CommandBuffer::null(); 2];

    let commands: [vk::CommandBuffer; 2] = array::from_fn(|i| {
        compute_commands(
            device,
            pipeline_layout,
            pipeline,
            command_buffers,
            descriptor_sets[i],
            i,
        )
        .unwrap() // TODO: Figure out a way to avoid the unwrap
    });

    // for c in commands.iter_mut() {
    //     c = self.compute_commands(command_buffers, self.descriptor_sets[0], 0)?;
    // }
    Ok(commands)
}

fn copy_commands(
    device: &ash::Device,
    command_buffers: &Vec<vk::CommandBuffer>,
    current_frame_index: usize,
    draw_frame: usize,
    radiance_images: &[vk::Image; 2],
    output_images: &[vk::Image; 2],
) -> Result<vk::CommandBuffer> {
    assert_eq!(command_buffers.len(), 4);
    let command_buffer = command_buffers[1 << draw_frame | current_frame_index];

    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    let mut compute_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(radiance_images[current_frame_index])
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let mut draw_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(output_images[draw_frame])
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

    let mut image_memory_barriers = [compute_barrier, draw_barrier];

    let subresource = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };

    let regions = [vk::ImageCopy {
        src_subresource: subresource,
        dst_subresource: subresource,
        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: IDEAL_RADIANCE_IMAGE_SIZE_WIDTH, // TODO: Resize dynamically
            height: IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT,
            depth: 1,
        },
    }];

    let begin_info = vk::CommandBufferBeginInfo::default();
    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_memory_barriers,
        );
        device.cmd_copy_image(
            command_buffer,
            radiance_images[current_frame_index],
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            output_images[draw_frame],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
    }

    compute_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
    compute_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
    draw_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    draw_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
    compute_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    compute_barrier.new_layout = vk::ImageLayout::GENERAL;
    draw_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    draw_barrier.new_layout = vk::ImageLayout::GENERAL;

    image_memory_barriers = [compute_barrier, draw_barrier]; // TODO: Is this necessary or will the changes already be here?

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_memory_barriers,
        );
        device.end_command_buffer(command_buffer)?;
    }

    Ok(command_buffer)
}

pub fn record_copy_commands(
    device: &ash::Device,
    command_buffers: &Vec<vk::CommandBuffer>,
    radiance_images: &[vk::Image; 2],
    output_images: &[vk::Image; 2],
) -> Result<[vk::CommandBuffer; 4]> {
    // We need a set of commands for each permutation of the frames lining up
    Ok([
        copy_commands(
            device,
            command_buffers,
            0,
            0,
            radiance_images,
            output_images,
        )?,
        copy_commands(
            device,
            command_buffers,
            1,
            0,
            radiance_images,
            output_images,
        )?,
        copy_commands(
            device,
            command_buffers,
            0,
            1,
            radiance_images,
            output_images,
        )?,
        copy_commands(
            device,
            command_buffers,
            1,
            1,
            radiance_images,
            output_images,
        )?,
    ])
}
