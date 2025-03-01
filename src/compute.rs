use std::{
    array,
    ffi::CString,
    path::Path,
    sync::{mpsc, Arc, RwLock},
};

use anyhow::{anyhow, Result};
use ash::vk;

use crate::{
    core, structures,
    tools::read_shader_code,
    vulkan::{self, Destructor},
    MAX_FRAMES_IN_FLIGHT, WINDOW_HEIGHT, WINDOW_WIDTH,
};

pub fn thread<'a>(
    device: ash::Device,
    queue_family_indices: structures::QueueFamilyIndices,
    images: [vulkan::Image<'a>; 2],

    should_threads_die: Arc<RwLock<bool>>,
    notify_transfer_wait: mpsc::Receiver<u64>,
    notify_complete_frame: mpsc::Sender<u8>,
    transfer_semaphore: vk::Semaphore,
) {
    log::trace!("Creating thread");
    let compute = match Compute::new(device, queue_family_indices, images) {
        Ok(v) => v,
        Err(e) => panic!("Failed to create compute object: {}", e),
    };

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
                log::error!("rwlock is poisoned, ending thread: {e}")
            }
        }

        if !first_run {
            match core::block_on_semaphore(
                &compute.device,
                compute.semaphore.get(),
                current_timestamp,
                10_000_000, // 10ms
            ) {
                Err(vk::Result::TIMEOUT) => continue,
                Err(e) => {
                    log::error!("Failed to wait for compute task to finish: {e}");
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

        // Perform a render
        // TODO: Measure time taken
        match compute.perform_compute(
            current_frame_index,
            transfer_semaphore,
            current_transfer_timestamp,
            current_timestamp,
            current_timestamp + 1,
        ) {
            Err(e) => {
                log::error!("Failed to perform compute: {}", e)
            }
            _ => {}
        };
        first_run = false;
        current_timestamp += 1;
    }
    log::warn!("Ending thread");
}

#[allow(dead_code)]
struct Compute<'a> {
    pub device: ash::Device,
    semaphore: Destructor<vk::Semaphore>,

    radiance_images: [vulkan::Image<'a>; 2],

    queue: vk::Queue,
    descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    pipeline_layout: Destructor<vk::PipelineLayout>,
    pipeline: Destructor<vk::Pipeline>,
    command_pool: Destructor<vk::CommandPool>,
    commands: [vk::CommandBuffer; 2],
    descriptor_pool: Destructor<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl<'a> Compute<'a> {
    pub fn new(
        device: ash::Device,
        queue_family_indices: structures::QueueFamilyIndices,
        radiance_images: [vulkan::Image<'a>; 2],
    ) -> Result<Self> {
        log::trace!("Creating object");
        // Populates queue
        let queue = core::create_queue(&device, queue_family_indices.compute_family.unwrap());

        let (descriptor_pool, descriptor_set_layout, descriptor_sets, pipeline_layout, pipeline) =
            create_pipeline(&device, &radiance_images)?;
        // Populates command_pool, commands, and copy_commands
        let (command_pool, commands) = create_commands(
            &device,
            queue_family_indices,
            &descriptor_sets,
            pipeline_layout.get(),
            pipeline.get(),
        )?;
        // populates semaphore
        let semaphore = Destructor::new(
            &device,
            core::create_semaphore(&device)?,
            device.fp_v1_0().destroy_semaphore,
        );

        Ok(Self {
            device,
            semaphore,
            radiance_images,
            queue,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            command_pool,
            commands,
            descriptor_pool,
            descriptor_sets,
        })
    }
    fn perform_compute(
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
}

fn create_pipeline(
    device: &ash::Device,
    radiance_images: &[vulkan::Image; 2],
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
            .image_view(radiance_images[i].view())];

        let storage_image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(radiance_images[(i + 1) % 2].view())];

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

    // let shader_code = read_shader_code(Path::new("shaders/spv/ray.comp.spv"))?;
    let shader_code = read_shader_code(Path::new("shaders/spv/ray.slang.spv"))?;
    let shader_module = core::create_shader_module(&device, &shader_code)?;
    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.
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
            Ok(v) => v[0], // We are only creating one timeline so we only want the first object in the vector
            Err(e) => return Err(anyhow!(e.1)),
        },
        device.fp_v1_0().destroy_pipeline,
    );

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

    let commands = record_commands(
        device,
        pipeline_layout,
        pipeline,
        &descriptor_sets,
        &command_buffers,
    )?;

    Ok((command_pool, commands))
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
        // TODO: Regenerate these commands when the window is resized so we can change these group counts
        device.cmd_dispatch(
            command_buffer,
            WINDOW_WIDTH / 32 + 1,
            WINDOW_HEIGHT / 32 + 1,
            1,
        );
        device.end_command_buffer(command_buffer)?;
    }
    Ok(command_buffer)
}

fn record_commands(
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
