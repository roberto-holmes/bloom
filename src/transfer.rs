use std::{
    sync::{mpsc, Arc, Mutex, RwLock},
    time::Duration,
};

use anyhow::{anyhow, Result};
use ash::vk;

use crate::{
    api::BloomAPI, core, structures, vulkan::Destructor, IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT,
    IDEAL_RADIANCE_IMAGE_SIZE_WIDTH,
};

pub fn thread(
    device: ash::Device,
    queue_family_indices: structures::QueueFamilyIndices,
    semaphore: vk::Semaphore,
    compute_images: &[vk::Image; 2],
    graphic_images: &[vk::Image; 2],

    should_threads_die: Arc<RwLock<bool>>,
    listener: mpsc::Receiver<u8>,
    compute_channel: mpsc::Sender<u64>,
    graphic_channel: mpsc::Sender<u64>,
    api: Arc<Mutex<BloomAPI>>,
) {
    log::trace!("Creating thread");
    let transfer = match Transfer::new(
        device,
        semaphore,
        queue_family_indices,
        compute_images,
        graphic_images,
    ) {
        Err(e) => {
            log::error!("Failed to create transfer object: {e}");
            return;
        }
        Ok(t) => t,
    };

    let mut compute_valid_frame_index = 0;
    let mut timestamp = 0;

    loop {
        // Check if we should end the thread
        match should_threads_die.read() {
            Ok(should_die) => {
                if *should_die == true {
                    break;
                }
            }
            Err(e) => {
                log::error!("Transfer rwlock is poisoned, ending thread: {}", e)
            }
        }

        // Wait for either Graphics or Compute queues to be ready
        let graphic_valid_frame_index = match listener.recv_timeout(Duration::new(0, 10_000_000)) {
            Ok(v) => {
                if v & 0x80 != 0 {
                    log::trace!("Received new compute frame");
                    // If received from compute, make note of the current compute frame that can be used to copy
                    compute_valid_frame_index = v & !0x80;
                    api.lock().expect("Failed to unlock API").uniform.tick_ray();
                    continue;
                }
                log::trace!("Received new graphic frame");
                api.lock().expect("Failed to unlock API").uniform.tick();
                // else begin the copy process and notify compute and graphics of when they can begin doing stuff
                v
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::error!("Transfer channel has disconnected");
                break;
            }
        };

        // Notify compute to pause any further computations
        match compute_channel.send(timestamp + 1) {
            Err(e) => {
                log::error!(
                    "Transfer failed to notify compute channel that it should pause compute: {e}"
                )
            }
            Ok(()) => {}
        }
        // Let graphics know when the data will be ready
        match graphic_channel.send(timestamp + 1) {
            Err(e) => {
                log::error!("Transfer failed to notify graphics channel: {e}")
            }
            Ok(()) => {}
        }

        // log::trace!(
        //     "Copying compute {compute_valid_frame_index} to graphic {graphic_valid_frame_index}"
        // );

        // Perform a copy
        match transfer.perform_compute_copy(
            graphic_valid_frame_index as usize,
            compute_valid_frame_index as usize,
            timestamp,
            timestamp + 1,
        ) {
            Err(e) => {
                log::error!("Failed to perform copy operation {e}")
            }
            Ok(()) => {}
        }
        timestamp += 1;
    }
}

struct Transfer {
    pub device: ash::Device,
    queue: vk::Queue,
    semaphore: vk::Semaphore,
    #[allow(dead_code)]
    command_pool: Destructor<vk::CommandPool>,
    commands: [vk::CommandBuffer; 4],
}

impl Transfer {
    pub fn new(
        device: ash::Device,
        semaphore: vk::Semaphore,
        queue_family_indices: structures::QueueFamilyIndices,
        compute_images: &[vk::Image; 2],
        graphic_images: &[vk::Image; 2],
    ) -> Result<Self> {
        log::trace!("Creating object");
        let queue = core::create_queue(&device, queue_family_indices.transfer_family.unwrap());
        let (command_pool, commands) = create_commands(
            &device,
            queue_family_indices,
            compute_images,
            graphic_images,
        )?;
        Ok(Self {
            device,
            queue,
            semaphore,
            command_pool,
            commands,
        })
    }
    pub fn perform_compute_copy(
        &self,
        graphic_frame_index: usize,
        compute_frame_index: usize,
        timestamp_to_wait: u64,
        timestamp_to_signal: u64,
    ) -> Result<()> {
        let wait_timestamps = [timestamp_to_wait];
        let signal_timestamps = [timestamp_to_signal];
        let semaphores = [self.semaphore];
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];

        let command_index = 1 << graphic_frame_index | compute_frame_index;
        let command_buffer = [self.commands[command_index]];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_timestamps)
            .wait_semaphore_values(&wait_timestamps);
        let submit_info = [vk::SubmitInfo::default()
            .command_buffers(&command_buffer)
            .signal_semaphores(&semaphores)
            .wait_semaphores(&semaphores)
            .wait_dst_stage_mask(&wait_stages)
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
}

impl Drop for Transfer {
    fn drop(&mut self) {
        unsafe {
            match self.device.queue_wait_idle(self.queue) {
                Err(e) => log::error!("Transfer failed to wait for its queue to finish: {e}"),
                _ => {}
            }
        }
    }
}

fn create_commands(
    device: &ash::Device,
    queue_family_indices: structures::QueueFamilyIndices,
    compute_images: &[vk::Image; 2],
    graphic_images: &[vk::Image; 2],
) -> Result<(Destructor<vk::CommandPool>, [vk::CommandBuffer; 4])> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_indices.transfer_family.unwrap().0);
    let command_pool = Destructor::new(
        device,
        unsafe { device.create_command_pool(&pool_info, None)? },
        device.fp_v1_0().destroy_command_pool,
    );

    let command_buffers = core::create_command_buffers(device, command_pool.get(), 4)?;

    let copy_commands =
        record_copy_commands(device, &command_buffers, &compute_images, &graphic_images)?;

    Ok((command_pool, copy_commands))
}

fn record_copy_commands(
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
