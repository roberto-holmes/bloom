use anyhow::{anyhow, Result};
use ash::vk;
use bus::Bus;
use std::{
    sync::{mpsc, Arc, RwLock},
    thread::JoinHandle,
    time::{Duration, Instant},
    u64,
};
use winit::dpi::PhysicalSize;

use crate::{
    api::{self, Bloomable},
    core, ray, structures, uniforms, viewport,
    vulkan::Destructor,
    MAX_FRAMES_IN_FLIGHT,
};

pub enum ResizedSource {
    Viewport((PhysicalSize<u32>, [vk::Image; 2])),
    Ray((PhysicalSize<u32>, [vk::Image; 2])),
}

pub fn thread<T: Bloomable>(
    user_app: T,
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: structures::QueueFamilyIndices,
    semaphore: vk::Semaphore,
    should_threads_die: Arc<RwLock<bool>>,

    should_systems_die: Arc<RwLock<bool>>,
    mut event_broadcaster: Bus<api::Event>,
    mut system_threads: Vec<Option<JoinHandle<()>>>,
    system_sync: (mpsc::Sender<bool>, mpsc::Receiver<bool>), // TODO: Replace bool with a thread identifier

    ray_frame_done: Arc<RwLock<ray::Update>>,
    draw_request: mpsc::Receiver<bool>,
    mut resize_request: single_value_channel::Receiver<PhysicalSize<u32>>,

    resized: mpsc::Receiver<ResizedSource>,
    ubo: mpsc::Sender<uniforms::Event>,
    ray: mpsc::SyncSender<ray::TransferCommand>,
    viewport: mpsc::SyncSender<viewport::TransferCommand>,
    ray_resize: single_value_channel::Updater<PhysicalSize<u32>>,
    viewport_resize: single_value_channel::Updater<PhysicalSize<u32>>,
) {
    log::trace!("Creating thread");
    let mut transfer = match Transfer::new(
        device,
        instance,
        physical_device,
        semaphore,
        queue_family_indices,
        ubo,
        ray,
        viewport,
        ray_resize,
        viewport_resize,
        resized,
    ) {
        Err(e) => {
            log::error!("Failed to create transfer object: {e}");
            return;
        }
        Ok(t) => t,
    };

    let mut output_frame_index = 0;
    let mut timestamp = 0;
    let mut is_minimised = false;
    let mut was_minimised = false;
    let mut old_size = PhysicalSize {
        width: 0,
        height: 0,
    };

    let mut last_physics_time = Instant::now();

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

        // Check if the window has changed size
        let new_size = *resize_request.latest();
        if new_size != old_size {
            if let Err(e) = resize(
                &mut transfer,
                &mut old_size,
                new_size,
                &mut is_minimised,
                &mut was_minimised,
                timestamp,
            ) {
                log::error!("Failed to resize: {e}");
                break;
            }
        }

        // Check if it's time to present a new frame
        match draw_request.try_recv() {
            Ok(true) => {
                if let Err(e) = draw(
                    &mut transfer,
                    is_minimised,
                    timestamp,
                    output_frame_index,
                    &ray_frame_done,
                ) {
                    log::error!("Failed to draw: {e}");
                    break;
                }
                if let Err(e) = broadcast(
                    api::Event::GraphicsUpdate,
                    &mut event_broadcaster,
                    &system_threads,
                    &system_sync.1,
                ) {
                    log::error!("Failed to broadcast graphics update: {e}");
                    break;
                }
                timestamp += 1;
                output_frame_index += 1;
                output_frame_index %= MAX_FRAMES_IN_FLIGHT;
            }
            Ok(false) => {}
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Draw channel has disconnected");
                break;
            }
        }

        // Check if physics needs updating
        if user_app.get_physics_update_period() <= last_physics_time.elapsed() {
            if let Err(e) = broadcast(
                api::Event::PrePhysics,
                &mut event_broadcaster,
                &system_threads,
                &system_sync.1,
            ) {
                log::error!("Failed to broadcast pre physics update: {e}");
                break;
            }
            if let Err(e) = broadcast(
                api::Event::Physics,
                &mut event_broadcaster,
                &system_threads,
                &system_sync.1,
            ) {
                log::error!("Failed to broadcast physics update: {e}");
                break;
            }
            if let Err(e) = broadcast(
                api::Event::PostPhysics,
                &mut event_broadcaster,
                &system_threads,
                &system_sync.1,
            ) {
                log::error!("Failed to broadcast post physics update: {e}");
                break;
            }
            last_physics_time = Instant::now();
        }
    }

    *should_systems_die.write().unwrap() = true;
    for ot in system_threads.iter_mut() {
        if let Some(t) = ot.take() {
            t.join().unwrap();
        };
    }
}

fn broadcast(
    event: api::Event,
    event_broadcaster: &mut Bus<api::Event>,
    system_threads: &Vec<Option<JoinHandle<()>>>,
    system_sync_receiver: &mpsc::Receiver<bool>,
) -> Result<()> {
    // Broadcast graohics update and wait for all systems to finish
    event_broadcaster.broadcast(event);

    let total_systems: u32 = system_threads
        .iter()
        .map(|x| match x {
            Some(_) => 1,
            None => 0,
        })
        .sum();

    let mut finished_threads = 0;
    while finished_threads < total_systems {
        match system_sync_receiver.recv() {
            Ok(true) => finished_threads += 1,
            Ok(false) => log::warn!("Invalid response from thread sync channel"),
            Err(e) => return Err(anyhow!("Failed to receive sync from a system: {e}")),
        }
    }
    Ok(())
}

fn resize(
    transfer: &mut Transfer,
    old_size: &mut PhysicalSize<u32>,
    new_size: PhysicalSize<u32>,
    is_minimised: &mut bool,
    was_minimised: &mut bool,
    timestamp_to_wait: u64,
) -> Result<()> {
    *old_size = new_size;
    if new_size.width == 0 || new_size.height == 0 {
        log::debug!("Window is minimized");
        *is_minimised = true;
        *was_minimised = true;
    } else {
        *is_minimised = false;
    }
    match transfer.resize(
        new_size,
        !*is_minimised && !*was_minimised,
        timestamp_to_wait,
    ) {
        Ok((true, ray_images, viewport_images)) => {
            log::trace!(
                "Have ray images {:?} and viewport images {:?}",
                ray_images,
                viewport_images
            );
            transfer.update_commands(&ray_images, &viewport_images, new_size)?;
        }
        Ok((false, _, _)) => {
            log::info!("Not rebuilding transfer commands");
        }
        Err(e) => return Err(e),
    }
    if *was_minimised == true && *is_minimised == false {
        *was_minimised = false;
    }
    return Ok(());
}

fn draw(
    transfer: &mut Transfer,
    is_minimised: bool,
    timestamp: u64,
    output_frame_index: usize,
    ray_frame_done: &Arc<RwLock<ray::Update>>,
) -> Result<()> {
    if is_minimised {
        return Ok(());
    }
    log::trace!("Drawing");
    // Notify ray to pause any further ray tracing until the image is free to use
    transfer
        .ray
        .send(ray::TransferCommand::Pause(timestamp + 1))?;

    let (ray_frame, accumulated_frame_count) = match ray_frame_done.read() {
        Ok(v) => (v.current_frame_index, v.accumulated_frames),
        Err(e) => {
            return Err(anyhow!(
                "Transfer failed to receive lastest frame from ray: {e}"
            ));
        }
    };

    // Let the viewport know when the data will be ready and which frame to use
    transfer.viewport.send(viewport::TransferCommand {
        timestamp: timestamp + 1,
        frame_index: output_frame_index,
        accumulated_frame_count,
    })?;

    // Perform a copy
    transfer.perform_compute_copy(
        output_frame_index,
        ray_frame as usize,
        timestamp,
        timestamp + 1,
    )?;
    Ok(())
}
struct Transfer {
    ubo: mpsc::Sender<uniforms::Event>,
    ray: mpsc::SyncSender<ray::TransferCommand>,
    viewport: mpsc::SyncSender<viewport::TransferCommand>,
    ray_resize: single_value_channel::Updater<PhysicalSize<u32>>,
    viewport_resize: single_value_channel::Updater<PhysicalSize<u32>>,
    resized: mpsc::Receiver<ResizedSource>,

    last_timestamp: u64,
    queue: vk::Queue,
    semaphore: vk::Semaphore,
    #[allow(dead_code)]
    command_pool: Destructor<vk::CommandPool>,
    commands: [vk::CommandBuffer; 4],
    pub device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
}

impl Transfer {
    pub fn new(
        device: ash::Device,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        semaphore: vk::Semaphore,
        queue_family_indices: structures::QueueFamilyIndices,

        ubo: mpsc::Sender<uniforms::Event>,
        ray: mpsc::SyncSender<ray::TransferCommand>,
        viewport: mpsc::SyncSender<viewport::TransferCommand>,
        ray_resize: single_value_channel::Updater<PhysicalSize<u32>>,
        viewport_resize: single_value_channel::Updater<PhysicalSize<u32>>,
        resized: mpsc::Receiver<ResizedSource>,
    ) -> Result<Self> {
        log::trace!("Creating object");
        let queue = core::create_queue(&device, queue_family_indices.transfer_family.unwrap());
        let (command_pool, commands) = create_commands(&device, &queue_family_indices)?;
        Ok(Self {
            last_timestamp: 0,
            device,
            instance,
            physical_device,
            queue,
            semaphore,
            command_pool,
            commands,

            ubo,
            ray,
            viewport,
            ray_resize,
            viewport_resize,
            resized,
        })
    }
    pub fn perform_compute_copy(
        &mut self,
        viewport_frame_index: usize,
        ray_frame_index: usize,
        timestamp_to_wait: u64,
        timestamp_to_signal: u64,
    ) -> Result<()> {
        let wait_timestamps = [timestamp_to_wait];
        let signal_timestamps = [timestamp_to_signal];
        let semaphores = [self.semaphore];
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];

        let command_index = viewport_frame_index << 1 | ray_frame_index;
        let command_buffer = [self.commands[command_index]];

        // Only allow one copy operation to be in flight
        let last_timestamp = if timestamp_to_wait > 1 {
            timestamp_to_wait - 1
        } else {
            0
        };

        match core::wait_on_semaphore(&self.device, self.semaphore, last_timestamp, 100_000_000) {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => log::warn!("Semaphore timed out"),
            Err(e) => {
                log::error!("Failed to wait for compute copy to complete: {:?}", e)
            }
        }

        log::trace!(
            "Copying {ray_frame_index} to {viewport_frame_index} (command {command_index}/{:?}) - Waiting on timestamp {}, will signal {}",
            command_buffer[0],
            timestamp_to_wait,
            timestamp_to_signal
        );

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
        self.last_timestamp = timestamp_to_signal;
        Ok(())
    }
    fn update_commands(
        &mut self,
        ray_images: &[vk::Image; 2],
        viewport_images: &[vk::Image; 2],
        size: PhysicalSize<u32>,
    ) -> Result<()> {
        for ray_frame_index in 0..2 {
            for viewport_frame_index in 0..2 {
                self.copy_commands(
                    ray_frame_index,
                    viewport_frame_index,
                    ray_images,
                    viewport_images,
                    size,
                )?;
            }
        }
        log::info!("Prepared commands {:?}", self.commands);
        Ok(())
    }

    fn copy_commands(
        &mut self,
        ray_frame_index: usize,
        viewport_frame_index: usize,
        ray_images: &[vk::Image; 2],
        viewport_images: &[vk::Image; 2],
        size: PhysicalSize<u32>,
    ) -> Result<()> {
        let command_buffer = self.commands[viewport_frame_index << 1 | ray_frame_index];

        let subresource_range = [vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }];

        let mut ray_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(ray_images[ray_frame_index])
            .subresource_range(subresource_range[0])
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let mut viewport_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(viewport_images[viewport_frame_index])
            .subresource_range(subresource_range[0])
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        let mut image_memory_barriers = [ray_barrier, viewport_barrier];

        let subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };

        let width = std::cmp::min(
            size.width,
            core::get_max_image_size(&self.instance, self.physical_device),
        );
        let height = std::cmp::min(
            size.height,
            core::get_max_image_size(&self.instance, self.physical_device),
        );

        log::trace!("Copy region is {width}x{height}");

        let regions = [vk::ImageCopy {
            src_subresource: subresource,
            dst_subresource: subresource,
            src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        }];

        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barriers,
            );
            self.device.cmd_copy_image(
                command_buffer,
                ray_images[ray_frame_index],
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                viewport_images[viewport_frame_index],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }

        ray_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        ray_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
        viewport_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        viewport_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
        ray_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        ray_barrier.new_layout = vk::ImageLayout::GENERAL;
        viewport_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        viewport_barrier.new_layout = vk::ImageLayout::GENERAL;

        image_memory_barriers = [ray_barrier, viewport_barrier]; // TODO: Is this necessary or will the changes already be here?

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barriers,
            );

            let clear_value = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            };

            // Clear old ray image
            self.device.cmd_clear_color_image(
                command_buffer,
                ray_images[ray_frame_index],
                vk::ImageLayout::GENERAL,
                &clear_value,
                &subresource_range,
            );

            self.device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }
    fn resize(
        &self,
        new_size: PhysicalSize<u32>,
        need_response: bool,
        timestamp_to_wait: u64,
    ) -> Result<(bool, [vk::Image; 2], [vk::Image; 2])> {
        // Wait for the previous copies to finish so the images are not in use
        match core::wait_on_semaphore(&self.device, self.semaphore, timestamp_to_wait, 100_000_000)
        {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => log::warn!("Semaphore timed out"),
            Err(e) => {
                log::error!("Failed to wait for compute copy to complete: {:?}", e)
            }
        }

        self.ray_resize.update(new_size)?;
        self.viewport_resize.update(new_size)?;
        self.ubo.send(uniforms::Event::Resize(new_size))?;
        log::info!("Sent size {}x{}", new_size.width, new_size.height);

        if !need_response {
            return Ok((false, [vk::Image::null(); 2], [vk::Image::null(); 2]));
        }

        // Check that we get a confirmation from Ray and Viewport
        let mut ray_resized = false;
        let mut viewport_resized = false;

        let mut ray_images = None;
        let mut viewport_images = None;

        loop {
            match self.resized.recv_timeout(Duration::new(1, 000_000_000)) {
                Ok(ResizedSource::Ray((reported_size, v))) if reported_size == new_size => {
                    ray_resized = true;
                    ray_images = Some(v);
                    if viewport_resized {
                        break;
                    }
                }
                Ok(ResizedSource::Ray((reported_size, _))) => {
                    log::warn!(
                        "Ray reported it has resized to {}x{} but expected {}x{}",
                        reported_size.width,
                        reported_size.height,
                        new_size.width,
                        new_size.height
                    )
                }
                Ok(ResizedSource::Viewport((reported_size, v))) if reported_size == new_size => {
                    viewport_resized = true;
                    viewport_images = Some(v);
                    if ray_resized {
                        break;
                    }
                }
                Ok(ResizedSource::Viewport((reported_size, _))) => {
                    log::warn!(
                        "Viewport reported it has resized to {}x{} but expected {}x{}",
                        reported_size.width,
                        reported_size.height,
                        new_size.width,
                        new_size.height
                    )
                }
                Err(e) => {
                    return Err(anyhow!(
                        "Resize to {}x{} (ray - {}, viewport - {}): {e}",
                        new_size.width,
                        new_size.height,
                        if ray_resized {
                            "completed"
                        } else {
                            "timed out"
                        },
                        if viewport_resized {
                            "completed"
                        } else {
                            "timed out"
                        }
                    ));
                }
            }
        }

        Ok((true, ray_images.unwrap(), viewport_images.unwrap()))
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        // TODO: Set timeline semaphore to max value
        let signal_info = vk::SemaphoreSignalInfo {
            semaphore: self.semaphore,
            value: i32::MAX as u64,
            ..Default::default()
        };
        unsafe {
            match core::wait_on_semaphore(
                &self.device,
                self.semaphore,
                self.last_timestamp,
                100_000_000,
            ) {
                Ok(()) => {}
                Err(vk::Result::TIMEOUT) => log::warn!("Semaphore timed out"),
                Err(e) => {
                    log::error!("Failed to wait for compute copy to complete: {:?}", e)
                }
            }
            match self.device.signal_semaphore(&signal_info) {
                Err(e) => log::error!("Transfer failed to signal the end of the semaphore: {e}"),
                _ => {}
            }
        }
        log::trace!("Cleaned up Transfer");
    }
}

fn create_commands(
    device: &ash::Device,
    queue_family_indices: &structures::QueueFamilyIndices,
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

    Ok((
        command_pool,
        [
            command_buffers[0],
            command_buffers[1],
            command_buffers[2],
            command_buffers[3],
        ],
    ))
}
