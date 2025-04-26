use std::{
    ffi::CString,
    path::Path,
    sync::{mpsc, Arc, RwLock},
    time::Duration,
};

use anyhow::{anyhow, Result};
use ash::vk;
use memoffset::offset_of;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    core::{self, transition_image_layout},
    structures::{self, SwapChainStuff},
    tools::read_shader_code,
    transfer,
    uniforms::{self, UniformBufferObject},
    vulkan::{self, Destructor},
    MAX_FRAMES_IN_FLIGHT,
};

pub enum TransferCommand {
    Ready(u64, usize), // Timeline semaphore timestamp
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct Vertex {
    pos: [f32; 3],
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }
    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 1] {
        [vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Self, pos) as u32)]
    }
}

// Vertices required to cover the viewport
const VERTICES: [Vertex; 4] = [
    Vertex {
        pos: [-1.0, -1.0, 0.0],
    },
    Vertex {
        pos: [1.0, -1.0, 0.0],
    },
    Vertex {
        pos: [1.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-1.0, 1.0, 0.0],
    },
];

pub const INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

pub fn thread(
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: structures::SurfaceStuff,
    window: Arc<RwLock<Window>>,

    queue_family_indices: structures::QueueFamilyIndices,

    uniform_buffers: [vk::Buffer; 2],

    transfer_sender: mpsc::Sender<transfer::ResizedSource>,
    transfer_semaphore: vk::Semaphore,
    viewport_semaphore: vk::Semaphore,

    should_threads_die: Arc<RwLock<bool>>,
    event_receiver: mpsc::Receiver<TransferCommand>,
    mut resize_receiver: single_value_channel::Receiver<PhysicalSize<u32>>,
    latest_frame_index: single_value_channel::Updater<usize>,
) {
    log::trace!("Creating thread");
    let mut allocator_create_info =
        vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
    allocator_create_info.vulkan_api_version = vk::API_VERSION_1_3;
    allocator_create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

    // TODO: Use our own allocator or should everyone use the same one?
    let allocator = Arc::new(
        match unsafe { vk_mem::Allocator::new(allocator_create_info) } {
            Ok(v) => v,
            Err(e) => {
                panic!("Failed to create an allocator: {:?}", e);
            }
        },
    );

    // Build viewport object
    let mut viewport = match Viewport::new(
        window,
        device,
        instance,
        physical_device,
        allocator,
        queue_family_indices,
        surface_stuff,
        uniform_buffers,
    ) {
        Ok(v) => v,
        Err(e) => {
            log::error!("Failed to create viewport object: {e}");
            return;
        }
    };

    let mut active_frame_index = 0;
    let mut was_minimised = false;

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

        let size = resize_receiver.latest();

        if size.width == 0 || size.height == 0 {
            was_minimised = true;
            // If we are minimised then we shouldn't bother doing everything else
            std::thread::sleep(Duration::from_millis(10));
            continue;
        }

        // We only want to send an update if we've actually resized
        if let (true, new_images) = match viewport.resize(*size, active_frame_index, was_minimised)
        {
            Err(e) => {
                log::error!("Failed to create new output images: {e}");
                break;
            }
            Ok(v) => v,
        } {
            log::info!(
                "Images {:?} are now {}x{}",
                new_images,
                size.width,
                size.height
            );
            match transfer_sender.send(transfer::ResizedSource::Viewport((*size, new_images))) {
                Err(e) => log::error!("Failed to update transfer with new images: {e}"),
                Ok(()) => {}
            };
        }

        // Check for new event
        match event_receiver.recv_timeout(Duration::new(0, 10_000_000)) {
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::error!("Viewport event channel has disconnected");
                break;
            }

            Ok(TransferCommand::Ready(timestamp, frame)) => {
                active_frame_index = frame;
                match viewport.draw(transfer_semaphore, timestamp, viewport_semaphore, frame) {
                    Err(e) => {
                        log::error!("Failed to draw frame: {e}");
                        break;
                    }
                    Ok(false) => log::warn!("Need to redraw"),
                    Ok(true) => {
                        if let Err(e) = latest_frame_index.update(frame) {
                            log::error!("Failed to send latest frame index to uniforms: {e}");
                            break;
                        }
                    }
                }
            }
        }
        was_minimised = false;
    }
}

struct Viewport<'a> {
    image_available: [Destructor<vk::Semaphore>; 2],
    render_finished: [Destructor<vk::Semaphore>; 2],
    in_flight_fences: [Destructor<vk::Fence>; 2],
    images: Option<[vulkan::Image<'a>; 2]>,
    uniform_buffers: [vk::Buffer; 2],

    vertex_buffer: vulkan::Buffer,
    index_buffer: vulkan::Buffer,

    queue_family_indices: structures::QueueFamilyIndices,
    queue: vk::Queue,
    descriptor_set_layout: Destructor<vk::DescriptorSetLayout>,
    pipeline_layout: Destructor<vk::PipelineLayout>,
    pipeline: Destructor<vk::Pipeline>,
    command_pool: Destructor<vk::CommandPool>,
    commands: [vk::CommandBuffer; 2],
    descriptor_pool: Destructor<vk::DescriptorPool>,
    descriptor_sets: Option<[vk::DescriptorSet; 2]>,
    swapchain_stuff: structures::SwapChainStuff,
    surface_stuff: structures::SurfaceStuff,

    query_pool_timestamps: Destructor<vk::QueryPool>,
    timestamps: Vec<u64>,

    window: Arc<RwLock<Window>>,

    allocator: Arc<vk_mem::Allocator>,
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
}

impl<'a> Viewport<'a> {
    pub fn new(
        window: Arc<RwLock<Window>>,
        device: ash::Device,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        allocator: Arc<vk_mem::Allocator>,
        queue_family_indices: structures::QueueFamilyIndices,
        surface_stuff: structures::SurfaceStuff,
        uniform_buffers: [vk::Buffer; 2],
    ) -> Result<Self> {
        let queue = core::create_queue(&device, queue_family_indices.graphics_family.unwrap());
        let swapchain_stuff = SwapChainStuff::new(
            &instance,
            &device,
            physical_device,
            &surface_stuff,
            &queue_family_indices,
        )?;
        let (query_pool_timestamps, timestamps) = core::prepare_timestamp_queries(&device)?;
        let set_layout = create_descriptor_set_layout(&device)?;
        let descriptor_pool = create_descriptor_pool(&device)?;
        let (pipeline_layout, pipeline) =
            create_pipeline(&device, &swapchain_stuff, set_layout.get())?;

        let command_pool =
            core::create_command_pool(&device, queue_family_indices.graphics_family.unwrap().0)?;

        let vertex_buffer = vulkan::Buffer::new_populated_staged(
            &device,
            command_pool.get(),
            queue,
            &allocator,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            VERTICES.as_ptr(),
            VERTICES.len(),
        )?;

        let index_buffer = vulkan::Buffer::new_populated_staged(
            &device,
            command_pool.get(),
            queue,
            &allocator,
            vk::BufferUsageFlags::INDEX_BUFFER,
            INDICES.as_ptr(),
            INDICES.len(),
        )?;

        let commands =
            core::create_command_buffers(&device, command_pool.get(), MAX_FRAMES_IN_FLIGHT as u32)?;

        let (image_available, render_finished, in_flight_fences) = create_sync_object(&device)?;

        Ok(Self {
            device,
            instance,
            allocator,
            physical_device,
            images: None,
            uniform_buffers,
            vertex_buffer,
            index_buffer,
            queue_family_indices,
            queue,
            descriptor_set_layout: set_layout,
            pipeline_layout,
            pipeline,
            command_pool,
            commands: [commands[0], commands[1]],
            descriptor_pool,
            descriptor_sets: None,
            swapchain_stuff,
            surface_stuff,
            query_pool_timestamps,
            timestamps,
            image_available,
            render_finished,
            in_flight_fences,
            window,
        })
    }
    fn initial_size(&mut self, size: PhysicalSize<u32>) -> Result<[vk::Image; 2]> {
        log::trace!("Creating initial images and commands");
        // Recreate swap chain
        self.recreate_swap_chain();
        // Recreate output images
        self.images = Some(core::create_storage_image_pair(
            &self.device,
            &self.instance,
            &self.allocator,
            self.physical_device,
            self.command_pool.get(),
            self.queue,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            size,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        )?);
        // Record new commands
        self.descriptor_sets = Some(create_descriptor_sets(
            &self.device,
            self.descriptor_pool.get(),
            self.descriptor_set_layout.get(),
            &self.uniform_buffers,
            self.images.as_ref().unwrap(),
        )?);

        let mut raw_images = [vk::Image::null(); 2];
        for i in 0..self.images.as_ref().unwrap().len() {
            raw_images[i] = self.images.as_ref().unwrap()[i].get();
        }
        let mut raw_images = [vk::Image::null(); 2];
        for i in 0..self.images.as_ref().unwrap().len() {
            raw_images[i] = self.images.as_ref().unwrap()[i].get();
        }
        Ok(raw_images)
    }
    fn resize(
        &mut self,
        new_size: PhysicalSize<u32>,
        active_frame: usize,
        was_minimised: bool,
    ) -> Result<(bool, [vk::Image; 2])> {
        if was_minimised {
            self.recreate_swap_chain();
        }

        if self.images.is_none() {
            return Ok((true, self.initial_size(new_size)?));
        }

        let mut raw_images = [vk::Image::null(); 2];
        // Check if both images are already the correct size
        if self.images.as_ref().unwrap()[0].is_correct_size(new_size.width, new_size.height)
            && self.images.as_ref().unwrap()[1].is_correct_size(new_size.width, new_size.height)
        {
            return Ok((false, raw_images));
        }
        // Wait for the previous frame to finish
        unsafe {
            self.device.wait_for_fences(
                &[self.in_flight_fences[active_frame].get()],
                true,
                100_000_000,
            )?; // 100ms timeout
        }
        if !was_minimised {
            self.recreate_swap_chain();
        }

        for image in self.images.as_mut().unwrap().iter_mut() {
            let _ = image.resize(new_size.width, new_size.height)?;
            core::transition_image_layout(
                &self.device,
                self.command_pool.get(),
                self.queue,
                image.get(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            )?;
        }
        log::debug!("Resizing to {}x{}", new_size.width, new_size.height);
        for (i, &descriptor_set) in self.descriptor_sets.unwrap().iter().enumerate() {
            // Now need to update the descriptor to reference the new image
            let image_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(self.images.as_ref().unwrap()[i].view())];

            let descriptor_writes = [vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                ..Default::default()
            }
            .image_info(&image_info)];
            unsafe { self.device.update_descriptor_sets(&descriptor_writes, &[]) };
        }
        for i in 0..self.images.as_ref().unwrap().len() {
            raw_images[i] = self.images.as_ref().unwrap()[i].get();
        }

        Ok((true, raw_images))
    }
    /// Returns whether we successfully presented an image (or if we need to redraw)
    fn draw(
        &mut self,
        transfer_semaphore: vk::Semaphore,
        wait_timestamp: u64,
        viewport_semaphore: vk::Semaphore,
        frame: usize,
    ) -> Result<bool> {
        let start = std::time::Instant::now();
        if self.images.is_none() {
            return Err(anyhow!("Tried to draw with an uninitialised image"));
        }

        log::trace!(
            "[{} μs] Drawing {frame} once timestamp reaches {}",
            start.elapsed().as_micros(),
            wait_timestamp
        );

        // Wait for the previous frame to finish
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fences[frame].get()], true, u64::MAX)?;
        }
        log::trace!(
            "[{} μs] Back from fences [{:.2} fps]",
            start.elapsed().as_micros(),
            1.0 / start.elapsed().as_secs_f32()
        );
        let swapchain_index = match unsafe {
            self.swapchain_stuff.get_loader().acquire_next_image(
                self.swapchain_stuff.get_swapchain(),
                1_000_000_000, // 1 second in nanoseconds
                self.image_available[frame].get(),
                vk::Fence::null(),
            )
        } {
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                log::warn!("Swapchain reports ERROR_OUT_OF_DATE_KHR",);
                self.recreate_swap_chain();
                return Ok(false);
            }
            Err(e) => {
                return Err(anyhow!("Failed to get swapchain image: {e}"));
            }
            Ok((v, _is_suboptimal)) => v,
        };
        unsafe {
            self.device
                .reset_fences(&[self.in_flight_fences[frame].get()])?
        };

        self.record_commands(swapchain_index as usize, frame)?;

        let wait_semaphores = [self.image_available[frame].get(), transfer_semaphore];
        let wait_stages = [
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
        ];
        let wait_values = [
            0, // Binary so will be ignored
            wait_timestamp,
        ];
        let submit_signal_semaphores = [self.render_finished[frame].get(), viewport_semaphore];
        let command_buffers = [self.commands[frame]];

        // We want to forward the timestamp value when we complete so we wait and submit the same value
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&wait_values);

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&submit_signal_semaphores)
            .push(&mut timeline_info);

        unsafe {
            self.device.queue_submit(
                self.queue,
                &[submit_info],
                self.in_flight_fences[frame].get(),
            )?
        };
        let present_wait_semaphores = [self.render_finished[frame].get()];

        let swapchains = [self.swapchain_stuff.get_swapchain()];
        let image_indices = [swapchain_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&present_wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // Tell the window that we are about to present so that it can prepare
        self.window.write().unwrap().pre_present_notify();

        match unsafe {
            self.swapchain_stuff
                .get_loader()
                .queue_present(self.queue, &present_info)
        } {
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                log::warn!("Queue Present reports ERROR_OUT_OF_DATE_KHR",);
                self.recreate_swap_chain();
                return Ok(false);
            }
            Err(e) => {
                return Err(anyhow!("Failed to present queue: {e}"));
            }
            Ok(_) => {}
        }

        // Get timestamps
        // TODO: Move to using vk::QueryResultFlags::WITH_AVAILABILITY copying values to a vk::Buffer or otherwise removing this wait to avoid stalls
        // https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html
        match unsafe {
            self.device.get_query_pool_results(
                self.query_pool_timestamps.get(),
                0,
                &mut self.timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        } {
            Err(e) => {
                log::error!("Failed to get timestamps: {}", e)
            }
            Ok(_) => {
                let timestamp_period = unsafe {
                    self.instance
                        .get_physical_device_properties(self.physical_device)
                        .limits
                        .timestamp_period
                };
                // Make sure that the values make some sense
                if self.timestamps[1] > self.timestamps[0] {
                    let delta_in_ms = (self.timestamps[1] - self.timestamps[0]) as f32
                        * timestamp_period
                        / 1_000_000.0;
                    log::trace!(
                        "Graphics pipeline took {:.2} ms [{:.2} fps]",
                        delta_in_ms,
                        1000.0 / delta_in_ms
                    );
                }
            }
        };

        match self.window.write() {
            Ok(v) => {
                // TODO: Find a way to avoid the window panicking at cleanup
                if let Err(e) = std::panic::catch_unwind(|| v.request_redraw()) {
                    log::error!("Request redraw panicked: {:?}", e);
                }
            }
            Err(e) => log::error!("Viewport failed to write to window: {e}"),
        }
        Ok(true)
    }
    fn recreate_swap_chain(&mut self) {
        self.swapchain_stuff
            .reset(
                &self.instance,
                &self.device,
                self.physical_device,
                &self.surface_stuff,
                &self.queue_family_indices,
            )
            .unwrap();
    }

    pub fn record_commands(&mut self, swapchain_index: usize, frame_index: usize) -> Result<()> {
        unsafe {
            self.device.reset_command_buffer(
                self.commands[frame_index],
                vk::CommandBufferResetFlags::empty(),
            )?
        };
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
                .begin_command_buffer(self.commands[frame_index], &begin_info)?;
            self.device.cmd_reset_query_pool(
                self.commands[frame_index],
                self.query_pool_timestamps.get(),
                0,
                self.timestamps.len() as u32,
            );
            self.device.cmd_write_timestamp(
                self.commands[frame_index],
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool_timestamps.get(),
                0,
            );
        }

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.05, 0.05, 0.05, 1.0],
            },
        };

        let colour_attachments = [vk::RenderingAttachmentInfo::default()
            .clear_value(clear_value)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL_KHR)
            .image_view(self.swapchain_stuff.image_views[swapchain_index].get())
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)];

        let rendering_info = vk::RenderingInfo::default()
            .color_attachments(&colour_attachments)
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_stuff.extent,
            });

        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.swapchain_stuff.images[swapchain_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        unsafe {
            self.device.cmd_pipeline_barrier(
                self.commands[frame_index],
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }

        unsafe {
            self.device
                .cmd_begin_rendering(self.commands[frame_index], &rendering_info);
            self.device.cmd_bind_pipeline(
                self.commands[frame_index],
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.get(),
            );

            let viewports = [vk::Viewport::default()
                .width(self.swapchain_stuff.extent.width as f32)
                .height(self.swapchain_stuff.extent.height as f32)
                .max_depth(1.0)];
            self.device
                .cmd_set_viewport(self.commands[frame_index], 0, &viewports);

            let scissors = [vk::Rect2D::default().extent(self.swapchain_stuff.extent)];
            self.device
                .cmd_set_scissor(self.commands[frame_index], 0, &scissors);

            let vertex_buffers = [self.vertex_buffer.get()];
            let offsets = [0];
            self.device.cmd_bind_vertex_buffers(
                self.commands[frame_index],
                0,
                &vertex_buffers,
                &offsets,
            );
            self.device.cmd_bind_index_buffer(
                self.commands[frame_index],
                self.index_buffer.get(),
                0,
                vk::IndexType::UINT16,
            );

            let descriptor_sets_to_bind = [self.descriptor_sets.unwrap()[frame_index]];

            self.device.cmd_bind_descriptor_sets(
                self.commands[frame_index],
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout.get(),
                0,
                &descriptor_sets_to_bind,
                &[],
            );

            self.device.cmd_draw_indexed(
                self.commands[frame_index],
                INDICES.len() as u32,
                1,
                0,
                0,
                0,
            );

            self.device.cmd_end_rendering(self.commands[frame_index]);

            let barrier = [vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.swapchain_stuff.images[swapchain_index])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())];

            self.device.cmd_pipeline_barrier(
                self.commands[frame_index],
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );

            self.device.cmd_write_timestamp(
                self.commands[frame_index],
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool_timestamps.get(),
                1,
            );
            self.device.end_command_buffer(self.commands[frame_index])?;
        };

        Ok(())
    }
}

impl<'a> Drop for Viewport<'a> {
    fn drop(&mut self) {
        // Wait until the gpu is finished
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                let _ = self.device.wait_for_fences(
                    &[self.in_flight_fences[i].get()],
                    true,
                    10_000_000,
                );
            }
            match self.device.queue_wait_idle(self.queue) {
                Err(e) => log::error!("Viewport failed to wait for its queue to finish: {e}"),
                _ => {}
            }
        }
    }
}

pub fn create_sync_object(
    device: &ash::Device,
) -> Result<(
    [Destructor<vk::Semaphore>; 2],
    [Destructor<vk::Semaphore>; 2],
    [Destructor<vk::Fence>; 2],
)> {
    // Semaphore is to tell the GPU to wait
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    // Fence is to tell the CPU to wait
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED); // Start the fence signalled so we can immediately wait on it

    unsafe {
        Ok((
            [
                Destructor::new(
                    device,
                    device.create_semaphore(&semaphore_info, None)?,
                    device.fp_v1_0().destroy_semaphore,
                ),
                Destructor::new(
                    device,
                    device.create_semaphore(&semaphore_info, None)?,
                    device.fp_v1_0().destroy_semaphore,
                ),
            ],
            [
                Destructor::new(
                    device,
                    device.create_semaphore(&semaphore_info, None)?,
                    device.fp_v1_0().destroy_semaphore,
                ),
                Destructor::new(
                    device,
                    device.create_semaphore(&semaphore_info, None)?,
                    device.fp_v1_0().destroy_semaphore,
                ),
            ],
            [
                Destructor::new(
                    device,
                    device.create_fence(&fence_info, None)?,
                    device.fp_v1_0().destroy_fence,
                ),
                Destructor::new(
                    device,
                    device.create_fence(&fence_info, None)?,
                    device.fp_v1_0().destroy_fence,
                ),
            ],
        ))
    }
}

pub fn create_pipeline(
    device: &ash::Device,
    swapchain_stuff: &SwapChainStuff,
    set_layout: vk::DescriptorSetLayout,
) -> Result<(Destructor<vk::PipelineLayout>, Destructor<vk::Pipeline>)> {
    let vert_shader_code = read_shader_code(Path::new("shaders/spv/viewport.vert.spv"))?;
    let frag_shader_code = read_shader_code(Path::new("shaders/spv/frag.slang.spv"))?;

    let vert_shader_module = core::create_shader_module(device, &vert_shader_code)?;
    let frag_shader_module = core::create_shader_module(device, &frag_shader_code)?;

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module.get())
            .name(&main_function_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module.get())
            .name(&main_function_name),
    ];

    // Viewport and Scissor
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let binding_description = [Vertex::get_binding_description()];
    let attribute_descriptions = Vertex::get_attribute_descriptions();

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_description)
        .vertex_attribute_descriptions(&attribute_descriptions);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // let viewports = [vk::Viewport::default()
    //     .width(swap_chain_stuff.swapchain_extent.width as f32)
    //     .height(swap_chain_stuff.swapchain_extent.height as f32)
    //     .max_depth(1.0)];

    // let scissors = [vk::Rect2D::default().extent(swap_chain_stuff.swapchain_extent)];

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        // .viewports(&viewports)
        .scissor_count(1);
    // .scissors(&scissors);

    // Rasterizer
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // Colour Blending
    let colour_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];

    let colour_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&colour_blend_attachments);

    // let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
    //     .depth_test_enable(true)
    //     .depth_write_enable(true)
    //     .depth_compare_op(vk::CompareOp::LESS);

    let set_layouts = [set_layout];
    // Pipeline Layout (Uniforms are declared here)
    let format = [swapchain_stuff.format];
    let mut pipeline_rendering_create_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&format);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

    let pipeline_layout = Destructor::new(
        device,
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? },
        device.fp_v1_0().destroy_pipeline_layout,
    );

    // Actual pipeline
    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&colour_blending)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout.get())
        // .depth_stencil_state(&depth_stencil)
        // .subpass(0)
        .push(&mut pipeline_rendering_create_info)];

    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
    };

    match pipeline {
        Ok(v) => Ok((
            pipeline_layout,
            Destructor::new(device, v[0], device.fp_v1_0().destroy_pipeline),
        )),
        Err(e) => Err(anyhow!("Failed to create graphics pipeline: {:?}", e)),
    }
}

pub fn create_descriptor_pool(device: &ash::Device) -> Result<Destructor<vk::DescriptorPool>> {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::STORAGE_IMAGE),
        vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
            .ty(vk::DescriptorType::UNIFORM_BUFFER),
    ];

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

    Ok(Destructor::new(
        device,
        unsafe { device.create_descriptor_pool(&pool_info, None)? },
        device.fp_v1_0().destroy_descriptor_pool,
    ))
}

pub fn create_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    uniforms_buffers: &[vk::Buffer; 2],
    images: &[vulkan::Image; 2],
) -> Result<[vk::DescriptorSet; 2]> {
    let layouts = [descriptor_set_layout; 2];

    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&allocate_info)? };

    // TODO: Target 4 descriptors (apparently this is the guaranteed supported amount and more can be slow)
    for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniforms_buffers[i])
            .offset(0)
            .range(std::mem::size_of::<UniformBufferObject>() as u64)];

        // Swap radiance image views around each frame
        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(images[i].view())];

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
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(&buffer_info),
        ];

        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    Ok([descriptor_sets[0], descriptor_sets[1]])
}

pub fn create_descriptor_set_layout(
    device: &ash::Device,
) -> Result<Destructor<vk::DescriptorSetLayout>> {
    let image_layout_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let ubo_layout_bindings = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = [image_layout_binding, ubo_layout_bindings];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    Ok(Destructor::new(
        device,
        unsafe { device.create_descriptor_set_layout(&layout_info, None) }?,
        device.fp_v1_0().destroy_descriptor_set_layout,
    ))
}
