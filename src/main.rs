mod debug;
mod structures;
mod tools;
mod utils;
mod vec;

use std::{
    time::{Duration, SystemTime},
    u64,
};

use anyhow::Result;
use ash::{ext::debug_utils, vk, Entry, Instance};
use cgmath::{Deg, Matrix4, Point3, SquareMatrix, Vector3};
use debug::{setup_debug_utils, ValidationInfo};
use structures::{DeviceExtension, QueueFamilyIndices, SurfaceStuff, SwapChainStuff};
use utils::*;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{self, ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const MAX_FRAMES_IN_FLIGHT: usize = 2;

const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};
const DEVICE_EXTENSIONS: DeviceExtension = DeviceExtension {
    names: ["VK_KHR_swapchain"],
};

fn main() -> Result<()> {
    env_logger::init();
    println!("Hello, world!");
    log::error!("Error");
    log::warn!("Warn");
    log::info!("Info");
    log::debug!("Debug");
    log::trace!("Trace");

    let mut app = App::new();
    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(event_loop::ControlFlow::Poll);

    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    window: Option<Window>,
    vulkan: Option<VulkanApp>,
    last_frame_time: SystemTime,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            vulkan: None,
            last_frame_time: SystemTime::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Vulkan Tutorial")
                    .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
            )
            .unwrap();
        self.vulkan = Some(VulkanApp::new(&window).expect("Failed to create vulkan app"));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                log::info!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                // log::info!(
                //     "Requested window resize to {}x{}",
                //     physical_size.width,
                //     physical_size.height
                // );
                self.vulkan.as_mut().unwrap().resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
                let delta_time = match self.last_frame_time.elapsed() {
                    Ok(elapsed) => elapsed,
                    Err(e) => {
                        log::error!("Failed to get time since last frame: {e:?}");
                        Duration::default()
                    }
                };
                self.last_frame_time = SystemTime::now();
                self.vulkan.as_mut().unwrap().draw_frame(delta_time);
            }
            _ => (),
        }
    }
}

struct VulkanApp {
    _entry: Entry,
    instance: Instance,
    debug_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device, // Logical Device
    graphics_queue: vk::Queue,
    _presentation_queue: vk::Queue,
    surface_stuff: SurfaceStuff,
    swap_chain_stuff: SwapChainStuff,
    swap_chain_image_views: Vec<vk::ImageView>,
    queue_family_indices: QueueFamilyIndices,

    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipelines: Vec<vk::Pipeline>,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_transform: utils::UniformBufferObject,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut utils::UniformBufferObject>,

    image_texture: vk::Image,
    image_texture_memory: vk::DeviceMemory,
    image_texture_view: vk::ImageView,
    texture_sampler: vk::Sampler,

    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    minimized: bool,
    resized: bool,

    current_frame: usize,
    frame_count: u128,
}

impl VulkanApp {
    pub fn new(window: &Window) -> Result<Self> {
        log::debug!("Initialising vulkan application");
        let entry = unsafe { Entry::load()? };
        // Set up Vulkan API
        let instance = utils::create_instance(&entry, window)?;
        debug::print_available_instance_extensions(&entry)?;
        // Set up callback for Vulkan debug messages
        let debug_messenger = setup_debug_utils(&entry, &instance)?;

        let surface_stuff = SurfaceStuff::new(&entry, &instance, window)?;
        // Pick a graphics card
        let physical_device = pick_physical_device(&instance, &surface_stuff)?;
        let (device, queue_family_indices) =
            create_logical_device(&instance, &physical_device, &surface_stuff)?;
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0) };
        let presentation_queue =
            unsafe { device.get_device_queue(queue_family_indices.present_family.unwrap(), 0) };

        let swap_chain_stuff = SwapChainStuff::new(
            &instance,
            &device,
            &physical_device,
            &surface_stuff,
            &queue_family_indices,
        );

        let swap_chain_image_views = create_image_views(
            &device,
            swap_chain_stuff.swapchain_format,
            &swap_chain_stuff.swapchain_images,
        )?;

        let render_pass =
            create_render_pass(&device, &instance, &physical_device, &swap_chain_stuff)?;

        let set_layout = create_descriptor_set_layout(&device)?;
        let descriptor_pool = create_descriptor_pool(&device)?;

        let (pipeline_layout, pipelines) =
            create_graphics_pipeline(&device, &swap_chain_stuff, &render_pass, &set_layout)?;

        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
            create_uniform_buffer(&device, &instance, &physical_device)?;

        let (depth_image, depth_image_memory, depth_image_view) =
            create_depth_resources(&device, &instance, &physical_device, &swap_chain_stuff)?;

        let framebuffers = create_framebuffers(
            &device,
            &swap_chain_image_views,
            &depth_image_view,
            &render_pass,
            &swap_chain_stuff,
        )?;

        let command_pool = create_command_pool(&device, &queue_family_indices)?;

        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
        )?;

        let (index_buffer, index_buffer_memory) = create_index_buffer(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
        )?;

        let (image_texture, image_texture_memory) = create_texture_image(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
        )?;

        let image_texture_view = create_texture_image_view(&device, &image_texture)?;

        let texture_sampler = create_texture_sampler(&device, &instance, &physical_device)?;

        let descriptor_sets = create_descriptor_sets(
            &device,
            &descriptor_pool,
            &set_layout,
            &uniform_buffers,
            &texture_sampler,
            &image_texture_view,
        )?;

        let command_buffers = create_command_buffers(&device, &command_pool)?;
        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            create_sync_object(&device)?;

        Ok(Self {
            uniform_transform: utils::UniformBufferObject {
                model: Matrix4::<f32>::identity(),
                view: Matrix4::look_at_rh(
                    Point3::new(2.0, 2.0, 2.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 0.0, 1.0),
                ),
                proj: cgmath::perspective(
                    Deg(45.0),
                    swap_chain_stuff.swapchain_extent.width as f32
                        / swap_chain_stuff.swapchain_extent.height as f32,
                    0.1,
                    10.0,
                ),
            },

            _entry: entry,
            instance,
            debug_messenger,
            physical_device,
            device,
            graphics_queue,
            _presentation_queue: presentation_queue,
            surface_stuff,
            swap_chain_stuff,
            swap_chain_image_views,
            queue_family_indices,

            pipeline_layout,
            render_pass,
            pipelines,
            framebuffers,
            command_pool,
            command_buffers,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,

            set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,

            image_texture,
            image_texture_memory,
            image_texture_view,
            texture_sampler,

            depth_image,
            depth_image_memory,
            depth_image_view,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,

            minimized: false,
            resized: false,
            current_frame: 0,
            frame_count: 0,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            log::debug!("Window is minimized");
            self.minimized = true;
        } else {
            self.minimized = false;
            self.resized = true;
        }
    }

    pub fn draw_frame(&mut self, delta_time: Duration) {
        unsafe {
            match self.device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]],
                true,
                u64::MAX,
            ) {
                Err(e) => log::error!("Failed to wait on in_flight_fence: {:?}", e),
                _ => {}
            }
        }
        let image_index = unsafe {
            match self.swap_chain_stuff.swapchain_loader.acquire_next_image(
                self.swap_chain_stuff.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            ) {
                Err(e) => {
                    match e {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => {
                            log::debug!(
                                "[{}]\tSwapchain reports ERROR_OUT_OF_DATE_KHR",
                                self.frame_count
                            );
                            self.recreate_swap_chain();
                            self.frame_count += 1;
                        }
                        _ => log::warn!("Failed to acquire next image from the swap chain: {}", e),
                    }
                    return;
                }
                Ok((v, _is_suboptimal)) => v,
            }
        };
        unsafe {
            match self
                .device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])
            {
                Err(e) => log::error!("Failed to reset in_flight_fence: {:?}", e),
                _ => {}
            }
        }

        match unsafe {
            self.device.reset_command_buffer(
                self.command_buffers[self.current_frame],
                vk::CommandBufferResetFlags::empty(),
            )
        } {
            Err(e) => {
                log::error!("Failed to reset command buffer: {:?}", e);
                return;
            }
            _ => {}
        }
        match record_command_buffer(
            &self.device,
            &self.render_pass,
            &self.swap_chain_stuff,
            &self.framebuffers,
            &self.pipelines[0],
            &self.pipeline_layout,
            &self.command_buffers[self.current_frame],
            &self.vertex_buffer,
            &self.index_buffer,
            &self.descriptor_sets,
            image_index,
            self.current_frame,
        ) {
            Err(e) => {
                log::error!("Failed to record command buffer: {:?}", e);
                return;
            }
            _ => {}
        }

        self.update_uniform_buffer(self.current_frame as u32, delta_time);

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let command_buffers = [self.command_buffers[self.current_frame]];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        match unsafe {
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[self.current_frame],
            )
        } {
            Err(e) => {
                log::error!("Failed to submit queue: {}", e);
                return;
            }
            _ => {}
        }
        let swapchains = [self.swap_chain_stuff.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match unsafe {
            self.swap_chain_stuff
                .swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
        } {
            Err(e) => match e {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                    log::debug!(
                        "[{}]\tQueue reports ERROR_OUT_OF_DATE_KHR | SUBOPTIMAL_KHR",
                        self.frame_count
                    );
                    self.recreate_swap_chain();
                    self.frame_count += 1;
                    return;
                }
                _ => log::error!("Failed to present queue: {}", e),
            },
            _ => {}
        }
        if self.minimized || self.resized {
            log::debug!("[{}]\tThe frame buffer has been resized", self.frame_count);
            self.recreate_swap_chain();
        }
        self.frame_count += 1;
        self.current_frame += 1;
        self.current_frame %= MAX_FRAMES_IN_FLIGHT;
    }

    fn wait_idle(&self) {
        match unsafe { self.device.device_wait_idle() } {
            Err(e) => log::error!("Failed to wait for the device to return to idle: {}", e),
            _ => {}
        }
    }

    fn update_uniform_buffer(&mut self, current_image: u32, delta_time: Duration) {
        self.uniform_transform.model = Matrix4::from_axis_angle(
            Vector3::new(0.0, 0.0, 1.0),
            // Deg(90.0) * delta_time.as_millis() as f32,
            Deg(90.0) * delta_time.as_secs_f32(),
        ) * self.uniform_transform.model;

        let ubos = [self.uniform_transform.clone()];

        unsafe {
            self.uniform_buffers_mapped[current_image as usize]
                .copy_from_nonoverlapping(ubos.as_ptr(), ubos.len())
        };
    }

    fn recreate_swap_chain(&mut self) {
        if self.minimized == true {
            return;
        }
        // log::debug!("Recreating swap chain");
        self.wait_idle();

        unsafe { self.cleanup_swap_chain() };

        self.swap_chain_stuff = SwapChainStuff::new(
            &self.instance,
            &self.device,
            &self.physical_device,
            &self.surface_stuff,
            &self.queue_family_indices,
        );

        self.swap_chain_image_views = match create_image_views(
            &self.device,
            self.swap_chain_stuff.swapchain_format,
            &self.swap_chain_stuff.swapchain_images,
        ) {
            Err(e) => {
                log::error!(
                    "Failed to create image view when recreating swap chain: {}",
                    e
                );
                return;
            }
            Ok(v) => v,
        };

        (
            self.depth_image,
            self.depth_image_memory,
            self.depth_image_view,
        ) = match create_depth_resources(
            &self.device,
            &self.instance,
            &self.physical_device,
            &self.swap_chain_stuff,
        ) {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "Failed to create depth buffer when recreating swap chain: {}",
                    e
                );
                return;
            }
        };

        self.render_pass = match create_render_pass(
            &self.device,
            &self.instance,
            &self.physical_device,
            &self.swap_chain_stuff,
        ) {
            Err(e) => {
                log::error!(
                    "Failed to create render pass when recreating swap chain: {}",
                    e
                );
                return;
            }
            Ok(v) => v,
        };

        self.framebuffers = match create_framebuffers(
            &self.device,
            &self.swap_chain_image_views,
            &self.depth_image_view,
            &self.render_pass,
            &self.swap_chain_stuff,
        ) {
            Ok(v) => v,
            Err(e) => {
                log::error!(
                    "Failed to create framebuffer when recreating swap chain: {}",
                    e
                );
                return;
            }
        };
        //? Rebuild command buffers?

        self.resized = false;
    }

    unsafe fn cleanup_swap_chain(&self) {
        self.device.destroy_image_view(self.depth_image_view, None);
        self.device.destroy_image(self.depth_image, None);
        self.device.free_memory(self.depth_image_memory, None);

        for &framebuffer in self.framebuffers.iter() {
            self.device.destroy_framebuffer(framebuffer, None);
        }
        self.device.destroy_render_pass(self.render_pass, None);

        for &image_view in self.swap_chain_image_views.iter() {
            self.device.destroy_image_view(image_view, None);
        }

        self.swap_chain_stuff
            .swapchain_loader
            .destroy_swapchain(self.swap_chain_stuff.swapchain, None);
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        log::debug!("Cleaning up");
        unsafe {
            self.wait_idle();

            self.cleanup_swap_chain();

            self.device.destroy_sampler(self.texture_sampler, None);
            self.device
                .destroy_image_view(self.image_texture_view, None);
            self.device.destroy_image(self.image_texture, None);
            self.device.free_memory(self.image_texture_memory, None);

            for &buffer in self.uniform_buffers.iter() {
                self.device.destroy_buffer(buffer, None);
            }
            for &memory in self.uniform_buffers_memory.iter() {
                self.device.free_memory(memory, None);
            }

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            for &pipeline in self.pipelines.iter() {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            for &semaphore in self.image_available_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in self.render_finished_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &fence in self.in_flight_fences.iter() {
                self.device.destroy_fence(fence, None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);

            if VALIDATION.is_enable {
                if let Some((report, messenger)) = self.debug_messenger.take() {
                    report.destroy_debug_utils_messenger(messenger, None);
                }
            }
            self.surface_stuff
                .surface_loader
                .destroy_surface(self.surface_stuff.surface, None);

            self.instance.destroy_instance(None);
        }
    }
}
