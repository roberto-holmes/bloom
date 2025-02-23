mod bvh;
mod camera;
mod compute;
mod core;
mod debug;
mod material;
mod primitives;
mod select;
mod structures;
mod tools;
mod transfer;
mod vec;
mod vulkan;

use std::{
    sync::{mpsc, Arc, RwLock},
    thread::{self, JoinHandle},
    time::{Duration, SystemTime},
    u64,
};

use anyhow::Result;
use ash::{ext::debug_utils, vk, Entry};
use camera::Camera;
use core::*;
use debug::{setup_debug_utils, ValidationInfo};
use primitives::Scene;
use structures::{QueueFamilyIndices, SurfaceStuff, SwapChainStuff};
use vec::Vec3;
use vulkan::Destructor;
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{self, ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
const IDEAL_RADIANCE_IMAGE_SIZE_WIDTH: u32 = 5120;
const IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT: u32 = 2160;

pub const MAX_MATERIAL_COUNT: usize = 10;
pub const MAX_SPHERE_COUNT: usize = 20;
pub const MAX_QUAD_COUNT: usize = 20;
pub const MAX_TRIANGLE_COUNT: usize = 20;
pub const MAX_OBJECT_COUNT: usize = MAX_SPHERE_COUNT + MAX_QUAD_COUNT + MAX_TRIANGLE_COUNT;

const FOCAL_DISTANCE: f32 = 4.5;
const VFOV_DEG: f32 = 40.;
const DOF_SCALE: f32 = 0.05;

#[cfg(debug_assertions)]
const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};

#[cfg(not(debug_assertions))]
const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: false,
    required_validation_layers: [""],
};

fn main() -> Result<()> {
    env_logger::init();
    // log::error!("Testing Error");
    // log::warn!("Testing Warn");
    // log::info!("Testing Info");
    // log::debug!("Testing Debug");
    // log::trace!("Testing Trace");

    let mut app = App::new();
    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(event_loop::ControlFlow::Poll);

    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Debug, Default)]
struct MouseState {
    left_pressed: bool,
    right_pressed: bool,
    middle_pressed: bool,
    forward_pressed: bool,
    backward_pressed: bool,

    click_position: PhysicalPosition<f64>,

    last_position: PhysicalPosition<f64>,
    current_position: PhysicalPosition<f64>,
}

impl MouseState {
    pub fn update_button(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Left => {
                self.left_pressed = state.is_pressed();
                if state.is_pressed() {
                    self.click_position = self.current_position
                }
            }
            MouseButton::Right => self.right_pressed = state.is_pressed(),
            MouseButton::Middle => self.middle_pressed = state.is_pressed(),
            MouseButton::Forward => self.forward_pressed = state.is_pressed(),
            MouseButton::Back => self.backward_pressed = state.is_pressed(),
            MouseButton::Other(v) => {
                log::warn!("Ignoring mouse button {}", v)
            }
        }
    }
    pub fn update_position(&mut self, position: PhysicalPosition<f64>) {
        self.last_position = self.current_position;
        self.current_position = position;
    }
    pub fn get_pos_delta(&self) -> (f32, f32) {
        (
            (self.current_position.x - self.last_position.x) as f32,
            (self.current_position.y - self.last_position.y) as f32,
        )
    }
    pub fn get_click_delta(&self) -> f32 {
        // We just need an approximation to decide if the mouse has moved too much to ignore the click
        ((self.current_position.x - self.click_position.x).abs()
            + (self.current_position.y - self.click_position.y).abs()) as f32
    }
}

struct App {
    window: Option<Window>,
    vulkan: Option<VulkanApp>,
    last_frame_time: SystemTime,

    mouse_state: MouseState,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            vulkan: None,
            last_frame_time: SystemTime::now(),

            mouse_state: MouseState::default(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Bloom")
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
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_state.update_position(position);
                let (mut dx, mut dy) = self.mouse_state.get_pos_delta();
                dx *= -0.01;
                dy *= 0.01;

                if self.mouse_state.left_pressed {
                    self.vulkan.as_mut().unwrap().camera.orbit(dx, dy);
                    self.vulkan.as_mut().unwrap().uniform.reset_samples();
                } else if self.mouse_state.middle_pressed {
                    self.vulkan.as_mut().unwrap().camera.pan(dx, dy);
                    self.vulkan.as_mut().unwrap().uniform.reset_samples();
                } else if self.mouse_state.right_pressed {
                    self.vulkan.as_mut().unwrap().camera.zoom(-dy);
                    self.vulkan.as_mut().unwrap().uniform.reset_samples();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.mouse_state.update_button(button, state);

                if button == MouseButton::Left
                    && !state.is_pressed()
                    && self.mouse_state.get_click_delta() < 5.0
                {
                    let (hit_object, dist_to_object) = select::get_selected_object(
                        &self.mouse_state.current_position,
                        &self.vulkan.as_ref().unwrap().uniform,
                        &self.vulkan.as_ref().unwrap().scene.get_sphere_arr(),
                    );
                    if hit_object == usize::MAX {
                        self.vulkan.as_mut().unwrap().camera.uniforms.dof_scale = 0.;
                    } else {
                        self.vulkan.as_mut().unwrap().camera.uniforms.focal_distance =
                            dist_to_object;
                        self.vulkan.as_mut().unwrap().camera.uniforms.dof_scale = DOF_SCALE;
                    }
                    self.vulkan.as_mut().unwrap().uniform.reset_samples();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::PixelDelta(delta) => 0.001 * delta.y as f32,
                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                };
                self.vulkan.as_mut().unwrap().camera.zoom(delta);
                self.vulkan.as_mut().unwrap().uniform.reset_samples();
            }
            _ => (),
        }
    }
}

struct VulkanApp {
    should_compute_die: Arc<RwLock<bool>>,
    should_transfer_die: Arc<RwLock<bool>>,
    compute_thread: Option<JoinHandle<()>>,
    transfer_thread: Option<JoinHandle<()>>,
    transfer_sender: mpsc::Sender<u8>,
    graphic_receiver: mpsc::Receiver<u64>,
    graphic_transfer_semaphore: Destructor<vk::Semaphore>, // TODO: Should this object own the destructor

    graphics_queue: vk::Queue,
    _presentation_queue: vk::Queue,
    swap_chain_stuff: SwapChainStuff,
    swap_chain_image_views: Vec<vk::ImageView>,
    queue_family_indices: QueueFamilyIndices,

    query_pool_timestamps: vk::QueryPool,
    timestamps: Vec<u64>,

    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform: core::UniformBufferObject,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut core::UniformBufferObject>,

    material_buffers: Vec<vk::Buffer>,
    material_buffers_memory: Vec<vk::DeviceMemory>,
    material_buffers_mapped: Vec<*mut material::Material>,

    bvh_buffer: vk::Buffer,
    bvh_buffer_memory: vk::DeviceMemory,
    sphere_buffer: vk::Buffer,
    sphere_buffer_memory: vk::DeviceMemory,
    quad_buffer: vk::Buffer,
    quad_buffer_memory: vk::DeviceMemory,
    triangle_buffer: vk::Buffer,
    triangle_buffer_memory: vk::DeviceMemory,

    graphic_images: [vk::Image; 2],
    graphic_image_memories: [vk::DeviceMemory; 2],
    graphic_image_views: [vk::ImageView; 2],

    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    minimized: bool,
    resized: bool,

    camera: Camera,
    scene: Scene,

    current_frame_index: usize,
    frame_count: u128,

    device: vulkan::Device, // Logical Device
    physical_device: vk::PhysicalDevice,
    debug_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface_stuff: SurfaceStuff,
    instance: vulkan::Instance,
    _entry: Entry,
}

impl VulkanApp {
    pub fn new(window: &Window) -> Result<Self> {
        let mut scene = Scene::new();
        scene.add_material(material::Material::new_basic(Vec3::new(0.5, 0.5, 0.5), 0.));
        scene.add_sphere(primitives::Sphere::new(
            Vec3::new(0., -1000., -1.),
            1000.,
            1,
        ));

        {
            scene.add_sphere(primitives::Sphere::new(Vec3::new(2., 1., -2.), 1.0, 0));
            let mut current_material_index =
                scene.add_material(material::Material::new_clear(Vec3::new(1., 1., 1.)));
            scene.add_sphere(primitives::Sphere::new(
                Vec3::new(-2., 1., 0.),
                1.0,
                current_material_index,
            ));
            current_material_index = scene.add_material(material::Material::new(
                Vec3::new(0.9, 0.0, 0.3),
                0.,
                1.,
                0.67,
                1.,
                0.1,
                Vec3::new(0.9, 0.0, 0.3),
            ));
            scene.add_sphere(primitives::Sphere::new(
                Vec3::new(0., 3., 0.),
                0.5,
                current_material_index,
            ));
            current_material_index = scene.add_material(material::Material::new(
                Vec3::new(1.0, 1.0, 1.0),
                0.,
                1.,
                0.67,
                1.,
                1.,
                Vec3::new(1.0, 1.0, 1.0),
            ));
            scene.add_sphere(primitives::Sphere::new(
                Vec3::new(0., 3., -1.5),
                0.5,
                current_material_index,
            ));
            scene.add_quad(primitives::Quad::default());
            scene.add_triangle(primitives::Triangle::default());
        }

        let bvh = vec![bvh::create_bvh(&mut scene)];

        log::debug!("Initialising vulkan application");
        let entry = unsafe { Entry::load()? };
        // Set up Vulkan API
        let instance = core::create_instance(&entry, window)?;
        // debug::print_available_instance_extensions(&entry)?;
        // Set up callback for Vulkan debug messages
        let debug_messenger = setup_debug_utils(&entry, instance.get())?;

        let surface_stuff = SurfaceStuff::new(&entry, instance.get(), window)?;
        // Pick a graphics card
        let physical_device = pick_physical_device(instance.get(), &surface_stuff)?;
        let (device, queue_family_indices) =
            create_logical_device(instance.get(), &physical_device, &surface_stuff)?;

        let graphics_queue =
            core::create_queue(device.get(), queue_family_indices.graphics_family.unwrap());
        let presentation_queue =
            core::create_queue(device.get(), queue_family_indices.present_family.unwrap());

        let (query_pool_timestamps, timestamps) = prepare_timestamp_queries(device.get())?;

        let swap_chain_stuff = SwapChainStuff::new(
            instance.get(),
            &device,
            &physical_device,
            &surface_stuff,
            &queue_family_indices,
        );

        let swap_chain_image_views = create_image_views(
            device.get(),
            swap_chain_stuff.swapchain_format,
            &swap_chain_stuff.swapchain_images,
        )?;

        let set_layout = create_descriptor_set_layout(device.get())?;
        let descriptor_pool = create_descriptor_pool(device.get())?;

        let (pipeline_layout, pipelines) =
            create_graphics_pipeline(device.get(), &swap_chain_stuff, &set_layout)?;

        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
            create_uniform_buffer::<UniformBufferObject>(
                device.get(),
                instance.get(),
                &physical_device,
                1,
            )?;

        let (material_buffers, material_buffers_memory, material_buffers_mapped) =
            create_uniform_buffer::<material::Material>(
                device.get(),
                instance.get(),
                &physical_device,
                MAX_MATERIAL_COUNT as u64,
            )?;

        let (depth_image, depth_image_memory, depth_image_view) = create_depth_resources(
            device.get(),
            instance.get(),
            &physical_device,
            &swap_chain_stuff,
        )?;

        let command_pool = create_command_pool(
            device.get(),
            queue_family_indices.graphics_family.unwrap().0,
        )?;

        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(
            device.get(),
            instance.get(),
            &physical_device,
            &command_pool,
            &graphics_queue,
        )?;

        let (index_buffer, index_buffer_memory) = create_index_buffer(
            device.get(),
            instance.get(),
            &physical_device,
            &command_pool,
            &graphics_queue,
        )?;

        let (graphic_images, graphic_image_memories, graphic_image_views) =
            create_storage_image_pair(
                device.get(),
                instance.get(),
                &physical_device,
                &command_pool,
                &graphics_queue,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            )?;

        let (compute_images, compute_image_memories, compute_image_views) =
            create_storage_image_pair(
                device.get(),
                instance.get(),
                &physical_device,
                &command_pool,
                &graphics_queue,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            )?;

        let (bvh_buffer, bvh_buffer_memory) = create_storage_buffer(
            device.get(),
            instance.get(),
            &physical_device,
            &command_pool,
            &graphics_queue,
            &bvh,
        )?;

        let (sphere_buffer, sphere_buffer_memory) = create_storage_buffer(
            device.get(),
            instance.get(),
            &physical_device,
            &command_pool,
            &graphics_queue,
            &scene.get_sphere_arr().to_vec(),
        )?;

        let (quad_buffer, quad_buffer_memory) = create_storage_buffer(
            device.get(),
            instance.get(),
            &physical_device,
            &command_pool,
            &graphics_queue,
            &scene.get_quad_arr().to_vec(),
        )?;

        let (triangle_buffer, triangle_buffer_memory) = create_storage_buffer(
            device.get(),
            instance.get(),
            &physical_device,
            &command_pool,
            &graphics_queue,
            &scene.get_triangle_arr().to_vec(),
        )?;

        let descriptor_sets = create_descriptor_sets(
            device.get(),
            &descriptor_pool,
            &set_layout,
            &uniform_buffers,
            &material_buffers,
            bvh_buffer,
            sphere_buffer,
            quad_buffer,
            triangle_buffer,
            &graphic_image_views,
        )?;

        let command_buffers =
            create_command_buffers(device.get(), command_pool, MAX_FRAMES_IN_FLIGHT as u32)?;

        let should_transfer_die = Arc::new(RwLock::new(false));
        let should_compute_die = Arc::new(RwLock::new(false));
        let transfer_mutex = Arc::clone(&should_transfer_die);
        let compute_mutex = Arc::clone(&should_compute_die);

        let compute_device = device.get().clone();
        let transfer_device = device.get().clone();

        let (transfer_sender, transfer_receiver) = mpsc::channel();
        let (graphic_sender, graphic_receiver) = mpsc::channel();
        let (compute_sender, compute_receiver) = mpsc::channel();

        let compute_transfer_send = transfer_sender.clone();

        let transfer_semaphore = Destructor::new(
            device.get(),
            core::create_semaphore(device.get())?,
            device.get().fp_v1_0().destroy_semaphore,
        );

        let compute_transfer_semaphore = transfer_semaphore.get().clone();
        let transfer_transfer_semaphore = transfer_semaphore.get().clone();

        let compute_thread =
            match thread::Builder::new()
                .name("Compute".to_string())
                .spawn(move || {
                    compute::compute_thread(
                        compute_device,
                        queue_family_indices,
                        compute_images,
                        compute_image_views,
                        compute_image_memories,
                        compute_mutex,
                        compute_receiver,
                        compute_transfer_send,
                        compute_transfer_semaphore,
                    );
                }) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!("Failed to create compute thread: {e}");
                    None
                }
            };

        let transfer_thread =
            match thread::Builder::new()
                .name("Transfer".to_string())
                .spawn(move || {
                    transfer::transfer_thread(
                        transfer_device,
                        queue_family_indices,
                        transfer_transfer_semaphore,
                        &compute_images,
                        &graphic_images,
                        transfer_mutex,
                        transfer_receiver,
                        compute_sender,
                        graphic_sender,
                    );
                }) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!("Failed to create transfer thread: {e}");
                    None
                }
            };

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            create_sync_object(device.get())?;

        let camera = Camera::look_at(
            Vec3::new(3., 2., 3.),
            Vec3::new(0., 1., 0.),
            Vec3::new(0., 1., 0.),
            FOCAL_DISTANCE,
            VFOV_DEG,
            DOF_SCALE,
        );

        // log::debug!("{:#?}", bvh[0][0]);
        // log::debug!("{:#?}", scene.get_sphere_arr().to_vec());

        Ok(Self {
            uniform: core::UniformBufferObject::new().update(swap_chain_stuff.swapchain_extent),

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

            transfer_sender,
            graphic_receiver,
            should_compute_die,
            should_transfer_die,
            compute_thread,
            transfer_thread,
            graphic_transfer_semaphore: transfer_semaphore,

            query_pool_timestamps,
            timestamps,

            pipeline_layout,
            pipelines,
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

            material_buffers,
            material_buffers_memory,
            material_buffers_mapped,

            bvh_buffer,
            bvh_buffer_memory,
            sphere_buffer,
            sphere_buffer_memory,
            quad_buffer,
            quad_buffer_memory,
            triangle_buffer,
            triangle_buffer_memory,

            graphic_images,
            graphic_image_memories,
            graphic_image_views,

            depth_image,
            depth_image_memory,
            depth_image_view,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,

            camera,
            scene,

            minimized: false,
            resized: false,
            current_frame_index: 0,
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
        // Wait for the previous frame to finish
        unsafe {
            match self.device.get().wait_for_fences(
                &[self.in_flight_fences[self.current_frame_index]],
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
                1_000_000_000, // 1 second in nanoseconds
                self.image_available_semaphores[self.current_frame_index],
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
                .get()
                .reset_fences(&[self.in_flight_fences[self.current_frame_index]])
            {
                Err(e) => log::error!("Failed to reset in_flight_fence: {:?}", e),
                _ => {}
            }
        }

        match self.transfer_sender.send(self.current_frame_index as u8) {
            Err(e) => {
                log::error!("Failed to notify transfer channel: {e}")
            }
            Ok(()) => {}
        };

        let timestamp = match self.graphic_receiver.recv() {
            Ok(v) => v,
            Err(mpsc::RecvError) => {
                log::error!("Transfer channel is dead");
                return;
            }
        };

        match unsafe {
            self.device.get().reset_command_buffer(
                self.command_buffers[self.current_frame_index],
                vk::CommandBufferResetFlags::empty(),
            )
        } {
            Err(e) => {
                log::error!("Failed to reset command buffer: {:?}", e);
                return;
            }
            _ => {}
        }
        match self.record_command_buffer(image_index) {
            Err(e) => {
                log::error!("Failed to record command buffer: {:?}", e);
                return;
            }
            _ => {}
        }

        self.update_uniform_buffers(self.current_frame_index as u32, delta_time);

        let wait_semaphores = [
            self.image_available_semaphores[self.current_frame_index],
            self.graphic_transfer_semaphore.get(),
        ];
        let wait_stages = [
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
        ];
        let wait_values = [
            0, // Binary so will be ignored
            timestamp,
        ];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame_index]];
        let command_buffers = [self.command_buffers[self.current_frame_index]];

        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .push(&mut timeline_info);

        match unsafe {
            self.device.get().queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[self.current_frame_index],
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

        // Get timestamps
        // TODO: Move to using vk::QueryResultFlags::WITH_AVAILABILITY copying values to a vk::Buffer or otherwise removing this wait to avoid stalls
        // https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html
        match unsafe {
            self.device.get().get_query_pool_results(
                self.query_pool_timestamps,
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
                        .get()
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
                        "Pipeline took {:.2} ms [{:.2} fps]",
                        delta_in_ms,
                        1000.0 / delta_in_ms
                    );
                }
            }
        };

        if self.minimized || self.resized {
            log::debug!("[{}]\tThe window has been resized", self.frame_count);
            self.recreate_swap_chain();
        }
        self.frame_count += 1;
        self.current_frame_index += 1;
        self.current_frame_index %= MAX_FRAMES_IN_FLIGHT;
    }

    fn wait_idle(&self) {
        // unsafe {
        //     match self.device.get().queue_wait_idle(self.graphics_queue) {
        //         Err(e) => log::error!("Failed to wait for graphics queue to finish: {:?}", e),
        //         _ => {}
        //     }
        // };
        match unsafe { self.device.get().device_wait_idle() } {
            Err(e) => log::error!("Failed to wait for the device to return to idle: {}", e),
            _ => {}
        }
    }

    fn update_uniform_buffers(&mut self, current_image: u32, _delta_time: Duration) {
        self.uniform.tick();
        self.uniform.update(self.swap_chain_stuff.swapchain_extent);

        self.uniform.update_camera(&self.camera);

        let ubos = [self.uniform.clone()];
        let mats = self.scene.get_material_arr();

        unsafe {
            self.uniform_buffers_mapped[current_image as usize]
                .copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());
            self.material_buffers_mapped[current_image as usize]
                .copy_from_nonoverlapping(mats.as_ptr(), mats.len());
        };
    }

    fn recreate_swap_chain(&mut self) {
        if self.minimized == true {
            return;
        }
        log::debug!("Recreating swap chain");
        // self.wait_idle();
        // self.update_compute();
        unsafe {
            match self.device.get().queue_wait_idle(self.graphics_queue) {
                Err(e) => log::error!("Failed to wait for graphics queue to finish: {:?}", e),
                _ => {}
            }
        };
        log::debug!("Queue is ready for changes");

        unsafe { self.cleanup_swap_chain() };

        self.swap_chain_stuff = SwapChainStuff::new(
            self.instance.get(),
            &self.device,
            &self.physical_device,
            &self.surface_stuff,
            &self.queue_family_indices,
        );

        self.swap_chain_image_views = match create_image_views(
            self.device.get(),
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
            self.device.get(),
            self.instance.get(),
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

        //? Rebuild command buffers?

        self.uniform.reset_samples();
        self.resized = false;
        log::debug!("Swapchain recreated");
    }

    unsafe fn cleanup_swap_chain(&self) {
        self.device
            .get()
            .destroy_image_view(self.depth_image_view, None);
        self.device.get().destroy_image(self.depth_image, None);
        self.device.get().free_memory(self.depth_image_memory, None);

        for &image_view in self.swap_chain_image_views.iter() {
            self.device.get().destroy_image_view(image_view, None);
        }

        self.swap_chain_stuff
            .swapchain_loader
            .destroy_swapchain(self.swap_chain_stuff.swapchain, None);
    }

    pub fn record_command_buffer(&mut self, image_index: u32) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();

        let command_buffer = self.command_buffers[self.current_frame_index];
        let graphics_pipeline = self.pipelines[0];

        unsafe {
            self.device
                .get()
                .begin_command_buffer(command_buffer, &begin_info)?;
            self.device.get().cmd_reset_query_pool(
                command_buffer,
                self.query_pool_timestamps,
                0,
                self.timestamps.len() as u32,
            );
        }

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.05, 0.05, 0.05, 1.0],
            },
        };

        // let depth_clear_value = vk::ClearValue {
        //     depth_stencil: vk::ClearDepthStencilValue {
        //         depth: 1.0,
        //         stencil: 0,
        //     },
        // };

        let colour_attachments = [vk::RenderingAttachmentInfo::default()
            .clear_value(clear_value)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL_KHR)
            .image_view(self.swap_chain_image_views[image_index as usize]) // Can't remember if this is supposed to be image_index or current_frame_index
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)];

        // let depth_attachment = vk::RenderingAttachmentInfoKHR::default()
        //     .clear_value(depth_clear_value)
        //     .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL_KHR)
        //     .image_view(self.swap_chain_image_views[image_index as usize]) // Can't remember if this is supposed to be image_index or current_frame_index
        //     .load_op(vk::AttachmentLoadOp::CLEAR)
        //     .store_op(vk::AttachmentStoreOp::STORE);

        let rendering_info = vk::RenderingInfo::default()
            .color_attachments(&colour_attachments)
            // .depth_attachment(&depth_attachment)
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swap_chain_stuff.swapchain_extent,
            });

        transition_image_layout(
            self.device.get(),
            &self.command_pool,
            &self.graphics_queue,
            self.swap_chain_stuff.swapchain_images[image_index as usize],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;

        unsafe {
            self.device.get().cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool_timestamps,
                0,
            );

            self.device
                .get()
                .cmd_begin_rendering(command_buffer, &rendering_info);
            self.device.get().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );

            let viewports = [vk::Viewport::default()
                .width(self.swap_chain_stuff.swapchain_extent.width as f32)
                .height(self.swap_chain_stuff.swapchain_extent.height as f32)
                .max_depth(1.0)];
            self.device
                .get()
                .cmd_set_viewport(command_buffer, 0, &viewports);

            let scissors = [vk::Rect2D::default().extent(self.swap_chain_stuff.swapchain_extent)];
            self.device
                .get()
                .cmd_set_scissor(command_buffer, 0, &scissors);

            let vertex_buffers = [self.vertex_buffer];
            let offsets = [0];
            self.device
                .get()
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            self.device.get().cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );

            let descriptor_sets_to_bind = [self.descriptor_sets[self.current_frame_index]];

            self.device.get().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &descriptor_sets_to_bind,
                &[],
            );

            self.device
                .get()
                .cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);

            self.device.get().cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool_timestamps,
                1,
            );

            self.device.get().cmd_end_rendering(command_buffer);

            transition_image_layout(
                self.device.get(),
                &self.command_pool,
                &self.graphics_queue,
                self.swap_chain_stuff.swapchain_images[image_index as usize],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            )?;

            self.device.get().end_command_buffer(command_buffer)?;
        };
        Ok(())
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        log::debug!("Cleaning up");
        // TODO: Replace with a deletion queue
        unsafe {
            // Kill threads
            match self.should_transfer_die.write() {
                Ok(mut m) => *m = true,
                Err(e) => log::error!("Failed to write to transfer death mutex: {e}"),
            }
            if let Some(t) = self.transfer_thread.take() {
                log::debug!("Wating for transfer to finish");
                t.join().unwrap();
            }
            match self.should_compute_die.write() {
                Ok(mut m) => *m = true,
                Err(e) => log::error!("Failed to write to transfer death mutex: {e}"),
            }
            if let Some(t) = self.compute_thread.take() {
                log::debug!("Wating for compute to finish");
                t.join().unwrap();
            }

            self.wait_idle();
            log::debug!("Device now idle");

            self.cleanup_swap_chain();

            for image_view in self.graphic_image_views {
                self.device.get().destroy_image_view(image_view, None);
            }
            for image in self.graphic_images {
                self.device.get().destroy_image(image, None);
            }
            for image_memory in self.graphic_image_memories {
                self.device.get().free_memory(image_memory, None);
            }

            self.device.get().destroy_buffer(self.bvh_buffer, None);
            self.device.get().destroy_buffer(self.sphere_buffer, None);
            self.device.get().destroy_buffer(self.quad_buffer, None);
            self.device.get().destroy_buffer(self.triangle_buffer, None);

            self.device
                .get()
                .free_memory(self.sphere_buffer_memory, None);
            self.device.get().free_memory(self.bvh_buffer_memory, None);
            self.device.get().free_memory(self.quad_buffer_memory, None);
            self.device
                .get()
                .free_memory(self.triangle_buffer_memory, None);

            for &buffer in self.material_buffers.iter() {
                self.device.get().destroy_buffer(buffer, None);
            }
            for &memory in self.material_buffers_memory.iter() {
                self.device.get().free_memory(memory, None);
            }

            for &buffer in self.uniform_buffers.iter() {
                self.device.get().destroy_buffer(buffer, None);
            }
            for &memory in self.uniform_buffers_memory.iter() {
                self.device.get().free_memory(memory, None);
            }

            // self.device.get().destroy_pipeline(self.compute_pipeline, None);
            // self.device
            //     .destroy_pipeline_layout(self.compute_pipeline_layout, None);
            // self.device
            //     .destroy_descriptor_set_layout(self.compute_descriptor_set_layout, None);

            self.device
                .get()
                .destroy_descriptor_pool(self.descriptor_pool, None);

            // self.device
            //     .destroy_descriptor_pool(self.compute_descriptor_pool, None);

            self.device
                .get()
                .destroy_descriptor_set_layout(self.set_layout, None);

            self.device.get().destroy_buffer(self.index_buffer, None);
            self.device
                .get()
                .free_memory(self.index_buffer_memory, None);

            self.device.get().destroy_buffer(self.vertex_buffer, None);
            self.device
                .get()
                .free_memory(self.vertex_buffer_memory, None);

            for &pipeline in self.pipelines.iter() {
                self.device.get().destroy_pipeline(pipeline, None);
            }
            self.device
                .get()
                .destroy_pipeline_layout(self.pipeline_layout, None);

            for &semaphore in self.image_available_semaphores.iter() {
                self.device.get().destroy_semaphore(semaphore, None);
            }
            for &semaphore in self.render_finished_semaphores.iter() {
                self.device.get().destroy_semaphore(semaphore, None);
            }
            for &fence in self.in_flight_fences.iter() {
                self.device.get().destroy_fence(fence, None);
            }

            // self.device.get().destroy_semaphore(self.compute_semaphore, None);

            self.device
                .get()
                .destroy_query_pool(self.query_pool_timestamps, None);

            // self.device
            //     .destroy_command_pool(self.compute_command_pool, None);
            self.device
                .get()
                .destroy_command_pool(self.command_pool, None);

            // self.device.get().destroy_device(None);

            if VALIDATION.is_enable {
                if let Some((report, messenger)) = self.debug_messenger.take() {
                    report.destroy_debug_utils_messenger(messenger, None);
                }
            }
            self.surface_stuff
                .surface_loader
                .destroy_surface(self.surface_stuff.surface, None);

            // self.instance.destroy_instance(None);

            log::info!("VulkanApp has been dropped");
        }
    }
}
