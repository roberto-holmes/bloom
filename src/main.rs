mod bvh;
mod camera;
mod debug;
mod material;
mod primitives;
mod select;
mod structures;
mod tools;
mod vec;
mod vulkan;

use std::{
    time::{Duration, SystemTime},
    u64,
};

use anyhow::Result;
use ash::{ext::debug_utils, vk, Entry, Instance};
use camera::Camera;
use debug::{setup_debug_utils, ValidationInfo};
use primitives::Scene;
use structures::{QueueFamilyIndices, SurfaceStuff, SwapChainStuff};
use vec::Vec3;
use vulkan::*;
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

    compute_queue: vk::Queue,
    compute_descriptor_set_layout: vk::DescriptorSetLayout,
    compute_pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,
    compute_command_pool: vk::CommandPool,
    compute_commands: [vk::CommandBuffer; 2],
    compute_copy_commands: [vk::CommandBuffer; 4],
    compute_fence: vk::Fence,
    compute_copy_semaphores: [vk::Semaphore; 1],
    compute_descriptor_pool: vk::DescriptorPool,

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
    uniform: vulkan::UniformBufferObject,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut vulkan::UniformBufferObject>,

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

    radiance_images: [vk::Image; 4],
    radiance_image_memories: [vk::DeviceMemory; 4],
    radiance_image_views: [vk::ImageView; 4],

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

    compute_frame: usize,
    current_frame: usize,
    frame_count: u128,
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
        let instance = vulkan::create_instance(&entry, window)?;
        // debug::print_available_instance_extensions(&entry)?;
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
        let compute_queue =
            unsafe { device.get_device_queue(queue_family_indices.compute_family.unwrap(), 0) }; // TODO:  Get a different queue to the other ones (for async)

        let (query_pool_timestamps, timestamps) = prepare_timestamp_queries(&device)?;

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

        let set_layout = create_descriptor_set_layout(&device)?;
        let descriptor_pool = create_descriptor_pool(&device)?;

        let (pipeline_layout, pipelines) =
            create_graphics_pipeline(&device, &swap_chain_stuff, &set_layout)?;

        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
            create_uniform_buffer::<UniformBufferObject>(&device, &instance, &physical_device, 1)?;

        let (material_buffers, material_buffers_memory, material_buffers_mapped) =
            create_uniform_buffer::<material::Material>(
                &device,
                &instance,
                &physical_device,
                MAX_MATERIAL_COUNT as u64,
            )?;

        let (depth_image, depth_image_memory, depth_image_view) =
            create_depth_resources(&device, &instance, &physical_device, &swap_chain_stuff)?;

        let command_pool =
            create_command_pool(&device, queue_family_indices.graphics_family.unwrap())?;

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

        let (radiance_images, radiance_image_memories, radiance_image_views) =
            create_storage_images(
                &device,
                &instance,
                &physical_device,
                &command_pool,
                &graphics_queue,
            )?;

        let (bvh_buffer, bvh_buffer_memory) = create_storage_buffer(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
            &bvh,
        )?;

        let (sphere_buffer, sphere_buffer_memory) = create_storage_buffer(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
            &scene.get_sphere_arr().to_vec(),
        )?;

        let (quad_buffer, quad_buffer_memory) = create_storage_buffer(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
            &scene.get_quad_arr().to_vec(),
        )?;

        let (triangle_buffer, triangle_buffer_memory) = create_storage_buffer(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
            &scene.get_triangle_arr().to_vec(),
        )?;

        let descriptor_sets = create_descriptor_sets(
            &device,
            &descriptor_pool,
            &set_layout,
            &uniform_buffers,
            &material_buffers,
            bvh_buffer,
            sphere_buffer,
            quad_buffer,
            triangle_buffer,
            &radiance_image_views[2..4],
        )?;

        let command_buffers =
            create_command_buffers(&device, &command_pool, MAX_FRAMES_IN_FLIGHT as u32)?;

        let (
            compute_descriptor_pool,
            compute_descriptor_set_layout,
            compute_descriptor_sets,
            compute_pipeline_layout,
            compute_pipeline,
        ) = create_compute_pipeline(&device, &radiance_image_views[0..2])?;

        let compute_command_pool =
            create_command_pool(&device, queue_family_indices.compute_family.unwrap())?;

        let compute_command_buffers = create_command_buffers(&device, &compute_command_pool, 2)?;

        let compute_copy_command_buffers =
            create_command_buffers(&device, &compute_command_pool, 4)?;

        let compute_commands = record_compute_commands(
            &device,
            &compute_command_buffers,
            &compute_pipeline,
            &compute_pipeline_layout,
            &compute_descriptor_sets,
        )?;
        let compute_copy_commands =
            record_compute_copy_commands(&device, &compute_copy_command_buffers, &radiance_images)?;

        let fence_info = vk::FenceCreateInfo::default(); // Start the fence signalled so we can immediately wait on it
        let compute_fence = unsafe { device.create_fence(&fence_info, None)? };
        // let mut semaphore_type_info =
        //     vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        // let semaphore_info = vk::SemaphoreCreateInfo::default().push(&mut semaphore_type_info);
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let compute_copy_semaphores = unsafe { [device.create_semaphore(&semaphore_info, None)?] };

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            create_sync_object(&device)?;

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
            uniform: vulkan::UniformBufferObject::new().update(swap_chain_stuff.swapchain_extent),

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

            compute_queue,
            compute_descriptor_set_layout,
            compute_pipeline_layout,
            compute_pipeline,
            compute_command_pool,
            compute_frame: 0,
            compute_commands,
            compute_copy_commands,
            compute_fence,
            compute_copy_semaphores,
            compute_descriptor_pool,

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

            radiance_images,
            radiance_image_memories,
            radiance_image_views,

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

        self.update_compute();

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
        match self.record_command_buffer(image_index) {
            Err(e) => {
                log::error!("Failed to record command buffer: {:?}", e);
                return;
            }
            _ => {}
        }

        self.update_uniform_buffers(self.current_frame as u32, delta_time);

        let wait_semaphores = [
            self.image_available_semaphores[self.current_frame],
            self.compute_copy_semaphores[0],
        ];
        let wait_stages = [
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ];
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

        // Get timestamps
        // TODO: Move to using vk::QueryResultFlags::WITH_AVAILABILITY copying values to a vk::Buffer or otherwise removing this wait to avoid stalls
        // https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html
        match unsafe {
            self.device.get_query_pool_results(
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
        self.current_frame += 1;
        self.current_frame %= MAX_FRAMES_IN_FLIGHT;
    }

    fn wait_idle(&self) {
        // unsafe {
        //     match self.device.queue_wait_idle(self.graphics_queue) {
        //         Err(e) => log::error!("Failed to wait for graphics queue to finish: {:?}", e),
        //         _ => {}
        //     }
        // };
        match unsafe { self.device.device_wait_idle() } {
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
            match self.device.queue_wait_idle(self.graphics_queue) {
                Err(e) => log::error!("Failed to wait for graphics queue to finish: {:?}", e),
                _ => {}
            }
        };
        log::debug!("Queue is ready for changes");

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

        //? Rebuild command buffers?

        self.uniform.reset_samples();
        self.resized = false;
        log::debug!("Swapchain recreated");
    }

    unsafe fn cleanup_swap_chain(&self) {
        self.device.destroy_image_view(self.depth_image_view, None);
        self.device.destroy_image(self.depth_image, None);
        self.device.free_memory(self.depth_image_memory, None);

        for &image_view in self.swap_chain_image_views.iter() {
            self.device.destroy_image_view(image_view, None);
        }

        self.swap_chain_stuff
            .swapchain_loader
            .destroy_swapchain(self.swap_chain_stuff.swapchain, None);
    }

    pub fn record_command_buffer(&mut self, image_index: u32) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();

        let command_buffer = self.command_buffers[self.current_frame];
        let graphics_pipeline = self.pipelines[0];

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
            self.device.cmd_reset_query_pool(
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
            .image_view(self.swap_chain_image_views[image_index as usize]) // Can't remember if this is supposed to be image_index or current_frame
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)];

        // let depth_attachment = vk::RenderingAttachmentInfoKHR::default()
        //     .clear_value(depth_clear_value)
        //     .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL_KHR)
        //     .image_view(self.swap_chain_image_views[image_index as usize]) // Can't remember if this is supposed to be image_index or current_frame
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
            &self.device,
            &self.command_pool,
            &self.graphics_queue,
            self.swap_chain_stuff.swapchain_images[image_index as usize],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;

        unsafe {
            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool_timestamps,
                0,
            );

            self.device
                .cmd_begin_rendering(command_buffer, &rendering_info);
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );

            let viewports = [vk::Viewport::default()
                .width(self.swap_chain_stuff.swapchain_extent.width as f32)
                .height(self.swap_chain_stuff.swapchain_extent.height as f32)
                .max_depth(1.0)];
            self.device.cmd_set_viewport(command_buffer, 0, &viewports);

            let scissors = [vk::Rect2D::default().extent(self.swap_chain_stuff.swapchain_extent)];
            self.device.cmd_set_scissor(command_buffer, 0, &scissors);

            let vertex_buffers = [self.vertex_buffer];
            let offsets = [0];
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );

            let descriptor_sets_to_bind = [self.descriptor_sets[self.current_frame]];

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &descriptor_sets_to_bind,
                &[],
            );

            self.device
                .cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);

            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool_timestamps,
                1,
            );

            self.device.cmd_end_rendering(command_buffer);

            transition_image_layout(
                &self.device,
                &self.command_pool,
                &self.graphics_queue,
                self.swap_chain_stuff.swapchain_images[image_index as usize],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            )?;

            self.device.end_command_buffer(command_buffer)?;
        };
        Ok(())
    }

    fn update_compute(&mut self) {
        // Perform a render
        self.perform_compute();

        // Wait for the render to complete
        self.wait_for_compute_operation();

        self.compute_frame += 1;
        self.compute_frame %= MAX_FRAMES_IN_FLIGHT;

        // Perform a copy
        self.perform_compute_copy();

        // Wait for the copy to complete
        self.wait_for_compute_operation();
    }
    fn wait_for_compute_operation(&mut self) {
        unsafe {
            match self
                .device
                .wait_for_fences(&[self.compute_fence], true, u64::MAX)
            {
                Err(e) => log::error!("Failed to wait on compute fence: {:?}", e),
                _ => {}
            }
        }
        unsafe {
            match self.device.reset_fences(&[self.compute_fence]) {
                Err(e) => log::error!("Failed to reset compute_fence: {:?}", e),
                _ => {}
            }
        }
        unsafe {
            match self.device.queue_wait_idle(self.compute_queue) {
                Err(e) => log::error!("Failed to wait for compute queue to finish: {:?}", e),
                _ => {}
            }
        };
    }
    fn perform_compute(&mut self) {
        let command_buffer = [self.compute_commands[self.compute_frame]];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffer)];

        match unsafe {
            self.device
                .queue_submit(self.compute_queue, &submit_info, self.compute_fence)
        } {
            Err(e) => {
                log::error!("Failed to submit compute commands: {}", e)
            }
            _ => {}
        }
    }

    fn perform_compute_copy(&mut self) {
        let command_index = 1 << self.current_frame | self.compute_frame;
        let command_buffer = [self.compute_copy_commands[command_index]];

        let submit_info = [vk::SubmitInfo::default()
            .command_buffers(&command_buffer)
            .signal_semaphores(&self.compute_copy_semaphores)];

        match unsafe {
            self.device
                .queue_submit(self.compute_queue, &submit_info, self.compute_fence)
        } {
            Err(e) => {
                log::error!("Failed to submit compute commands: {}", e)
            }
            _ => {}
        }
    }

    fn signal_compute(&self) {
        let signal_info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.compute_copy_semaphores[0])
            .value(1);
        match unsafe { self.device.signal_semaphore(&signal_info) } {
            Err(e) => {
                log::error!("Failed to signal compute copy semaphore: {}", e)
            }
            _ => {}
        };
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        log::debug!("Cleaning up");
        unsafe {
            self.update_compute();
            // self.signal_compute();
            log::debug!("Performed one last compute operation");
            self.wait_idle();
            log::debug!("Device now idle");

            self.cleanup_swap_chain();

            for image_view in self.radiance_image_views {
                self.device.destroy_image_view(image_view, None);
            }
            for image in self.radiance_images {
                self.device.destroy_image(image, None);
            }
            for image_memory in self.radiance_image_memories {
                self.device.free_memory(image_memory, None);
            }

            self.device.destroy_buffer(self.bvh_buffer, None);
            self.device.destroy_buffer(self.sphere_buffer, None);
            self.device.destroy_buffer(self.quad_buffer, None);
            self.device.destroy_buffer(self.triangle_buffer, None);

            self.device.free_memory(self.sphere_buffer_memory, None);
            self.device.free_memory(self.bvh_buffer_memory, None);
            self.device.free_memory(self.quad_buffer_memory, None);
            self.device.free_memory(self.triangle_buffer_memory, None);

            for &buffer in self.material_buffers.iter() {
                self.device.destroy_buffer(buffer, None);
            }
            for &memory in self.material_buffers_memory.iter() {
                self.device.free_memory(memory, None);
            }

            for &buffer in self.uniform_buffers.iter() {
                self.device.destroy_buffer(buffer, None);
            }
            for &memory in self.uniform_buffers_memory.iter() {
                self.device.free_memory(memory, None);
            }

            self.device.destroy_pipeline(self.compute_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.compute_pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.compute_descriptor_set_layout, None);

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device
                .destroy_descriptor_pool(self.compute_descriptor_pool, None);

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

            self.device.destroy_fence(self.compute_fence, None);

            for &semaphore in self.compute_copy_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }

            self.device
                .destroy_query_pool(self.query_pool_timestamps, None);

            self.device
                .destroy_command_pool(self.compute_command_pool, None);
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
