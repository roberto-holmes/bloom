pub mod api;
pub mod camera;
mod core;
mod debug;
pub mod material;
mod physics;
pub mod primitives;
pub mod quaternion;
mod ray;
mod structures;
mod tools;
mod transfer;
mod uniforms;
pub mod vec;
mod viewport;
mod vulkan;

use std::{
    sync::{mpsc, Arc, RwLock},
    thread::{self, JoinHandle},
    time::{Duration, SystemTime},
};

use anyhow::Result;
use api::{BloomAPI, Bloomable};
use ash::{vk, Entry};
use core::*;
use debug::ValidationInfo;
use structures::SurfaceStuff;
use uniforms::UniformBufferObject;
use vulkan::Destructor;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{self, ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const WINDOW_WIDTH: u32 = 1920;
const WINDOW_HEIGHT: u32 = 1080;
// const WINDOW_WIDTH: u32 = 400;
// const WINDOW_HEIGHT: u32 = 300;
const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub const MAX_MATERIAL_COUNT: usize = 10;
pub const MAX_SPHERE_COUNT: usize = 20;
pub const MAX_QUAD_COUNT: usize = 20;
pub const MAX_TRIANGLE_COUNT: usize = 20;
pub const MAX_OBJECT_COUNT: usize = MAX_SPHERE_COUNT + MAX_QUAD_COUNT + MAX_TRIANGLE_COUNT;

#[cfg(debug_assertions)]
const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: true,
    required_validation_layers: [
        "VK_LAYER_KHRONOS_shader_object",
        "VK_LAYER_KHRONOS_validation",
    ],
};

#[cfg(not(debug_assertions))]
const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: false,
    required_validation_layers: ["", ""],
};

pub fn run<T: Bloomable + Sync + Send + 'static>(user_app: T) {
    env_logger::init();
    // log::error!("Testing Error");
    // log::warn!("Testing Warn");
    // log::info!("Testing Info");
    // log::debug!("Testing Debug");
    // log::trace!("Testing Trace");

    let user_app = Arc::new(RwLock::new(user_app));
    let mut app = App::new(user_app);
    let event_loop = match EventLoop::new() {
        Ok(v) => v,
        Err(e) => {
            log::error!("Failed to create event loop: {e}");
            return;
        }
    };

    event_loop.set_control_flow(event_loop::ControlFlow::Poll);

    match event_loop.run_app(&mut app) {
        Err(e) => {
            log::error!("Failed to run bloom app {e}")
        }
        Ok(()) => {}
    };
}

struct App<T: Bloomable + Sync + Send + 'static> {
    vulkan: Option<VulkanApp<T>>,
    window: Option<Arc<RwLock<Window>>>,
    last_frame_time: SystemTime,

    is_initialised: bool,

    user_app: Arc<RwLock<T>>,
}

impl<T: Bloomable + Sync + Send + 'static> App<T> {
    pub fn new(user_app: Arc<RwLock<T>>) -> Self {
        Self {
            window: None,
            vulkan: None,
            last_frame_time: SystemTime::now(),

            is_initialised: false,

            user_app,
        }
    }
}

impl<T: Bloomable + Sync + Send + 'static> ApplicationHandler for App<T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.is_initialised {
            log::warn!("Window is already initialised but has been resumed");
            return;
        }
        // Create window with whatever default attributes we want
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Bloom")
                    .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
            )
            .unwrap();

        // Give the window to the user incase they want it
        let window = Arc::new(RwLock::new(window));
        // Initialise vulkan app
        let vulkan_user_app = Arc::clone(&self.user_app);
        self.vulkan = Some(
            VulkanApp::new(window.clone(), vulkan_user_app).expect("Failed to create vulkan app"),
        );

        match self.user_app.write() {
            Ok(mut app) => app.init_window(Arc::clone(&window)),
            Err(e) => log::error!("Failed to give user app the window: {e}"),
        }
        // Store the window for later use
        self.window = Some(window);
        self.is_initialised = true;
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
                // log::info!("Resize event");
                match self.vulkan.as_mut().unwrap().user_app.write() {
                    Ok(mut app) => app.resize(physical_size.width, physical_size.height),
                    Err(e) => log::error!("Failed to give user app the resize event: {e}"),
                }

                self.vulkan.as_mut().unwrap().resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                let delta_time = match self.last_frame_time.elapsed() {
                    Ok(elapsed) => elapsed,
                    Err(e) => {
                        log::error!("Failed to get time since last frame: {e:?}");
                        Duration::default()
                    }
                };
                log::trace!(
                    "Wanting frame time of {} μs [{:.2} fps]",
                    delta_time.as_micros(),
                    1.0 / delta_time.as_secs_f32()
                );
                self.last_frame_time = SystemTime::now();
                self.vulkan.as_mut().unwrap().draw(delta_time);
            }
            // WindowEvent::KeyboardInput {
            //     event:
            //         KeyEvent {
            //             state: ElementState::Pressed,
            //             physical_key: PhysicalKey::Code(KeyCode::KeyD),
            //             ..
            //         },
            //     ..
            // } => {
            //     log::info!("Draw button pressed");
            //     self.vulkan.as_mut().unwrap().draw(Duration::default());
            // }
            e => match self.user_app.write() {
                Ok(mut app) => app.input(e),
                Err(e) => log::error!("Failed to give user app the window event: {e}"),
            },
        }
    }
}

#[allow(dead_code)]
struct VulkanApp<T: Bloomable + Sync + Send + 'static> {
    user_app: Arc<RwLock<T>>,
    current_size: PhysicalSize<u32>,

    should_ray_die: Arc<RwLock<bool>>,
    ray_thread: Option<JoinHandle<()>>,

    should_transfer_die: Arc<RwLock<bool>>,
    transfer_thread: Option<JoinHandle<()>>,

    should_viewport_die: Arc<RwLock<bool>>,
    viewport_thread: Option<JoinHandle<()>>,

    should_uniforms_die: Arc<RwLock<bool>>,
    uniforms_thread: Option<JoinHandle<()>>,

    should_physics_die: Arc<RwLock<bool>>,
    physics_thread: Option<JoinHandle<()>>,

    draw_event: mpsc::SyncSender<bool>,
    resize_event: single_value_channel::Updater<PhysicalSize<u32>>,
    mem_properties: vk::PhysicalDeviceMemoryProperties,

    transfer_semaphore: vulkan::Destructor<vk::Semaphore>,
    viewport_semaphore: vulkan::Destructor<vk::Semaphore>,

    physical_device: vk::PhysicalDevice,
    debug_messenger: Option<debug::DebugUtils>,
    // surface_stuff: SurfaceStuff,
    allocator: Arc<vk_mem::Allocator>,
    device: vulkan::Device, // Logical Device
    instance: vulkan::Instance,
    _entry: Entry,
}

impl<T: Bloomable + Sync + Send + 'static> VulkanApp<T> {
    pub fn new(window: Arc<RwLock<Window>>, user_app: Arc<RwLock<T>>) -> Result<Self> {
        log::debug!("Initialising vulkan application");
        let entry = unsafe { Entry::load()? };
        // Set up Vulkan API
        let instance = core::create_instance(&entry, &window.read().unwrap())?;
        // debug::print_available_instance_extensions(&entry)?;
        // Set up callback for Vulkan debug messages
        let debug_messenger = debug::DebugUtils::new(&entry, instance.get())?;

        let surface_stuff = SurfaceStuff::new(&entry, instance.get(), &window.read().unwrap())?;
        // Pick a graphics card
        let physical_device = pick_physical_device(instance.get(), &surface_stuff)?;
        let (device, queue_family_indices) =
            create_logical_device(instance.get(), physical_device, &surface_stuff)?;

        let mut allocator_create_info =
            vk_mem::AllocatorCreateInfo::new(&instance.get(), &device.get(), physical_device);
        allocator_create_info.vulkan_api_version = vk::API_VERSION_1_3;
        allocator_create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator = Arc::new(unsafe { vk_mem::Allocator::new(allocator_create_info)? });

        let mem_properties = unsafe {
            instance
                .get()
                .get_physical_device_memory_properties(physical_device)
        };

        let ray_uniform_buffers = create_uniform_buffer::<UniformBufferObject>(&allocator)?;
        let viewport_uniform_buffers = create_uniform_buffer::<UniformBufferObject>(&allocator)?;

        let ray_raw_uniform_buffers = [ray_uniform_buffers[0].get(), ray_uniform_buffers[1].get()];
        let viewport_raw_uniform_buffers = [
            viewport_uniform_buffers[0].get(),
            viewport_uniform_buffers[1].get(),
        ];

        let should_transfer_die = Arc::new(RwLock::new(false));
        let should_viewport_die = Arc::new(RwLock::new(false));
        let should_uniforms_die = Arc::new(RwLock::new(false));
        let should_ray_die = Arc::new(RwLock::new(false));
        let should_physics_die = Arc::new(RwLock::new(false));

        let should_transfer_die_out = Arc::clone(&should_transfer_die);
        let should_viewport_die_out = Arc::clone(&should_viewport_die);
        let should_uniforms_die_out = Arc::clone(&should_uniforms_die);
        let should_ray_die_out = Arc::clone(&should_ray_die);
        let should_physics_die_out = Arc::clone(&should_physics_die);

        let transfer_device = device.get().clone();
        let viewport_device = device.get().clone();
        let ray_device = device.get().clone();

        let transfer_instance = instance.get().clone();
        let viewport_instance = instance.get().clone();
        let ray_instance = instance.get().clone();

        let transfer_physical_device = physical_device.clone();
        let viewport_physical_device = physical_device.clone();
        let ray_physical_device = physical_device.clone();

        // TODO: Replace mpsc with crossbeam?

        let (ray_transfer_resize_sender, transfer_resize_receiver) = mpsc::channel();
        let (ray_transfer_command_sender, ray_transfer_command_receiver) = mpsc::channel();
        let (viewport_transfer_command_sender, viewport_transfer_command_receiver) =
            mpsc::sync_channel(1);
        let (ray_frame_complete_receiver, ray_frame_complete_sender) =
            single_value_channel::channel_starting_with((0_u8, 0_u32));
        let (update_acceleration_structure_sender, update_acceleration_structure_receiver) =
            mpsc::channel();
        let (add_material_sender, add_material_receiver) = mpsc::channel();
        let (ray_update_uniforms_sender, update_uniforms_receiver) = mpsc::channel();
        let (update_scene_sender, update_scene_receiver) = mpsc::channel();

        // Need to create a rendezvouz channel to ensure we have updated the frame count in the uniform before we send the viewport to work
        let (ray_frame_count_sender, ray_frame_count_receiver) = mpsc::sync_channel(0);
        // let (ray_frame_count_sender, ray_frame_count_receiver) = mpsc::channel();

        let (resize_receiver, resize_sender) =
            single_value_channel::channel_starting_with(PhysicalSize {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
            });
        let (draw_sender, draw_receiver) = mpsc::sync_channel(1);

        let (ray_latest_frame_index_receiver, ray_latest_frame_index_sender) =
            single_value_channel::channel_starting_with(0_usize);
        let (viewport_latest_frame_index_receiver, viewport_latest_frame_index_sender) =
            single_value_channel::channel_starting_with(0_usize);

        let (ray_resize_receiver, ray_resize_updater) =
            single_value_channel::channel_starting_with(PhysicalSize {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
            });
        let (viewport_resize_receiver, viewport_resize_updater) =
            single_value_channel::channel_starting_with(PhysicalSize {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
            });

        let viewport_transfer_resize_sender = ray_transfer_resize_sender.clone();
        let transfer_uniforms_sender = ray_update_uniforms_sender.clone();
        // let viewport_update_uniforms_sender = ray_update_uniforms_sender.clone();
        let physics_update_uniforms_sender = ray_update_uniforms_sender.clone();

        let transfer_semaphore = Destructor::new(
            device.get(),
            core::create_semaphore(device.get())?,
            device.get().fp_v1_0().destroy_semaphore,
        );

        let viewport_semaphore = Destructor::new(
            device.get(),
            core::create_semaphore(device.get())?,
            device.get().fp_v1_0().destroy_semaphore,
        );

        let transfer_transfer_semaphore = transfer_semaphore.get().clone();
        let viewport_transfer_semaphore = transfer_semaphore.get().clone();

        let ray_viewport_semaphore = viewport_semaphore.get().clone();
        let viewport_viewport_semaphore = viewport_semaphore.get().clone();

        let api = BloomAPI::new(add_material_sender, update_scene_sender);

        match user_app.write() {
            Ok(mut a) => a.init(api)?,
            Err(e) => log::error!("Failed to call user's init function: {e}"),
        }

        let physics_user_app = Arc::clone(&user_app);

        let physics_thread =
            match thread::Builder::new()
                .name("Character".to_string())
                .spawn(move || {
                    physics::thread(
                        Duration::from_millis(10),
                        should_physics_die_out,
                        update_scene_receiver,
                        update_acceleration_structure_sender,
                        physics_update_uniforms_sender,
                        physics_user_app,
                    );
                }) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!("Failed to create transfer thread: {e}");
                    None
                }
            };

        // TODO: Ray either needs to be able to deal with empty scenes or wait until it has data before it builds
        thread::sleep(Duration::from_secs(2));

        let transfer_thread =
            match thread::Builder::new()
                .name("Transfer".to_string())
                .spawn(move || {
                    transfer::thread(
                        transfer_device,
                        transfer_instance,
                        transfer_physical_device,
                        queue_family_indices,
                        transfer_transfer_semaphore,
                        should_transfer_die_out,
                        ray_frame_complete_receiver,
                        ray_frame_count_sender,
                        draw_receiver,
                        resize_receiver,
                        transfer_resize_receiver,
                        transfer_uniforms_sender,
                        ray_transfer_command_sender,
                        viewport_transfer_command_sender,
                        ray_resize_updater,
                        viewport_resize_updater,
                    );
                }) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!("Failed to create transfer thread: {e}");
                    None
                }
            };

        let viewport_thread =
            match thread::Builder::new()
                .name("Viewport".to_string())
                .spawn(move || {
                    viewport::thread(
                        viewport_device,
                        viewport_instance,
                        viewport_physical_device,
                        surface_stuff,
                        window,
                        queue_family_indices,
                        viewport_raw_uniform_buffers,
                        viewport_transfer_resize_sender,
                        // viewport_update_uniforms_sender,
                        viewport_transfer_semaphore,
                        viewport_viewport_semaphore,
                        should_viewport_die_out,
                        viewport_transfer_command_receiver,
                        viewport_resize_receiver,
                        viewport_latest_frame_index_sender,
                    );
                }) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!("Failed to create transfer thread: {e}");
                    None
                }
            };

        let uniforms_thread =
            match thread::Builder::new()
                .name("Uniforms".to_string())
                .spawn(move || {
                    uniforms::thread(
                        update_uniforms_receiver,
                        ray_frame_count_receiver,
                        ray_latest_frame_index_receiver,
                        viewport_latest_frame_index_receiver,
                        should_uniforms_die_out,
                        ray_uniform_buffers,
                        viewport_uniform_buffers,
                    );
                }) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!("Failed to create transfer thread: {e}");
                    None
                }
            };

        // TODO: Choose whether to enable the compute thread or ray_tracing thread depending on HW capabilities
        let ray_thread = match thread::Builder::new()
            .name("Ray".to_string())
            .spawn(move || {
                ray::thread(
                    ray_device,
                    ray_instance,
                    ray_physical_device,
                    queue_family_indices,
                    ray_raw_uniform_buffers,
                    should_ray_die_out,
                    ray_viewport_semaphore,
                    ray_transfer_resize_sender,
                    ray_transfer_command_receiver,
                    ray_resize_receiver,
                    ray_frame_complete_sender,
                    update_acceleration_structure_receiver,
                    add_material_receiver,
                    ray_update_uniforms_sender,
                    ray_latest_frame_index_sender,
                );
            }) {
            Ok(v) => Some(v),
            Err(e) => {
                log::error!("Failed to create compute thread: {e}");
                None
            }
        };

        Ok(Self {
            user_app,
            current_size: PhysicalSize {
                width: 1920,
                height: 1080,
            },

            _entry: entry,
            instance,
            debug_messenger,
            physical_device,
            mem_properties,
            device,

            should_ray_die,
            should_transfer_die,
            ray_thread,
            transfer_thread,

            should_viewport_die,
            viewport_thread,
            should_uniforms_die,
            uniforms_thread,
            should_physics_die,
            physics_thread,
            allocator,
            draw_event: draw_sender,
            resize_event: resize_sender,
            transfer_semaphore,
            viewport_semaphore,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.current_size = new_size;
        log::trace!("Resizing to {}x{}", new_size.width, new_size.height);
        match self.resize_event.update(self.current_size) {
            Err(e) => {
                log::error!("Failed to send resize request: {e}")
            }
            Ok(()) => {}
        }
    }

    pub fn draw(&mut self, _delta_time: Duration) {
        match self.draw_event.try_send(true) {
            Err(mpsc::TrySendError::Full(_)) => {
                // log::warn!("Dropping draw request")
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                // log::error!("Failed to send draw request - channel has disconnected")
            }
            Ok(()) => {}
        }
    }

    fn wait_idle(&self) {
        match unsafe { self.device.get().device_wait_idle() } {
            Err(e) => log::error!("Failed to wait for the device to return to idle: {}", e),
            _ => {}
        }
    }
}

impl<T: Bloomable + Sync + Send + 'static> Drop for VulkanApp<T> {
    fn drop(&mut self) {
        log::debug!("Cleaning up");
        // Kill threads
        match self.should_transfer_die.write() {
            Ok(mut m) => *m = true,
            Err(e) => log::error!("Failed to write to transfer death mutex: {e}"),
        }
        if let Some(t) = self.transfer_thread.take() {
            log::debug!("Waiting for transfer to finish");
            t.join().unwrap();
        }
        std::thread::sleep(Duration::from_secs(1));
        match self.should_ray_die.write() {
            Ok(mut m) => *m = true,
            Err(e) => log::error!("Failed to write to ray death mutex: {e}"),
        }
        if let Some(t) = self.ray_thread.take() {
            log::debug!("Waiting for ray to finish");
            t.join().unwrap();
        }
        match self.should_viewport_die.write() {
            Ok(mut m) => *m = true,
            Err(e) => log::error!("Failed to write to viewport death mutex: {e}"),
        }
        match self.should_physics_die.write() {
            Ok(mut m) => *m = true,
            Err(e) => log::error!("Failed to write to physics death mutex: {e}"),
        }
        if let Some(t) = self.physics_thread.take() {
            log::debug!("Waiting for physics to finish");
            t.join().unwrap();
        }
        if let Some(t) = self.viewport_thread.take() {
            log::debug!("Waiting for viewport to finish");
            t.join().unwrap();
        }
        match self.should_uniforms_die.write() {
            Ok(mut m) => *m = true,
            Err(e) => log::error!("Failed to write to uniforms death mutex: {e}"),
        }
        if let Some(t) = self.uniforms_thread.take() {
            log::debug!("Waiting for uniforms to finish");
            t.join().unwrap();
        }

        self.wait_idle();
        log::debug!("Device now idle");
    }
}
