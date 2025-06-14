pub mod api;
// pub mod camera;
mod core;
mod debug;
pub mod material;
mod ocean;
mod physics;
pub mod primitives;
pub mod quaternion;
mod ray;
mod structures;
mod sync;
mod tools;
mod uniforms;
pub mod vec;
mod viewport;
mod vulkan;

use std::{
    io::Write,
    sync::{mpsc, Arc, RwLock},
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime},
};

use anyhow::Result;
use api::Bloomable;
use ash::{vk, Entry};
use bus::Bus;
use colored::Colorize;
use core::*;
use debug::ValidationInfo;
use hecs::World;
use log::LevelFilter;
use structures::SurfaceStuff;
use uniforms::UniformBufferObject;
use vulkan::Destructor;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
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

pub fn run<T: Bloomable + Clone + Sync + Send + 'static>(user_app: T) {
    env_logger::Builder::new()
        .format(|buf, record| {
            let level = match record.level() {
                log::Level::Error => "Error".red(),
                log::Level::Warn => "Warn ".yellow(),
                log::Level::Info => "Info ".green(),
                log::Level::Debug => "Debug".black(),
                log::Level::Trace => "Trace".cyan(),
            };
            let file_name = record.file().unwrap_or("unknown").to_string();
            let index_file_name_start = *file_name.rfind("src").get_or_insert(0) + 4;
            writeln!(
                buf,
                "{} {} {}::{}:{}>\t{}",
                chrono::Local::now().format("%H:%M:%S%.3f"),
                level,
                std::thread::current()
                    .name()
                    .unwrap_or(format!("{:?}", std::thread::current().id()).as_str())
                    .purple(),
                file_name
                    .get(index_file_name_start..)
                    .get_or_insert(file_name.as_str()),
                record.line().unwrap_or(0).to_string().yellow(),
                record.args()
            )
        })
        .filter(None, LevelFilter::Debug)
        // .filter(Some("logger_example"), LevelFilter::Debug)
        .init();
    // log::error!("Testing Error");
    // log::warn!("Testing Warn");
    // log::info!("Testing Info");
    // log::debug!("Testing Debug");
    // log::trace!("Testing Trace");

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

struct App<T: Bloomable + Clone + Sync + Send + 'static> {
    vulkan: Option<VulkanApp<T>>,
    window: Option<Arc<RwLock<Window>>>,
    last_frame_time: SystemTime,

    is_initialised: bool,
    world: Arc<RwLock<World>>,

    user_app: T,
}

impl<T: Bloomable + Clone + Sync + Send + 'static> App<T> {
    pub fn new(user_app: T) -> Self {
        Self {
            window: None,
            vulkan: None,
            world: Arc::new(RwLock::new(World::new())),
            last_frame_time: SystemTime::now(),

            is_initialised: false,

            user_app,
        }
    }
}

impl<T: Bloomable + Clone + Sync + Send + 'static> ApplicationHandler for App<T> {
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
        self.user_app.init_window(Arc::clone(&window));
        // Initialise vulkan app
        self.vulkan = Some(
            VulkanApp::new(self.world.clone(), window.clone(), self.user_app.clone())
                .expect("Failed to create vulkan app"),
        );

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
                self.user_app
                    .resize(physical_size.width, physical_size.height, &self.world);

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
                // Limit FPS
                if delta_time < Duration::from_micros((1_000_000.0 / 240.0) as u64) {
                    match self.window.as_ref().unwrap().write() {
                        Ok(v) => {
                            v.request_redraw();
                            return;
                        }
                        Err(e) => log::error!("Failed to delay redraw: {e}"),
                    }
                }
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
            e => self.user_app.input(e, &self.world),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        self.user_app.raw_input(device_id, event, &self.world);
    }
}

#[allow(dead_code)]
struct VulkanApp<T: Bloomable + Clone + Sync + Send + 'static> {
    world: Arc<RwLock<World>>,
    user_app: T,
    current_size: PhysicalSize<u32>,

    should_ray_die: Arc<RwLock<bool>>,
    ray_thread: Option<JoinHandle<()>>,

    should_transfer_die: Arc<RwLock<bool>>,
    sync_thread: Option<JoinHandle<()>>,

    should_viewport_die: Arc<RwLock<bool>>,
    viewport_thread: Option<JoinHandle<()>>,

    should_uniforms_die: Arc<RwLock<bool>>,
    uniforms_thread: Option<JoinHandle<()>>,

    draw_event: mpsc::SyncSender<bool>,
    resize_event: single_value_channel::Updater<PhysicalSize<u32>>,
    mem_properties: vk::PhysicalDeviceMemoryProperties,

    transfer_semaphore: vulkan::Destructor<vk::Semaphore>,
    viewport_semaphore: vulkan::Destructor<vk::Semaphore>,

    physical_device: vk::PhysicalDevice,
    debug_messenger: Option<debug::DebugUtils>,
    allocator: Arc<vk_mem::Allocator>,
    device: vulkan::Device, // Logical Device
    instance: vulkan::Instance,
    _entry: Entry,
}

impl<T> VulkanApp<T>
where
    T: Bloomable + Clone + Sync + Send + 'static,
{
    pub fn new(
        world: Arc<RwLock<World>>,
        window: Arc<RwLock<Window>>,
        mut user_app: T,
    ) -> Result<Self> {
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

        let should_transfer_die_out = Arc::clone(&should_transfer_die);
        let should_viewport_die_out = Arc::clone(&should_viewport_die);
        let should_uniforms_die_out = Arc::clone(&should_uniforms_die);
        let should_ray_die_out = Arc::clone(&should_ray_die);

        let sync_device = device.get().clone();
        let viewport_device = device.get().clone();
        let ray_device = device.get().clone();

        let sync_instance = instance.get().clone();
        let viewport_instance = instance.get().clone();
        let ray_instance = instance.get().clone();

        let sync_physical_device = physical_device.clone();
        let viewport_physical_device = physical_device.clone();
        let ray_physical_device = physical_device.clone();

        // TODO: Replace mpsc with crossbeam?

        let (ray_transfer_resize_sender, transfer_resize_receiver) = mpsc::channel();
        let (ray_transfer_command_sender, ray_transfer_command_receiver) = mpsc::sync_channel(0);
        let (viewport_transfer_command_sender, viewport_transfer_command_receiver) =
            mpsc::sync_channel(1);

        let ray_frame_complete_receiver = Arc::new(RwLock::new(ray::Update::default()));
        let ray_frame_complete_sender = ray_frame_complete_receiver.clone();

        let (ray_update_uniforms_sender, update_uniforms_receiver) = mpsc::channel();

        let (resize_receiver, resize_sender) =
            single_value_channel::channel_starting_with(PhysicalSize {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
            });
        let (draw_sender, draw_receiver) = mpsc::sync_channel(1);

        let ray_latest_frame_index_receiver = Arc::new(RwLock::new(0_usize));
        let ray_latest_frame_index_sender = ray_latest_frame_index_receiver.clone();

        let viewport_latest_frame_index_receiver = Arc::new(RwLock::new(0_usize));
        let viewport_latest_frame_index_sender = viewport_latest_frame_index_receiver.clone();

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

        user_app.init(&world)?;

        let should_systems_die = Arc::new(RwLock::new(false));
        let system_sync = mpsc::channel();
        let mut event_broadcaster = Bus::new(10);

        let ubo_event_rx = event_broadcaster.add_rx();

        let sync_user_app = user_app.clone();
        let ubo_user_app = user_app.clone();

        let ray_world = world.clone();
        let ubo_world = world.clone();

        user_app.physics_tick(Duration::ZERO, &world);

        let systems = vec![
            Self::add_system(
                user_app.clone(),
                world.clone(),
                should_systems_die.clone(),
                system_sync.0.clone(),
                api::Event::Physics,
                "Phys",
                physics::system,
                event_broadcaster.add_rx(),
            ),
            Self::add_system(
                user_app.clone(),
                world.clone(),
                should_systems_die.clone(),
                system_sync.0.clone(),
                api::Event::PrePhysics,
                "UsrP",
                T::physics_tick,
                event_broadcaster.add_rx(),
            ),
            Self::add_system(
                user_app.clone(),
                world.clone(),
                should_systems_die.clone(),
                system_sync.0.clone(),
                api::Event::GraphicsUpdate,
                "UsrD",
                T::display_tick,
                event_broadcaster.add_rx(),
            ),
        ];

        let sync_thread = match thread::Builder::new()
            .name("Sync".to_string())
            .spawn(move || {
                sync::thread(
                    sync_user_app,
                    sync_device,
                    sync_instance,
                    sync_physical_device,
                    queue_family_indices,
                    transfer_transfer_semaphore,
                    should_transfer_die_out,
                    should_systems_die,
                    event_broadcaster,
                    systems,
                    system_sync,
                    ray_frame_complete_receiver,
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
                .name("View".to_string())
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
                .name("Buff".to_string())
                .spawn(move || {
                    uniforms::thread(
                        ubo_user_app,
                        update_uniforms_receiver,
                        ray_latest_frame_index_receiver,
                        viewport_latest_frame_index_receiver,
                        should_uniforms_die_out,
                        ubo_event_rx,
                        ubo_world,
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
            .name("Ray ".to_string())
            .spawn(move || {
                ray::thread(
                    ray_world,
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
            world,
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
            sync_thread,

            should_viewport_die,
            viewport_thread,
            should_uniforms_die,
            uniforms_thread,
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

    pub fn add_system(
        mut user_app: T,
        world: Arc<RwLock<World>>,
        should_die: Arc<RwLock<bool>>,
        sync_tx: mpsc::Sender<bool>,
        step: api::Event,
        thread_name: &str,
        f: fn(&mut T, Duration, &Arc<RwLock<World>>),
        mut rx: bus::BusReader<api::Event>,
    ) -> Option<JoinHandle<()>> {
        let mut last_time = Instant::now();
        match std::thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || loop {
                match should_die.read() {
                    Ok(should_die) => {
                        if *should_die == true {
                            return;
                        }
                    }
                    Err(_) => return,
                }
                match rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(v) => {
                        if v == step {
                            f(&mut user_app, last_time.elapsed(), &world);
                            last_time = Instant::now();
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                let _ = sync_tx.send(true);
            }) {
            Ok(v) => Some(v),
            Err(_) => None,
        }
    }
}

impl<T: Bloomable + Clone + Sync + Send + 'static> Drop for VulkanApp<T> {
    fn drop(&mut self) {
        log::debug!("Cleaning up");
        // Kill threads
        match self.should_transfer_die.write() {
            Ok(mut m) => *m = true,
            Err(e) => log::error!("Failed to write to transfer death mutex: {e}"),
        }
        if let Some(t) = self.sync_thread.take() {
            log::debug!("Waiting for sync to finish");
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
