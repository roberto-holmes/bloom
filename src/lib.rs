pub mod api;
// mod bvh;
mod camera;
mod compute;
mod core;
mod debug;
pub mod material;
pub mod primitives;
mod ray;
// pub mod select;
mod structures;
mod tools;
mod transfer;
pub mod vec;
mod vulkan;

use std::{
    sync::{mpsc, Arc, Mutex, RwLock},
    thread::{self, JoinHandle},
    time::{Duration, SystemTime},
    u64,
};

use anyhow::Result;
use api::{BloomAPI, Bloomable};
use ash::{vk, Entry};
use core::*;
use debug::ValidationInfo;
use structures::{QueueFamilyIndices, SurfaceStuff, SwapChainStuff};
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
// const IDEAL_RADIANCE_IMAGE_SIZE_WIDTH: u32 = 5120;
const IDEAL_RADIANCE_IMAGE_SIZE_WIDTH: u32 = WINDOW_WIDTH;
// const IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT: u32 = 2160;
const IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT: u32 = WINDOW_HEIGHT;

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

pub fn run<T: Bloomable>(user_app: T) {
    env_logger::init();
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

struct App<'a, T: Bloomable> {
    window: Option<Window>,
    vulkan: Option<VulkanApp<'a, T>>,
    last_frame_time: SystemTime,

    user_app: Option<T>,
}

impl<'a, T: Bloomable> App<'a, T> {
    pub fn new(user_app: T) -> Self {
        Self {
            window: None,
            vulkan: None,
            last_frame_time: SystemTime::now(),

            user_app: Some(user_app),
        }
    }
}

impl<'a, T: Bloomable> ApplicationHandler for App<'a, T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Bloom")
                    .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
            )
            .unwrap();
        self.vulkan = Some(
            VulkanApp::new(&window, self.user_app.take().unwrap())
                .expect("Failed to create vulkan app"),
        );
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
                self.vulkan
                    .as_mut()
                    .unwrap()
                    .user_app
                    .resize(physical_size.width, physical_size.height);
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
            e => self.vulkan.as_mut().unwrap().user_app.input(e),
        }
    }
}

#[allow(dead_code)]
struct VulkanApp<'a, T: Bloomable> {
    pub user_app: T,
    api: Arc<Mutex<BloomAPI>>,
    user_app_initialised: bool,

    should_ray_die: Arc<RwLock<bool>>,
    should_transfer_die: Arc<RwLock<bool>>,
    ray_tracing_thread: Option<JoinHandle<()>>,
    transfer_thread: Option<JoinHandle<()>>,
    transfer_sender: mpsc::Sender<u8>,
    graphic_receiver: mpsc::Receiver<u64>,
    graphic_transfer_semaphore: Destructor<vk::Semaphore>, // TODO: Should this object own the destructor?

    mem_properties: vk::PhysicalDeviceMemoryProperties,
    graphics_queue: vk::Queue,
    _presentation_queue: vk::Queue,
    swapchain_stuff: SwapChainStuff,
    queue_family_indices: QueueFamilyIndices,

    query_pool_timestamps: Destructor<vk::QueryPool>,
    timestamps: Vec<u64>,

    pipeline_layout: Destructor<vk::PipelineLayout>,
    graphics_pipeline: Destructor<vk::Pipeline>,
    command_pool: Destructor<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    vertex_buffer: Destructor<vk::Buffer>,
    vertex_buffer_memory: Destructor<vk::DeviceMemory>,
    index_buffer: Destructor<vk::Buffer>,
    index_buffer_memory: Destructor<vk::DeviceMemory>,

    set_layout: Destructor<vk::DescriptorSetLayout>,
    descriptor_pool: Destructor<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    uniform_buffers: Vec<Destructor<vk::Buffer>>,
    uniform_buffers_memory: Vec<Destructor<vk::DeviceMemory>>,
    uniform_buffers_mapped: Vec<*mut core::UniformBufferObject>,

    graphic_images: [vulkan::Image<'a>; 2],

    depth_image_view: Destructor<vk::ImageView>,
    depth_image: Destructor<vk::Image>,
    depth_image_memory: Destructor<vk::DeviceMemory>,

    image_available_semaphores: Vec<Destructor<vk::Semaphore>>,
    render_finished_semaphores: Vec<Destructor<vk::Semaphore>>,
    in_flight_fences: Vec<Destructor<vk::Fence>>,
    minimized: bool,
    resized: bool,

    current_frame_index: usize,
    frame_count: u128,

    device: vulkan::Device, // Logical Device
    physical_device: vk::PhysicalDevice,
    debug_messenger: Option<debug::DebugUtils>,
    surface_stuff: SurfaceStuff,
    instance: vulkan::Instance,
    _entry: Entry,
}

impl<'a, T: Bloomable> VulkanApp<'a, T> {
    pub fn new(window: &Window, mut user_app: T) -> Result<Self> {
        let api = Arc::new(Mutex::new(BloomAPI::new()));
        user_app.init(Arc::downgrade(&api));

        log::debug!("Initialising vulkan application");
        let entry = unsafe { Entry::load()? };
        // Set up Vulkan API
        let instance = core::create_instance(&entry, window)?;
        // debug::print_available_instance_extensions(&entry)?;
        // Set up callback for Vulkan debug messages
        let debug_messenger = debug::DebugUtils::new(&entry, instance.get())?;

        let surface_stuff = SurfaceStuff::new(&entry, instance.get(), window)?;
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

        let graphics_queue =
            core::create_queue(device.get(), queue_family_indices.graphics_family.unwrap());
        let presentation_queue =
            core::create_queue(device.get(), queue_family_indices.present_family.unwrap());

        let (query_pool_timestamps, timestamps) = prepare_timestamp_queries(device.get())?;

        let swapchain_stuff = SwapChainStuff::new(
            instance.get(),
            device.get(),
            physical_device,
            &surface_stuff,
            &queue_family_indices,
        )?;
        let set_layout = create_descriptor_set_layout(device.get())?;
        let descriptor_pool = create_descriptor_pool(device.get())?;

        let (pipeline_layout, pipelines) =
            create_graphics_pipeline(device.get(), &swapchain_stuff, set_layout.get())?;

        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
            create_uniform_buffer::<UniformBufferObject>(device.get(), mem_properties, 1)?;

        let (depth_image, depth_image_memory, depth_image_view) = create_depth_resources(
            device.get(),
            instance.get(),
            physical_device,
            mem_properties,
            &swapchain_stuff,
        )?;

        let command_pool = create_command_pool(
            device.get(),
            queue_family_indices.graphics_family.unwrap().0,
        )?;

        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(
            device.get(),
            mem_properties,
            command_pool.get(),
            graphics_queue,
        )?;

        let (index_buffer, index_buffer_memory) = create_index_buffer(
            device.get(),
            mem_properties,
            command_pool.get(),
            graphics_queue,
        )?;

        let graphic_images = create_storage_image_pair(
            device.get(),
            instance.get(),
            &allocator,
            physical_device,
            command_pool.get(),
            graphics_queue,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let mut raw_graphic_images = [vk::Image::null(); 2];
        for i in 0..graphic_images.len() {
            raw_graphic_images[i] = graphic_images[i].get();
        }
        let mut raw_graphic_image_views = [vk::ImageView::null(); 2];
        for i in 0..graphic_images.len() {
            raw_graphic_image_views[i] = graphic_images[i].view();
        }

        let compute_images = create_storage_image_pair(
            device.get(),
            instance.get(),
            &allocator,
            physical_device,
            command_pool.get(),
            graphics_queue,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let mut raw_compute_images = [vk::Image::null(); 2];
        for i in 0..compute_images.len() {
            raw_compute_images[i] = compute_images[i].get();
        }

        let descriptor_sets = create_descriptor_sets(
            device.get(),
            descriptor_pool.get(),
            set_layout.get(),
            &uniform_buffers,
            &raw_graphic_image_views,
        )?;

        let command_buffers = create_command_buffers(
            device.get(),
            command_pool.get(),
            MAX_FRAMES_IN_FLIGHT as u32,
        )?;

        let should_transfer_die = Arc::new(RwLock::new(false));
        let should_ray_die = Arc::new(RwLock::new(false));
        let transfer_mutex = Arc::clone(&should_transfer_die);
        let compute_mutex = Arc::clone(&should_ray_die);

        let compute_device = device.get().clone();
        let transfer_device = device.get().clone();

        let compute_instance = instance.get().clone();

        let compute_api = Arc::clone(&api);
        let transfer_api = Arc::clone(&api);

        let (transfer_sender, transfer_receiver) = mpsc::channel();
        let (graphic_sender, graphic_receiver) = mpsc::channel();
        let (compute_sender, compute_receiver) = mpsc::channel();

        let compute_transfer_send = transfer_sender.clone();

        let transfer_semaphore = Destructor::new(
            device.get(),
            core::create_semaphore(device.get())?,
            device.get().fp_v1_0().destroy_semaphore,
        );

        let ray_tracing_transfer_semaphore = transfer_semaphore.get().clone();
        let transfer_transfer_semaphore = transfer_semaphore.get().clone();

        // TODO: Choose whether to enable the compute thread or ray_tracing thread depending on HW capabilitie
        let ray_tracing_thread = if true {
            match thread::Builder::new()
                .name("RayTracing".to_string())
                .spawn(move || {
                    ray::thread(
                        compute_device,
                        compute_instance,
                        physical_device,
                        queue_family_indices,
                        compute_images,
                        compute_mutex,
                        compute_receiver,
                        compute_transfer_send,
                        ray_tracing_transfer_semaphore,
                        IDEAL_RADIANCE_IMAGE_SIZE_WIDTH,
                        IDEAL_RADIANCE_IMAGE_SIZE_HEIGHT,
                        compute_api,
                    );
                }) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!("Failed to create compute thread: {e}");
                    None
                }
            }
        } else {
            match thread::Builder::new()
                .name("Compute".to_string())
                .spawn(move || {
                    compute::thread(
                        compute_device,
                        queue_family_indices,
                        compute_images,
                        compute_mutex,
                        compute_receiver,
                        compute_transfer_send,
                        ray_tracing_transfer_semaphore,
                    );
                }) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::error!("Failed to create compute thread: {e}");
                    None
                }
            }
        };

        let transfer_thread =
            match thread::Builder::new()
                .name("Transfer".to_string())
                .spawn(move || {
                    transfer::thread(
                        transfer_device,
                        queue_family_indices,
                        transfer_transfer_semaphore,
                        &raw_compute_images,
                        &raw_graphic_images,
                        transfer_mutex,
                        transfer_receiver,
                        compute_sender,
                        graphic_sender,
                        transfer_api,
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

        api.lock().unwrap().uniform.update(swapchain_stuff.extent);

        Ok(Self {
            user_app,
            api,
            user_app_initialised: false,

            _entry: entry,
            instance,
            debug_messenger,
            physical_device,
            mem_properties,
            device,
            graphics_queue,
            _presentation_queue: presentation_queue,
            surface_stuff,
            swapchain_stuff: swapchain_stuff,
            queue_family_indices,

            transfer_sender,
            graphic_receiver,
            should_ray_die,
            should_transfer_die,
            ray_tracing_thread,
            transfer_thread,
            graphic_transfer_semaphore: transfer_semaphore,

            query_pool_timestamps,
            timestamps,

            pipeline_layout,
            graphics_pipeline: pipelines,
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

            graphic_images,

            depth_image,
            depth_image_memory,
            depth_image_view,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,

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
                &[self.in_flight_fences[self.current_frame_index].get()],
                true,
                u64::MAX,
            ) {
                Err(e) => log::error!("Failed to wait on in_flight_fence: {:?}", e),
                _ => {}
            }
        }
        let image_index = unsafe {
            match self.swapchain_stuff.get_loader().acquire_next_image(
                self.swapchain_stuff.get_swapchain(),
                1_000_000_000, // 1 second in nanoseconds
                self.image_available_semaphores[self.current_frame_index].get(),
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
                .reset_fences(&[self.in_flight_fences[self.current_frame_index].get()])
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
            self.image_available_semaphores[self.current_frame_index].get(),
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
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame_index].get()];
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
                self.in_flight_fences[self.current_frame_index].get(),
            )
        } {
            Err(e) => {
                log::error!("Failed to submit queue: {}", e);
                return;
            }
            _ => {}
        }
        let swapchains = [self.swapchain_stuff.get_swapchain()];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match unsafe {
            self.swapchain_stuff
                .get_loader()
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
                        "Graphics pipeline took {:.2} ms [{:.2} fps]",
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
        match unsafe { self.device.get().device_wait_idle() } {
            Err(e) => log::error!("Failed to wait for the device to return to idle: {}", e),
            _ => {}
        }
    }

    fn update_uniform_buffers(&mut self, current_image: u32, _delta_time: Duration) {
        let mut api = self.api.lock().unwrap();
        // api.uniform.tick();
        api.uniform.update(self.swapchain_stuff.extent);
        // log::info!("Ray frame {}", api.uniform.ray_frame_num);

        api.update_camera();

        let ubos = [api.uniform.clone()];

        unsafe {
            self.uniform_buffers_mapped[current_image as usize]
                .copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());
        };
    }

    fn recreate_swap_chain(&mut self) {
        if self.minimized == true {
            return;
        }
        unsafe {
            match self.device.get().queue_wait_idle(self.graphics_queue) {
                Err(e) => log::error!("Failed to wait for graphics queue to finish: {:?}", e),
                _ => {}
            }
        };

        self.swapchain_stuff
            .reset(
                self.instance.get(),
                &self.device.get(),
                self.physical_device,
                &self.surface_stuff,
                &self.queue_family_indices,
            )
            .unwrap();

        (
            self.depth_image,
            self.depth_image_memory,
            self.depth_image_view,
        ) = match create_depth_resources(
            self.device.get(),
            self.instance.get(),
            self.physical_device,
            self.mem_properties,
            &self.swapchain_stuff,
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
        self.api.lock().unwrap().uniform.reset_samples();
        self.resized = false;
    }

    pub fn record_command_buffer(&mut self, image_index: u32) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();

        let command_buffer = self.command_buffers[self.current_frame_index];

        unsafe {
            self.device
                .get()
                .begin_command_buffer(command_buffer, &begin_info)?;
            self.device.get().cmd_reset_query_pool(
                command_buffer,
                self.query_pool_timestamps.get(),
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
            .image_view(self.swapchain_stuff.image_views[image_index as usize].get()) // Can't remember if this is supposed to be image_index or current_frame_index
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)];

        // let depth_attachment = vk::RenderingAttachmentInfoKHR::default()
        //     .clear_value(depth_clear_value)
        //     .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL_KHR)
        //     .image_view(self.swapchain_stuff.image_views[image_index as usize]) // Can't remember if this is supposed to be image_index or current_frame_index
        //     .load_op(vk::AttachmentLoadOp::CLEAR)
        //     .store_op(vk::AttachmentStoreOp::STORE);

        let rendering_info = vk::RenderingInfo::default()
            .color_attachments(&colour_attachments)
            // .depth_attachment(&depth_attachment)
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_stuff.extent,
            });

        transition_image_layout(
            self.device.get(),
            self.command_pool.get(),
            self.graphics_queue,
            self.swapchain_stuff.images[image_index as usize],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;

        unsafe {
            self.device.get().cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool_timestamps.get(),
                0,
            );

            self.device
                .get()
                .cmd_begin_rendering(command_buffer, &rendering_info);
            self.device.get().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.get(),
            );

            let viewports = [vk::Viewport::default()
                .width(self.swapchain_stuff.extent.width as f32)
                .height(self.swapchain_stuff.extent.height as f32)
                .max_depth(1.0)];
            self.device
                .get()
                .cmd_set_viewport(command_buffer, 0, &viewports);

            let scissors = [vk::Rect2D::default().extent(self.swapchain_stuff.extent)];
            self.device
                .get()
                .cmd_set_scissor(command_buffer, 0, &scissors);

            let vertex_buffers = [self.vertex_buffer.get()];
            let offsets = [0];
            self.device
                .get()
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            self.device.get().cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.get(),
                0,
                vk::IndexType::UINT16,
            );

            let descriptor_sets_to_bind = [self.descriptor_sets[self.current_frame_index]];

            self.device.get().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout.get(),
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
                self.query_pool_timestamps.get(),
                1,
            );

            self.device.get().cmd_end_rendering(command_buffer);

            transition_image_layout(
                self.device.get(),
                self.command_pool.get(),
                self.graphics_queue,
                self.swapchain_stuff.images[image_index as usize],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            )?;

            self.device.get().end_command_buffer(command_buffer)?;
        };
        Ok(())
    }
}

impl<'a, T: Bloomable> Drop for VulkanApp<'a, T> {
    fn drop(&mut self) {
        log::debug!("Cleaning up");
        // Kill threads
        match self.should_transfer_die.write() {
            Ok(mut m) => *m = true,
            Err(e) => log::error!("Failed to write to transfer death mutex: {e}"),
        }
        if let Some(t) = self.transfer_thread.take() {
            log::debug!("Wating for transfer to finish");
            t.join().unwrap();
        }
        match self.should_ray_die.write() {
            Ok(mut m) => *m = true,
            Err(e) => log::error!("Failed to write to ray death mutex: {e}"),
        }
        if let Some(t) = self.ray_tracing_thread.take() {
            log::debug!("Wating for ray to finish");
            t.join().unwrap();
        }

        self.wait_idle();
        log::debug!("Device now idle");
    }
}
