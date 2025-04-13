use std::{
    sync::{mpsc, Arc, RwLock},
    time::Duration,
};

use ash::vk;
use bytemuck::Zeroable;
use winit::dpi::PhysicalSize;

use crate::{
    api::{DOF_SCALE, FOCAL_DISTANCE, VFOV_DEG},
    camera::{Camera, CameraUniforms},
    vec::Vec3,
    vulkan, WINDOW_HEIGHT, WINDOW_WIDTH,
};

pub enum Event {
    RayTick,
    ViewportTick,
    UpdateCamera, // TODO
    ResetSamples,
    Resize(PhysicalSize<u32>),
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct UniformBufferObject {
    pub camera: CameraUniforms,
    pub frame_num: u32,
    pub ray_frame_num: u32, // TODO: Implement in a thread safe way
    pub width: u32,
    pub height: u32,
}

impl UniformBufferObject {
    pub fn new() -> Self {
        Self {
            camera: CameraUniforms::zeroed(),
            frame_num: 0,
            ray_frame_num: 0,
            width: 0,
            height: 0,
        }
    }
    pub fn update(&mut self, extent: vk::Extent2D) -> Self {
        self.width = extent.width;
        self.height = extent.height;
        *self
    }
    pub fn tick(&mut self) {
        self.frame_num += 1;
    }
    pub fn tick_ray(&mut self) {
        // log::trace!("Ray frame num is now {}", self.ray_frame_num);
        self.ray_frame_num += 1;
    }
    pub fn reset_samples(&mut self) {
        self.ray_frame_num = 1;
        self.frame_num = 0;
    }
    // pub fn update_camera(&mut self, camera: &Camera) {
    //     self.camera = *camera.uniforms();
    // }
}

pub fn thread(
    channel: mpsc::Receiver<Event>,
    mut latest_ray_frame_index: single_value_channel::Receiver<usize>,
    mut latest_viewport_frame_index: single_value_channel::Receiver<usize>,
    should_threads_die: Arc<RwLock<bool>>,

    mut ray_ubo_buffer: [vulkan::Buffer; 2],
    mut viewport_ubo_buffer: [vulkan::Buffer; 2],
) {
    let mut ubo = UniformBufferObject::new();
    // TODO: Get this data from somewher else
    let cam = Camera::look_at(
        Vec3::new(9., 6., 9.),
        Vec3::new(0., 0., 0.),
        Vec3::new(0., 1., 0.),
        FOCAL_DISTANCE,
        VFOV_DEG,
        DOF_SCALE,
    );
    ubo.camera = cam.uniforms;
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
        match channel.recv_timeout(Duration::new(0, 10_000_000)) {
            Ok(Event::RayTick) => ubo.tick_ray(),
            Ok(Event::ViewportTick) => ubo.tick(),
            Ok(Event::ResetSamples) => ubo.reset_samples(),
            Ok(Event::Resize(size)) => {
                ubo.width = size.width;
                ubo.height = size.height
            }
            Ok(Event::UpdateCamera) => {
                // TODO
                //ubo.update_camera();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::error!("Transfer channel has disconnected");
                break;
            }
        }

        let ray_index = *latest_ray_frame_index.latest();
        let viewport_index = *latest_viewport_frame_index.latest();

        match ray_ubo_buffer[ray_index].populate_mapped(&ubo, 1) {
            Ok(()) => {}
            Err(e) => log::warn!("Failed to populate Uniforms Buffer: {e}"),
        };
        match viewport_ubo_buffer[viewport_index].populate_mapped(&ubo, 1) {
            Ok(()) => {}
            Err(e) => log::warn!("Failed to populate Uniforms Buffer: {e}"),
        };
    }
}
