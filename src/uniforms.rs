use std::{
    sync::{Arc, RwLock, mpsc},
    time::Duration,
};

use hecs::World;
use winit::dpi::PhysicalSize;

use crate::{
    api::{self, Bloomable, DOF_SCALE, FOCAL_DISTANCE, VFOV_DEG},
    quaternion::Quaternion,
    vec::Vec3,
    vulkan,
};

pub enum Event {
    RayTick,
    Resize(PhysicalSize<u32>),
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct Camera {
    pub position: Vec3,
    pub vfov_rad: f32,
    pub quaternion: Quaternion,
    pub focal_distance: f32,
    pub dof_scale: f32,
    pub enabled: u32,
    _pad: [u32; 1], //  Aligned struct to 32 Bytes
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::default(),
            focal_distance: FOCAL_DISTANCE,
            quaternion: Quaternion::identity(),
            vfov_rad: VFOV_DEG.to_radians(),
            dof_scale: DOF_SCALE,
            enabled: 0,
            _pad: [0; 1],
        }
    }
}
impl Camera {
    fn update(&mut self, pos: Vec3, quat: Quaternion, cam: &api::Camera) {
        self.position = pos;
        self.quaternion = quat;
        self.focal_distance = cam.focal_distance;
        self.vfov_rad = cam.vfov_rad;
        self.dof_scale = cam.dof_scale;
    }
    fn enable(&mut self, enable: bool) {
        self.enabled = if enable { 1 } else { 0 };
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct UniformBufferObject {
    pub camera: Camera,
    ray_frame_num: u32,
    width: u32,
    height: u32,
    random_num: u32,
}

impl UniformBufferObject {
    pub fn new() -> Self {
        Self {
            camera: Camera::default(),
            ray_frame_num: 0,
            width: 0,
            height: 0,
            random_num: rand::random(),
        }
    }
    pub fn tick_ray(&mut self) {
        // log::trace!("Ray frame num is now {}", self.ray_frame_num);
        self.ray_frame_num += 1;
    }
    pub fn refresh_random_number(&mut self) {
        self.random_num = rand::random();
    }
}

pub fn thread<T: Bloomable>(
    user_app: T,

    channel: mpsc::Receiver<Event>,
    latest_ray_frame_index: Arc<RwLock<usize>>,
    latest_viewport_frame_index: Arc<RwLock<usize>>,
    should_threads_die: Arc<RwLock<bool>>,
    mut event_rx: bus::BusReader<api::Event>,
    world: Arc<RwLock<World>>,

    mut ray_ubo_buffer: [vulkan::Buffer; 2],
    mut viewport_ubo_buffer: [vulkan::Buffer; 2],
) {
    let mut ubo = UniformBufferObject::new();
    loop {
        // Check if we should end the thread
        match should_threads_die.read() {
            Ok(should_die) => {
                if *should_die == true {
                    break;
                }
            }
            Err(e) => {
                log::error!("Transfer rwlock is poisoned, ending thread: {}", e);
                break;
            }
        }
        match event_rx.recv_timeout(Duration::from_millis(1)) {
            Ok(api::Event::PostPhysics) => match user_app.get_active_camera() {
                Some(camera_entity) => {
                    ubo.camera.enable(true);
                    let mut w = world.write().unwrap();
                    let (camera, ori) = match w
                        .query_one_mut::<(&api::Camera, &mut api::Orientation)>(camera_entity)
                    {
                        Ok(v) => v,
                        Err(e) => {
                            log::error!("Failed to query camera: {e}");
                            return;
                        }
                    };
                    ubo.camera.update(ori.pos, ori.quat, camera);
                }
                None => {
                    ubo.camera.enable(false);
                }
            },
            Ok(_) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        match channel.try_recv() {
            Ok(Event::RayTick) => ubo.tick_ray(),
            Ok(Event::Resize(size)) => {
                ubo.width = size.width;
                ubo.height = size.height;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Transfer channel has disconnected");
                break;
            }
        }

        ubo.refresh_random_number();

        let ray_index = match latest_ray_frame_index.read() {
            Ok(v) => *v,
            Err(e) => {
                log::error!("Uniforms failed to get the latest ray frame index: {e}");
                break;
            }
        };
        let viewport_index = match latest_viewport_frame_index.read() {
            Ok(v) => *v,
            Err(e) => {
                log::error!("Uniforms failed to get the latest viewport frame index: {e}");
                break;
            }
        };

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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn new_ubo() {
        let ubo = UniformBufferObject::new();
        assert_eq!(ubo.ray_frame_num, 0);
        assert_eq!(ubo.width, 0);
        assert_eq!(ubo.height, 0);
    }

    #[test]
    fn tick_ray() {
        let frames = 762_436;
        let mut ubo = UniformBufferObject::new();
        ubo.tick_ray();
        assert_eq!(ubo.ray_frame_num, 1);
        for _ in 1..frames {
            ubo.tick_ray();
        }
        assert_eq!(ubo.ray_frame_num, frames);
    }
}
