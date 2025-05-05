use std::{
    sync::{mpsc, Arc, RwLock},
    time::Duration,
};

use winit::dpi::PhysicalSize;

use crate::{camera::Camera, quaternion::Quaternion, vec::Vec3, vulkan};

pub enum Event {
    RayTick,
    Resize(PhysicalSize<u32>),
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct UniformBufferObject {
    camera: Camera,
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
    pub fn update_camera_position(&mut self, position: Vec3) -> bool {
        // Return true if the value has changed
        if self.camera.position != position {
            self.camera.position = position;
            return true;
        }
        return false;
    }
    pub fn update_camera_quaternion(&mut self, quaternion: Quaternion) -> bool {
        // Return true if the value has changed
        if self.camera.quaternion != quaternion {
            self.camera.quaternion = quaternion;
            return true;
        }
        return false;
    }
    pub fn refresh_random_number(&mut self) {
        self.random_num = rand::random();
    }
}

pub fn thread(
    channel: mpsc::Receiver<Event>,
    latest_ray_frame_index: Arc<RwLock<usize>>,
    latest_viewport_frame_index: Arc<RwLock<usize>>,
    should_threads_die: Arc<RwLock<bool>>,
    camera_pos_in: Arc<RwLock<Vec3>>,
    camera_quat_in: Arc<RwLock<Quaternion>>,

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
                log::error!("Transfer rwlock is poisoned, ending thread: {}", e)
            }
        }
        match camera_pos_in.read() {
            Ok(v) => {
                let _ = ubo.update_camera_position(*v);
            }
            Err(e) => {
                log::error!("Uniform's camera pos arc is poisoned: {e}");
            }
        }
        match camera_quat_in.read() {
            Ok(v) => {
                let _ = ubo.update_camera_quaternion(*v);
            }
            Err(e) => {
                log::error!("Uniform's camera pos arc is poisoned: {e}");
            }
        }
        match channel.recv_timeout(Duration::from_millis(1)) {
            Ok(Event::RayTick) => ubo.tick_ray(),
            Ok(Event::Resize(size)) => {
                ubo.width = size.width;
                ubo.height = size.height;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
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
        // for _ in frames..u32::MAX {
        //     ubo.tick_ray();
        // }
        // assert_eq!(ubo.ray_frame_num, u32::MAX);
    }

    #[test]
    fn update_position() {
        let mut ubo = UniformBufferObject::new();
        let pos = Vec3::new(1.2, -123_023.135, 0.0);
        ubo.update_camera_position(pos);
        assert_eq!(ubo.camera.position, pos);
    }

    #[test]
    fn update_rotation() {
        let mut ubo = UniformBufferObject::new();
        let quat = Quaternion::new(1_465_021.212, -0.0, -123_023.135, 0.0);
        ubo.update_camera_quaternion(quat);
        assert_eq!(ubo.camera.quaternion, quat);
    }
}
