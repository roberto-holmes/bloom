use std::{
    sync::{mpsc, Arc, RwLock},
    time::Duration,
};

use winit::dpi::PhysicalSize;

use crate::{camera::Camera, quaternion::Quaternion, vec::Vec3, vulkan};

pub enum Event {
    RayTick,
    UpdateCameraPosition(Vec3),
    UpdateCameraQuaternion(Quaternion),
    ResetSamples,
    Resize(PhysicalSize<u32>),
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct UniformBufferObject {
    camera: Camera,
    ray_frame_num: u32,
    ray_frame_num_out: u32,
    width: u32,
    height: u32,
    random_num: u32,
}

impl UniformBufferObject {
    pub fn new() -> Self {
        Self {
            camera: Camera::default(),
            ray_frame_num: 0,
            ray_frame_num_out: 0,
            width: 0,
            height: 0,
            random_num: rand::random(),
        }
    }
    pub fn tick_ray(&mut self) {
        // log::trace!("Ray frame num is now {}", self.ray_frame_num);
        self.ray_frame_num += 1;
    }
    pub fn set_ray(&mut self, new_value: u32) {
        log::trace!("Ray frame out is now {}", self.ray_frame_num);
        self.ray_frame_num_out = new_value;
    }
    pub fn reset_samples(&mut self) {
        self.ray_frame_num = 0;
        self.ray_frame_num_out = 0;
    }
    pub fn update_camera_position(&mut self, position: Vec3) {
        self.camera.position = position;
    }
    pub fn update_camera_quaternion(&mut self, quaternion: Quaternion) {
        self.camera.quaternion = quaternion;
    }
    pub fn refresh_random_number(&mut self) {
        self.random_num = rand::random();
    }
}

pub fn thread(
    channel: mpsc::Receiver<Event>,
    ray_frame_count: mpsc::Receiver<u32>,
    mut latest_ray_frame_index: single_value_channel::Receiver<usize>,
    mut latest_viewport_frame_index: single_value_channel::Receiver<usize>,
    should_threads_die: Arc<RwLock<bool>>,

    mut ray_ubo_buffer: [vulkan::Buffer; 2],
    mut viewport_ubo_buffer: [vulkan::Buffer; 2],
) {
    let mut ubo = UniformBufferObject::new();
    let mut total_ray_frames = 0;
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
        match ray_frame_count.try_recv() {
            Ok(v) => {
                // Ray frame count will always increase so we need to figure out the resets ourselves
                ubo.set_ray(v - total_ray_frames);
                // log::warn!(
                //     "Updating frame count to {v} - {total_ray_frames} = {}",
                //     v - total_ray_frames
                // );
                // continue;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Ray frame channel has disconnected");
                break;
            }
        }
        match channel.recv_timeout(Duration::new(0, 1_000_000)) {
            Ok(Event::RayTick) => ubo.tick_ray(),
            Ok(Event::ResetSamples) => {
                total_ray_frames += ubo.ray_frame_num_out;
                ubo.reset_samples();
            }
            Ok(Event::Resize(size)) => {
                ubo.width = size.width;
                ubo.height = size.height;
                total_ray_frames += ubo.ray_frame_num_out;
                ubo.reset_samples();
            }
            Ok(Event::UpdateCameraPosition(position)) => {
                // log::info!("Updating position to {position}");
                ubo.update_camera_position(position);
                total_ray_frames += ubo.ray_frame_num_out;
                ubo.reset_samples();
            }
            Ok(Event::UpdateCameraQuaternion(quaternion)) => {
                // log::info!("Updating quaternion to {quaternion}");
                ubo.update_camera_quaternion(quaternion);
                total_ray_frames += ubo.ray_frame_num_out;
                ubo.reset_samples();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::error!("Transfer channel has disconnected");
                break;
            }
        }

        ubo.refresh_random_number();

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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn new_ubo() {
        let ubo = UniformBufferObject::new();
        assert_eq!(ubo.ray_frame_num, 0);
        assert_eq!(ubo.ray_frame_num_out, 0);
        assert_eq!(ubo.width, 0);
        assert_eq!(ubo.height, 0);
    }

    // #[test]
    // fn tick_viewport() {
    //     let frames = 19_165;
    //     let mut ubo = UniformBufferObject::new();
    //     ubo.tick();
    //     assert_eq!(ubo.frame_num, 1);
    //     for _ in 1..frames {
    //         ubo.tick();
    //     }
    //     assert_eq!(ubo.frame_num, frames);
    //     // for _ in frames..u32::MAX {
    //     //     ubo.tick();
    //     // }
    //     // assert_eq!(ubo.frame_num, u32::MAX);
    // }

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
    fn set_output_ray() {
        let mut ubo = UniformBufferObject::new();
        assert_eq!(ubo.ray_frame_num_out, 0);
        ubo.set_ray(1);
        assert_eq!(ubo.ray_frame_num_out, 1);
        ubo.set_ray(762_436);
        assert_eq!(ubo.ray_frame_num_out, 762_436);
        ubo.set_ray(u32::MAX);
        assert_eq!(ubo.ray_frame_num_out, u32::MAX);
    }

    #[test]
    fn reset_samples() {
        let mut ubo = UniformBufferObject::new();
        for _ in 0..26 {
            ubo.tick_ray();
        }
        ubo.set_ray(1_136_455);
        ubo.reset_samples();
        assert_eq!(ubo.ray_frame_num_out, 0);
        assert_eq!(ubo.ray_frame_num, 0);
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
