use std::sync::{Mutex, Weak};

use winit::event::WindowEvent;

use crate::{
    camera::Camera,
    core::{self, UniformBufferObject},
    // primitives::Scene,
    vec::Vec3,
};

pub const FOCAL_DISTANCE: f32 = 4.5;
pub const VFOV_DEG: f32 = 40.;
pub const DOF_SCALE: f32 = 0.05;

pub trait Bloomable {
    fn init(&mut self, api: Weak<Mutex<BloomAPI>>);
    fn input(&mut self, event: WindowEvent);
    fn resize(&mut self, width: u32, height: u32);
    // fn update(&mut self);
    // fn fixed_update(&mut self);
}

pub struct BloomAPI {
    // Needs to include a way to modify all of the buffers
    // pub scene: Scene, // TODO: not make this public so we know when it has changed and don't have to update every frame
    pub camera: Camera,
    pub uniform: core::UniformBufferObject,
}

impl BloomAPI {
    pub fn new() -> Self {
        let camera = Camera::look_at(
            Vec3::new(9., 6., 9.),
            Vec3::new(0., 0., 0.),
            Vec3::new(0., 1., 0.),
            FOCAL_DISTANCE,
            VFOV_DEG,
            DOF_SCALE,
        );
        Self {
            // scene: Scene::new(),
            camera,
            uniform: UniformBufferObject::new(),
        }
    }
    pub(crate) fn update_camera(&mut self) {
        self.uniform.update_camera(&self.camera);
    }
}
