use std::sync::{Mutex, Weak};

use winit::event::WindowEvent;

use crate::{
    camera::Camera,
    core::{self, UniformBufferObject},
    primitives,
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
    // pub scene: Scene,
    pub camera: Camera,
    pub uniform: core::UniformBufferObject,
    // scene: primitives::Scene, // TODO: How do we tell when it has changed so we can rebuild/update buffers
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
            // scene: primitives::Scene::default(),
        }
    }
    pub(crate) fn update_camera(&mut self) {
        self.uniform.update_camera(&self.camera);
    }
    //  TODO: Figure out how we expose the Sphere object without needing to provide an allocator
    //// Adds a sphere to the scene and returns the index of the new sphere
    // pub fn add_sphere(&mut self, sphere: primitives::Sphere) -> usize {
    //     self.scene.spheres.push(sphere);
    //     self.scene.spheres.len() - 1
    // }
    //// Adds a sphere to the scene and returns the index of the new sphere
    // pub fn batch_add_spheres(&mut self, sphere: [primitives::Sphere]) -> usize {
    //     self.scene.spheres.push(sphere);
    //     self.scene.spheres.len() - 1
    // }
    // pub fn remove_sphere(&mut self, spheres: usize) -> primitives::Sphere {}
    // pub fn batch_remove_spheres(&mut self, spheres: &[usize]) -> [primitives::Sphere] {}
    // pub fn move_sphere(&mut self, sphere: usize, dest: Vec3) {}
}
