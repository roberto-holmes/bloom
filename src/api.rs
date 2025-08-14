use std::{path::PathBuf, sync::RwLock, time::Duration};

use anyhow::Result;
use cgmath::{Matrix4, Rad, SquareMatrix};
use hecs::Entity;

use crate::{
    primitives::{Extrema, AABB},
    quaternion::Quaternion,
    structures::Cubemap,
    vec::Vec3,
};

pub const FOCAL_DISTANCE: f32 = 4.5;
pub const VFOV_DEG: f32 = 40.;
pub const DOF_SCALE: f32 = 0.0;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Event {
    PrePhysics,  // Allow the user to modify anything
    Physics,     // Compute collisions etc.
    PostPhysics, // Update UBO and Acceleration structure

    PreRay,
    Ray,
    PostRay,

    GraphicsUpdate,

    Input,
}

#[derive(Debug, Clone)]
pub struct Skybox {
    pub(crate) cubemap: Cubemap<PathBuf>,
}
impl Skybox {
    pub fn new(
        px: PathBuf,
        nx: PathBuf,
        py: PathBuf,
        ny: PathBuf,
        pz: PathBuf,
        nz: PathBuf,
    ) -> Self {
        Skybox {
            cubemap: Cubemap {
                px,
                nx,
                py,
                ny,
                pz,
                nz,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Collider {
    pub aabb: AABB,
    pub last_pos: Vec3,
    pub fixed: bool,
}
impl Collider {
    pub fn new<T: Extrema>(obj: &T, fixed: bool) -> Self {
        Self {
            aabb: AABB::new(obj),
            last_pos: Vec3::zero(),
            fixed,
        }
    }
}

#[derive(Debug)]
pub struct Orientation {
    pub pos: Vec3,
    pub quat: Quaternion,
    pub(crate) transformation: Matrix4<f32>,
}

impl Orientation {
    pub fn new(pos: Vec3, quat: Quaternion) -> Self {
        let transformation = Self::calc_transformation(pos, quat);
        Self {
            pos,
            quat,
            transformation,
        }
    }
    pub fn update(&mut self) {
        self.transformation = Self::calc_transformation(self.pos, self.quat);
    }
    fn calc_transformation(pos: Vec3, quat: Quaternion) -> Matrix4<f32> {
        // TODO: Calculate matrix from pos and quat
        let (axis, angle) = quat.to_axis_angle_rad();
        Matrix4::from_translation(pos.into()) * Matrix4::from_axis_angle(axis.into(), Rad(angle))
    }
}

impl Default for Orientation {
    fn default() -> Self {
        Self {
            pos: Vec3::zero(),
            quat: Quaternion::identity(),
            transformation: Matrix4::identity(),
        }
    }
}

/// Component of entities that are positioned relative to a parent entity
#[derive(Debug)]
pub struct Child {
    /// Parent entity
    pub parent: Entity,
    /// Converts child-relative coordinates to parent-relative coordinates
    pub offset_pos: Vec3,
    pub offset_quat: Quaternion,
}

#[derive(Debug)]
pub struct Instance {
    pub primitive: Entity,
    pub base_transform: Matrix4<f32>,
    pub initial_transform: Matrix4<f32>,
}

impl Instance {
    pub fn new(primitive: Entity) -> Self {
        Self {
            primitive,
            base_transform: Matrix4::identity(),
            initial_transform: Matrix4::identity(),
        }
    }
}

#[derive(Debug)]
pub struct Camera {
    pub vfov_rad: f32,
    pub focal_distance: f32,
    pub dof_scale: f32,
    pub enabled: u32,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            focal_distance: FOCAL_DISTANCE,
            vfov_rad: VFOV_DEG.to_radians(),
            dof_scale: DOF_SCALE,
            enabled: 0,
        }
    }
}

pub trait Bloomable {
    fn get_active_camera(&self) -> Option<Entity>; // How should we deal with no camera in the scene?
    fn get_physics_update_period(&self) -> Duration;
    /// Init Window is the only function that will be called before the object starts getting cloned and
    /// everythin that wants to be used by different functions will need Arc<RwLock<>>>
    fn init_window(&mut self, window: std::sync::Arc<RwLock<winit::window::Window>>);
    fn init(&mut self, world: &std::sync::Arc<RwLock<hecs::World>>) -> Result<()>;
    fn input(
        &mut self,
        event: winit::event::WindowEvent,
        world: &std::sync::Arc<RwLock<hecs::World>>,
    );
    fn raw_input(
        &mut self,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
        world: &std::sync::Arc<RwLock<hecs::World>>,
    );
    fn resize(&mut self, width: u32, height: u32, world: &std::sync::Arc<RwLock<hecs::World>>);
    fn display_tick(&mut self, delta_time: Duration, world: &std::sync::Arc<RwLock<hecs::World>>);
    fn physics_tick(&mut self, delta_time: Duration, world: &std::sync::Arc<RwLock<hecs::World>>);
}
