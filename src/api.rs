use std::{
    collections::HashSet,
    sync::{mpsc, RwLock},
    time::Duration,
};

use anyhow::{anyhow, Result};
use cgmath::Matrix4;
use rand::Rng;

use crate::{
    material,
    physics::{self, UpdatePhysics, UpdateScene},
    primitives::Primitive,
    quaternion::Quaternion,
    vec::Vec3,
};

pub const FOCAL_DISTANCE: f32 = 4.5;
pub const VFOV_DEG: f32 = 40.;
pub const DOF_SCALE: f32 = 0.0;

pub trait Bloomable {
    fn init_window(&mut self, window: std::sync::Arc<RwLock<winit::window::Window>>);
    fn init(&mut self, api: BloomAPI) -> Result<()>;
    fn input(&mut self, event: winit::event::WindowEvent);
    fn resize(&mut self, width: u32, height: u32);
    fn display_tick(&mut self);
    fn physics_tick(&mut self, delta_time: Duration);
}

pub struct BloomAPI {
    add_material: mpsc::Sender<material::Material>,
    update_physics: mpsc::Sender<physics::UpdatePhysics>,

    mat_ids: Vec<u32>,
    obj_ids: HashSet<u64>,
    ins_ids: HashSet<u64>,
}

impl BloomAPI {
    pub fn new(
        add_material: mpsc::Sender<material::Material>,
        update_physics: mpsc::Sender<physics::UpdatePhysics>,
    ) -> Self {
        // let camera = Camera::look_at(
        //     Vec3::new(9., 6., 9.),
        //     Vec3::new(0., 0., 0.),
        //     Vec3::new(0., 1., 0.),
        //     FOCAL_DISTANCE,
        //     VFOV_DEG,
        //     DOF_SCALE,
        // );
        Self {
            add_material,
            update_physics,
            mat_ids: Vec::with_capacity(100),
            obj_ids: HashSet::with_capacity(100),
            ins_ids: HashSet::with_capacity(100),
        }
    }
    // Materials can only be added, not removed
    pub fn add_material(&mut self, material: material::Material) -> Result<u32> {
        let id = self.mat_ids.len() as u32;
        self.mat_ids.push(id);
        self.add_material.send(material)?;
        Ok(id)
    }
    pub fn add_obj(&mut self, obj: Primitive) -> Result<u64> {
        let mut rng = rand::rng();
        let mut id = rng.random();
        while self.obj_ids.contains(&id) {
            // Ensure we create a unique ID
            id = rng.random();
        }
        if let Err(e) = self
            .update_physics
            .send(UpdatePhysics::Scene(UpdateScene::Add(id, obj)))
        {
            return Err(anyhow!("Failed to send new object: {e}"));
        };
        self.obj_ids.insert(id);
        Ok(id)
    }
    pub fn add_instance(&mut self, object_id: u64, transformation: Matrix4<f32>) -> Result<u64> {
        if !self.obj_ids.contains(&object_id) {
            return Err(anyhow!(
                "Tried to add an instance that referred to non-existant object {object_id}"
            ));
        }
        let mut rng = rand::rng();
        let mut id = rng.random();
        while self.ins_ids.contains(&id) {
            // Ensure we create a unique ID
            id = rng.random();
        }
        if let Err(e) = self
            .update_physics
            .send(UpdatePhysics::Scene(UpdateScene::AddInstance(
                id,
                object_id,
                transformation,
            )))
        {
            return Err(anyhow!("Failed to send new instance: {e}"));
        };
        self.ins_ids.insert(id);
        log::debug!("Added an instance");
        Ok(id)
    }
    pub fn update_camera_position(&self, pos: Vec3) {
        if let Err(e) = self
            .update_physics
            .send(UpdatePhysics::Camera(physics::Camera::Position(pos)))
        {
            log::error!("Failed to send new camera position: {e}");
        };
    }
    pub fn update_camera_quaternion(&self, q: Quaternion) {
        if let Err(e) = self
            .update_physics
            .send(UpdatePhysics::Camera(physics::Camera::Quaternion(q)))
        {
            log::error!("Failed to send new camera quaternion: {e}");
        };
    }
}
