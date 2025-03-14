use std::{
    collections::HashMap,
    sync::{Mutex, Weak},
};

use anyhow::{anyhow, Result};
use cgmath::Matrix4;
use rand::Rng;
use winit::event::WindowEvent;

use crate::{
    camera::Camera,
    core::{self, UniformBufferObject},
    material,
    primitives::Primitive,
    vec::Vec3,
};

pub const FOCAL_DISTANCE: f32 = 4.5;
pub const VFOV_DEG: f32 = 40.;
pub const DOF_SCALE: f32 = 0.0;

pub trait Bloomable {
    fn init(&mut self, api: Weak<Mutex<BloomAPI>>);
    fn input(&mut self, event: WindowEvent);
    fn resize(&mut self, width: u32, height: u32);
    // fn update(&mut self);
    // fn fixed_update(&mut self);
}

pub struct BloomAPI {
    // Needs to include a way to modify all of the buffers
    pub scene: Scene,
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
            scene: Scene::default(),
            camera,
            uniform: UniformBufferObject::new(),
        }
    }
    pub(crate) fn update_camera(&mut self) {
        self.uniform.update_camera(&self.camera);
    }
}

pub struct Scene {
    // Camera position and settings
    camera: Camera,
    // Vector of materials
    pub(crate) materials: Vec<material::Material>,
    // Collection containing each object in the scene
    pub(crate) primitives: HashMap<u64, Primitive>,
    pub(crate) instances: HashMap<u64, (u64, Matrix4<f32>)>,
}

impl Scene {
    pub fn add_material(&mut self, material: material::Material) -> u32 {
        let id = self.materials.len();
        self.materials.push(material);
        id as u32
    }
    pub fn add_materials(&mut self, materials: &[material::Material]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(materials.len());
        for m in materials {
            let id = self.materials.len() as u32;
            self.materials.push(*m);
            ids.push(id);
        }
        ids
    }
    pub fn add_obj(&mut self, obj: Primitive) -> u64 {
        let mut rng = rand::rng();
        let mut id = rng.random();
        while self.primitives.contains_key(&id) {
            // Ensure we create a unique ID
            id = rng.random();
        }
        self.primitives.insert(id, obj);
        id
    }
    pub fn add_objs(&mut self, objs: Vec<Primitive>) -> Vec<u64> {
        let mut ids = Vec::with_capacity(objs.len());
        self.primitives.reserve(objs.len());
        for o in objs {
            ids.push(self.add_obj(o));
        }
        ids
    }
    pub fn remove_obj(&mut self, id: u64) {
        self.primitives.remove(&id);
    }
    pub fn remove_batch_obj(&mut self, ids: &[u64]) {
        for id in ids {
            self.remove_obj(*id);
        }
    }
    pub fn add_instance(&mut self, object_id: u64, transformation: Matrix4<f32>) -> Result<u64> {
        if !self.primitives.contains_key(&object_id) {
            return Err(anyhow!(
                "Tried to add an instance that referred to non-existant object {object_id}"
            ));
        }
        let mut rng = rand::rng();
        let mut id = rng.random();
        while self.instances.contains_key(&id) {
            // Ensure we create a unique ID
            id = rng.random();
        }
        self.instances.insert(id, (object_id, transformation));
        Ok(id)
    }
    pub fn add_instances(
        &mut self,
        object_id: u64,
        transformations: &[Matrix4<f32>],
    ) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(transformations.len());
        self.instances.reserve(transformations.len());
        for t in transformations {
            ids.push(self.add_instance(object_id, t.clone())?);
        }
        Ok(ids)
    }
    pub fn remove_instance(&mut self, id: u64) {
        self.primitives.remove(&id);
    }
    pub fn remove_instances(&mut self, ids: &[u64]) {
        for id in ids {
            self.remove_instance(*id);
        }
    }
    pub fn get_transformation(&self, id: u64) -> Result<Matrix4<f32>> {
        match self.instances.get(&id) {
            Some(v) => Ok(v.1),
            None => Err(anyhow!("No object exists with id {id}")),
        }
    }
    pub fn set_transformation(&mut self, id: u64, transformation: Matrix4<f32>) -> Result<()> {
        // TODO: Figure out how to trigger an Acceleration Structure update
        match self.instances.get_mut(&id) {
            Some(v) => Ok(v.1 = transformation),
            None => Err(anyhow!("No object exists with id {id}")),
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            materials: vec![],
            primitives: HashMap::new(),
            instances: HashMap::new(),
        }
    }
}
