use std::{
    collections::HashSet,
    sync::{mpsc, Arc, RwLock},
    time::Duration,
};

use anyhow::{anyhow, Result};
use cgmath::Matrix4;
use rand::Rng;

use crate::{
    material,
    physics::{self, UpdatePhysics},
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
    fn raw_input(&mut self, device_id: winit::event::DeviceId, event: winit::event::DeviceEvent);
    fn resize(&mut self, width: u32, height: u32);
    fn display_tick(&mut self);
    fn physics_tick(&mut self, delta_time: Duration);
}

pub struct BloomAPI {
    add_material: mpsc::Sender<material::Material>,
    update_physics: mpsc::Sender<physics::UpdatePhysics>,
    update_camera_pos: Arc<RwLock<Vec3>>,
    update_camera_angles: Arc<RwLock<(f32, f32, f32)>>,

    mat_ids: Vec<u32>,
    obj_ids: HashSet<u64>,
    ins_ids: HashSet<u64>,
}

impl BloomAPI {
    pub fn new(
        add_material: mpsc::Sender<material::Material>,
        update_physics: mpsc::Sender<physics::UpdatePhysics>,
        update_camera_pos: Arc<RwLock<Vec3>>,
        update_camera_angles: Arc<RwLock<(f32, f32, f32)>>,
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
            update_camera_pos,
            update_camera_angles,
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
        if let Err(e) = self.update_physics.send(UpdatePhysics::Add(id, obj)) {
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
        if let Err(e) =
            self.update_physics
                .send(UpdatePhysics::AddInstance(id, object_id, transformation))
        {
            return Err(anyhow!("Failed to send new instance: {e}"));
        };
        self.ins_ids.insert(id);
        // log::debug!("Added an instance");
        Ok(id)
    }
    pub fn move_instance_to(&mut self, id: u64, transformation: Matrix4<f32>) -> Result<()> {
        if !self.ins_ids.contains(&id) {
            return Err(anyhow!("Tried to move an instance that doesn't exist {id}"));
        }
        if let Err(e) = self
            .update_physics
            .send(UpdatePhysics::MoveInstance(id, transformation))
        {
            return Err(anyhow!("Failed to send new instance: {e}"));
        };
        Ok(())
    }
    pub fn assign_camera_to(
        &mut self,
        id: u64,
        offset: Vec3,
        parent_base_transform: Matrix4<f32>,
    ) -> Result<()> {
        if let Err(e) = self
            .update_physics
            .send(UpdatePhysics::AttachCameraToInstance(
                id,
                offset,
                parent_base_transform,
            ))
        {
            return Err(anyhow!("Failed to send new instance: {e}"));
        };
        Ok(())
    }
    pub fn update_camera_position(&self, pos: Vec3) {
        match self.update_camera_pos.write() {
            Ok(mut v) => *v = pos,
            Err(e) => log::error!("Position is poisoned: {e}"),
        }
    }
    pub fn update_camera_angles(&self, pitch_rad: f32, roll_rad: f32, yaw_rad: f32) {
        match self.update_camera_angles.write() {
            Ok(mut v) => *v = (pitch_rad, roll_rad, yaw_rad),
            Err(e) => log::error!("Quaternion is poisoned: {e}"),
        }
    }
}
//     pub fn update_camera_quaternion(&self, q: Quaternion) {
//         match self.update_camera_quat.write() {
//             Ok(mut v) => *v = q,
//             Err(e) => log::error!("Quaternion is poisoned: {e}"),
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use crate::{material::Material, primitives::lentil::Lentil};

    use super::*;

    #[test]
    fn add_material() -> Result<()> {
        let test_material_count = 1024;
        let test_material = Material::new_basic(Vec3::unit_x(), 0.75);

        let (add_material_sender, add_material_receiver) = mpsc::channel();
        let (update_scene_sender, _) = mpsc::channel();
        let cam_pos = Arc::new(RwLock::new(Vec3::zero()));
        let cam_angles = Arc::new(RwLock::new((0.0, 0.0, 0.0)));

        let mut api = BloomAPI::new(
            add_material_sender,
            update_scene_sender,
            cam_pos,
            cam_angles,
        );

        for _ in 0..test_material_count {
            api.add_material(test_material)?;
        }
        let mut material_count = 0;

        loop {
            match add_material_receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(m) => {
                    assert_eq!(m, test_material);
                    material_count += 1;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("Material channel disconnected")
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
            };
        }

        // Check that we received all the materials
        assert_eq!(test_material_count, material_count);
        // Check that all the materials were saved
        assert_eq!(test_material_count, api.mat_ids.len());
        Ok(())
    }

    #[test]
    fn add_obj() -> Result<()> {
        let test_obj_count = 1 << 10;
        let test_obj = Primitive::Lentil(Lentil::new(0.2, 2.3, 4)?);

        let (add_material_sender, _) = mpsc::channel();
        let (update_scene_sender, update_scene_receiver) = mpsc::channel();
        let cam_pos = Arc::new(RwLock::new(Vec3::zero()));
        let cam_angles = Arc::new(RwLock::new((0.0, 0.0, 0.0)));

        let mut api = BloomAPI::new(
            add_material_sender,
            update_scene_sender,
            cam_pos,
            cam_angles,
        );

        for _ in 0..test_obj_count {
            api.add_obj(test_obj.clone())?;
        }
        let mut obj_count = 0;

        let mut ids = HashSet::with_capacity(test_obj_count);

        loop {
            match update_scene_receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(physics::UpdatePhysics::Add(id, obj)) => {
                    // Ensure all the IDs are unique
                    assert!(!ids.contains(&id));
                    ids.insert(id);
                    // Ensure the object hasn't been mangled somehow
                    assert_eq!(obj, test_obj);
                    obj_count += 1;
                }
                Ok(_) => panic!("Wrong data received"),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("Material channel disconnected")
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
            };
        }

        assert_eq!(test_obj_count, obj_count);
        // Make sure we have a unique ID for every object
        assert_eq!(ids.len(), obj_count);
        Ok(())
    }
}
