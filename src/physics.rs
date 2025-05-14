use cgmath::{Matrix4, SquareMatrix};

use crate::{
    api::Bloomable,
    primitives::{Primitive, AABB},
    quaternion::Quaternion,
    vec::Vec3,
};
use std::{
    collections::HashMap,
    sync::{mpsc, Arc, RwLock},
    time::{Duration, Instant},
    u64,
};

#[derive(Debug)]
pub enum PhysicsTypes {
    /// Not affected by physics
    None,
    /// Other object can collide with it
    Collidable,
    /// Can collide with collidable objects
    Physical,
}

#[derive(Debug)]
// TODO: Additional messages for modifying the primitive (warping mesh, changing lense values, etc.)
/// Commands that can be given to the ray tracing to modify the objects that will be rendered
pub enum UpdateScene {
    /// Add the provided primitive and give it the provided ID
    Add(u64, Primitive),
    /// ID of the primitive to delete
    Remove(u64),
    /// ID of the new instance, ID of the primitive to instantiate and transformation of the new instance
    AddInstance(u64, u64, Matrix4<f32>),
    /// ID of the instance to remove
    RemoveInstance(u64),
    /// ID of the instance to move and its new transformation matrix
    MoveInstance(u64, Matrix4<f32>),
}

pub enum UpdatePhysics {
    /// Add the provided primitive and give it the provided ID
    Add(u64, Primitive),
    /// ID of the new instance, ID of the primitive to instantiate, transformation of the new instance, and whether it is collidable
    AddInstance(u64, u64, Matrix4<f32>),
    /// ID of the instance to move and its new transformation matrix
    MoveInstance(u64, Matrix4<f32>),
    /// Attach the camera to an instance with a given offset from the center
    AttachCameraToInstance(u64, Vec3, Matrix4<f32>),
    // TODO: Adjust if objects should it should be collidable and if they should be affected by physics
    // Collidable(bool)
    // Physical(bool)
}

pub fn thread<T: Bloomable>(
    update_period: Duration,
    should_threads_die: Arc<RwLock<bool>>,
    update_channel: mpsc::Receiver<UpdatePhysics>,
    camera_pos_in: Arc<RwLock<Vec3>>, // Changes in position
    camera_quat_in: Arc<RwLock<(f32, f32, f32)>>, // Absolute pitch, roll, yaw (in rads)
    camera_pos_out: Arc<RwLock<Vec3>>, // Absolute position
    camera_quat_out: Arc<RwLock<Quaternion>>,
    update_acceleration_structure: mpsc::Sender<UpdateScene>,

    user_app: Arc<RwLock<T>>,
) {
    let mut physics = Physics::new();
    let mut last_run_time = Instant::now();
    let mut last_camera_angles = (0.0, 0.0, 0.0);
    loop {
        // Check if we should end the thread
        match should_threads_die.read() {
            Ok(should_die) => {
                if *should_die == true {
                    return;
                }
            }
            Err(e) => {
                log::error!("rwlock is poisoned, ending thread: {}", e);
                return;
            }
        }
        // Call the user's update function
        match user_app.write() {
            Ok(mut app) => app.physics_tick(last_run_time.elapsed()),
            Err(e) => log::error!("Physics' User App is poisoned: {e}"),
        }
        last_run_time = Instant::now();

        match update_channel.try_recv() {
            Ok(UpdatePhysics::Add(id, primitive)) => {
                physics.add_primitive(id, AABB::new(&primitive));
                if let Err(e) = update_acceleration_structure.send(UpdateScene::Add(id, primitive))
                {
                    log::error!("Failed to propagate add primitive to acceleration structure: {e}");
                    return;
                }
            }
            Ok(UpdatePhysics::AddInstance(instance_id, primitive_id, transformation)) => {
                physics.add_instance(instance_id, primitive_id, transformation);
            }
            Ok(UpdatePhysics::MoveInstance(id, new_transformation)) => {
                physics.move_object(id, new_transformation);
            }
            Ok(UpdatePhysics::AttachCameraToInstance(
                instance_id,
                offset,
                parent_base_transform,
            )) => {
                physics.add_camera(
                    instance_id,
                    offset,
                    Quaternion::identity(),
                    parent_base_transform,
                    0.0,
                    0.0,
                    0.0,
                );
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("User to Physics channel has disconnected");
                return;
            }
        }

        // Pass camera orientation onto the uniforms
        match camera_pos_in.read() {
            Ok(v) => {
                if let Some(new_pos) = physics.update_camera_pos(*v) {
                    match camera_pos_out.write() {
                        Ok(mut v_out) => {
                            *v_out = new_pos;
                        }
                        Err(e) => {
                            log::error!(
                                "Camera position between physics and uniforms is poisoned: {e}"
                            )
                        }
                    }
                }
            }
            Err(e) => {
                log::error!("Camera position between physics and user is poisoned: {e}");
                return;
            }
        }
        match camera_quat_in.read() {
            Ok(v) => {
                if *v != last_camera_angles {
                    last_camera_angles = *v;
                    match camera_quat_out.write() {
                        Ok(mut v_out) => *v_out = physics.update_camera_orientation(v.0, v.1, v.2),
                        Err(e) => {
                            log::error!(
                                "Camera quaternion between physics and uniforms is poisoned: {e}"
                            )
                        }
                    }
                    match camera_pos_out.write() {
                        Ok(mut v_out) => {
                            if let Some(pos) = physics.get_camera_pos() {
                                *v_out = pos;
                            }
                        }
                        Err(e) => {
                            log::error!(
                                "Camera position between physics and uniforms is poisoned: {e}"
                            )
                        }
                    }
                }
            }
            Err(e) => {
                log::error!("Camera quaternion between physics and user is poisoned: {e}")
            }
        }

        // Get any objects that may have moved since the last update
        for moved in physics.get_moved() {
            if let Err(e) = update_acceleration_structure
                .send(UpdateScene::MoveInstance(moved.0, moved.1.transformation))
            {
                log::error!("Failed to propagate physics tick to acceleration structure: {e}");
                return;
            }
        }
        physics.clear_moved();

        // Get any new objects that may have appeared since the last update
        for obj in physics.get_new() {
            if let Err(e) = update_acceleration_structure.send(UpdateScene::AddInstance(
                obj.0,
                obj.1,
                obj.2.transformation,
            )) {
                log::error!("Failed to propagate physics tick to acceleration structure: {e}");
                return;
            }
        }
        physics.clear_new();

        // Only wait if we are running on time
        if let Some(sleep_time) = update_period.checked_sub(last_run_time.elapsed()) {
            std::thread::sleep(sleep_time);
        } else {
            log::warn!("Physics loop is running late, trying to loop every {:?} but actually looping every {:?}", update_period, last_run_time.elapsed());
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct PhysicsObject {
    aabb: AABB,
    transformation: Matrix4<f32>,
}

impl Default for PhysicsObject {
    fn default() -> Self {
        Self {
            aabb: AABB::default(),
            transformation: Matrix4::identity(),
        }
    }
}

struct Camera {
    parent_id: u64,
    pos: Vec3,
    offset: Vec3,
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
    orientation: Quaternion,
    parent_base_transform: Matrix4<f32>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            parent_id: u64::MAX,
            pos: Vec3::zero(),
            offset: Vec3::zero(),
            pitch_rad: 0.0,
            roll_rad: 0.0,
            yaw_rad: 0.0,
            orientation: Quaternion::default(),
            parent_base_transform: Matrix4::identity(),
        }
    }
}

struct Physics {
    primitives: HashMap<u64, AABB>,
    objects: HashMap<u64, PhysicsObject>,
    camera: Camera,
    new_objects: Vec<(u64, u64, PhysicsObject)>,
    moved_objects: Vec<(u64, PhysicsObject)>,
}

impl Physics {
    fn new() -> Self {
        Self {
            objects: HashMap::with_capacity(100),
            camera: Camera::default(),
            primitives: HashMap::with_capacity(100),
            new_objects: Vec::with_capacity(100),
            moved_objects: Vec::with_capacity(100),
        }
    }
    fn add_primitive(&mut self, id: u64, aabb: AABB) {
        self.primitives.insert(id, aabb);
    }
    fn add_instance(&mut self, instance_id: u64, parent_id: u64, transformation: Matrix4<f32>) {
        let aabb = match self.primitives.get(&parent_id) {
            None => {
                log::warn!("Tried to add a collidable object with an invalid parent");
                return;
            }
            Some(&v) => v,
        };
        let obj = PhysicsObject {
            aabb,
            transformation,
        };
        self.objects.insert(instance_id, obj);

        // Queue up this object to be sent to the acceleration structure
        self.new_objects.push((instance_id, parent_id, obj));
    }
    fn add_camera(
        &mut self,
        parent_id: u64,
        offset: Vec3,
        orientation: Quaternion,
        parent_base_transform: Matrix4<f32>,
        pitch_rad: f32,
        roll_rad: f32,
        yaw_rad: f32,
    ) {
        self.camera = Camera {
            parent_id,
            pos: Vec3::zero(),
            offset,
            orientation,
            parent_base_transform,
            pitch_rad,
            roll_rad,
            yaw_rad,
        };
        self.update_camera_parent();
    }
    fn update_camera_orientation(
        &mut self,
        pitch_rad: f32,
        roll_rad: f32,
        yaw_rad: f32,
    ) -> Quaternion {
        self.camera.pitch_rad = pitch_rad;
        self.camera.roll_rad = roll_rad;
        self.camera.yaw_rad = yaw_rad;
        self.camera.orientation =
            Quaternion::from_euler(self.camera.pitch_rad, 0.0, self.camera.yaw_rad);

        self.update_camera_parent();
        self.camera.orientation
    }
    fn update_camera_pos(&mut self, delta: Vec3) -> Option<Vec3> {
        if delta == Vec3::zero() {
            return None;
        }

        let quat = Quaternion::from_euler(0.0, 0.0, self.camera.yaw_rad);
        let new_pos = self.camera.pos + quat.apply(delta);
        if self.check_camera_collision(new_pos) {
            self.camera.pos = new_pos;
            self.update_camera_parent();
        }
        // Compute position of the camera (incl offset from parent) and return it
        self.get_camera_pos()
    }
    /// Update the transformation of the object that the camera is attached to and flag it for being updated in the Acceleration Structure
    fn update_camera_parent(&mut self) {
        // Get the rotation matrix of the object
        let parent_orientation = Quaternion::from_euler(0.0, 0.0, self.camera.yaw_rad);
        let cg_q: cgmath::Quaternion<f32> = parent_orientation.into();
        let rotation: cgmath::Matrix4<f32> = cg_q.into();
        // Update the transformation matrix of the parent object
        match self.objects.get_mut(&self.camera.parent_id) {
            None => log::warn!("Trying to update camera parent without attaching it to an object"),
            Some(v) => {
                v.transformation = Matrix4::from_translation(self.camera.pos.into())
                    * rotation
                    * self.camera.parent_base_transform;
                self.moved_objects.push((self.camera.parent_id, *v));
            }
        }
    }
    fn get_camera_pos(&self) -> Option<Vec3> {
        let quat = Quaternion::from_euler(0.0, 0.0, self.camera.yaw_rad);
        if let Some(v) = self.objects.get(&self.camera.parent_id) {
            Some(
                Vec3::zero().apply_transformation(v.transformation)
                    + quat.apply(self.camera.offset),
            )
        } else {
            None
        }
    }
    /// True if there is no collisions
    fn check_camera_collision(&self, pos: Vec3) -> bool {
        let camera_parent = match self.objects.get(&self.camera.parent_id) {
            None => {
                log::warn!(
                    "Camera maps to an unknown instance {}",
                    self.camera.parent_id
                );
                return true;
            }
            Some(v) => v,
        };
        let mut accept_transformation = true;
        for obj in &self.objects {
            // Ignore collision with itself
            if *obj.0 == self.camera.parent_id {
                continue;
            }
            // If collision, we cannot accept the new transformation. If no collision, try next object
            if camera_parent
                .aabb
                .apply(Matrix4::from_translation(pos.into()) * self.camera.parent_base_transform)
                .collides(&obj.1.aabb.apply(obj.1.transformation))
            {
                accept_transformation = false;
            }
        }
        accept_transformation
    }
    fn move_object(&mut self, id: u64, new_t: Matrix4<f32>) {
        if let Some(o) = self.objects.get_mut(&id) {
            o.transformation = new_t;
            self.moved_objects.push((id, *o));
        } else {
            log::warn!("Tried to move non-existant instance {id}");
        }
    }
    fn get_moved(&self) -> &Vec<(u64, PhysicsObject)> {
        &self.moved_objects
    }
    fn clear_moved(&mut self) {
        self.moved_objects.clear();
    }
    fn get_new(&self) -> &Vec<(u64, u64, PhysicsObject)> {
        &self.new_objects
    }
    fn clear_new(&mut self) {
        self.new_objects.clear();
    }
}
