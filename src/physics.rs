use cgmath::Matrix4;

use crate::{api::Bloomable, primitives::Primitive, quaternion::Quaternion, vec::Vec3};
use std::{
    sync::{mpsc, Arc, RwLock},
    time::{Duration, Instant},
};

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
    // TODO: Attach the camera to an instance
    // AttachCameraToInstance(u64),
    // RelativePosition(Vec3),
    // TODO: Adjust if objects should it should be collidable and if they should be affected by physics
    // Collidable(bool)
    // Physical(bool)
}

pub enum UpdatePhysics {
    Scene(UpdateScene),
}

pub fn thread<T: Bloomable>(
    update_period: Duration,
    should_threads_die: Arc<RwLock<bool>>,
    update_channel: mpsc::Receiver<UpdatePhysics>,
    camera_pos_in: Arc<RwLock<Vec3>>,
    camera_quat_in: Arc<RwLock<Quaternion>>,
    camera_pos_out: Arc<RwLock<Vec3>>,
    camera_quat_out: Arc<RwLock<Quaternion>>,
    update_acceleration_structure: mpsc::Sender<UpdateScene>,

    user_app: Arc<RwLock<T>>,
) {
    let mut physics = Physics::new();
    let mut last_run_time = Instant::now();
    loop {
        // Check if we should end the thread
        match should_threads_die.read() {
            Ok(should_die) => {
                if *should_die == true {
                    break;
                }
            }
            Err(e) => {
                log::error!("rwlock is poisoned, ending thread: {}", e)
            }
        }
        // Call the user's update function
        match user_app.write() {
            Ok(mut app) => app.physics_tick(last_run_time.elapsed()),
            Err(e) => log::error!("Physics' User App is poisoned: {e}"),
        }
        last_run_time = Instant::now();

        match update_channel.try_recv() {
            Ok(UpdatePhysics::Scene(v)) => {
                // TODO: Add these into the physics object and do calculations on their positions
                // log::debug!(
                //     "Physics received an update scene request containing: {:?}",
                //     v
                // );
                if let Err(e) = update_acceleration_structure.send(v) {
                    log::error!("Failed to send an acceleration structure update: {e}");
                    break;
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Character control channel has disconnected");
                break;
            }
        }

        // Pass camera orientation onto the uniforms
        // TODO: Incorporate into collision detection
        match camera_pos_in.read() {
            Ok(v) => match camera_pos_out.write() {
                Ok(mut v_out) => *v_out = *v,
                Err(e) => {
                    log::error!("Camera position between physics and uniforms is poisoned: {e}")
                }
            },
            Err(e) => {
                log::error!("Camera position between physics and user is poisoned: {e}")
            }
        }
        match camera_quat_in.read() {
            Ok(v) => match camera_quat_out.write() {
                Ok(mut v_out) => *v_out = *v,
                Err(e) => {
                    log::error!("Camera quaterninon between physics and uniforms is poisoned: {e}")
                }
            },
            Err(e) => {
                log::error!("Camera quaterninon between physics and user is poisoned: {e}")
            }
        }

        physics.tick();
        // TODO: update_acceleration_structure if something changed

        // Only wait if we are running on time
        if let Some(sleep_time) = update_period.checked_sub(last_run_time.elapsed()) {
            std::thread::sleep(sleep_time);
        } else {
            log::warn!("Physics loop is running late, trying to loop every {:?} but actually looping every {:?}", update_period, last_run_time.elapsed());
        }
    }
}

struct Physics {}

impl Physics {
    fn new() -> Self {
        Self {}
    }
    fn tick(&mut self) {}
}
