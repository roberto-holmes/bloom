use cgmath::Matrix4;

use crate::{api::Bloomable, primitives::Primitive, quaternion::Quaternion, uniforms, vec::Vec3};
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
}

pub enum Camera {
    /// Change the position of the camera
    Position(Vec3),
    /// Change the rotation of the camera
    Quaternion(Quaternion),
    // TODO: Describe the shape of the camera for physics purposes and if it should be collidable
    // Shape,
    // RelativePosition(Vec3),
    // Collidable(bool)
}

pub enum UpdatePhysics {
    Scene(UpdateScene),
    Camera(Camera),
}

pub fn thread<T: Bloomable>(
    update_period: Duration,
    should_threads_die: Arc<RwLock<bool>>,
    update_channel: mpsc::Receiver<UpdatePhysics>,
    update_acceleration_structure: mpsc::Sender<UpdateScene>,
    uniforms: mpsc::Sender<uniforms::Event>,

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
            Ok(UpdatePhysics::Camera(Camera::Position(pos))) => {
                // TODO: Collision detection
                if let Err(e) = uniforms.send(uniforms::Event::UpdateCameraPosition(pos)) {
                    log::error!("Failed to send new camera position: {e}");
                    break;
                }
            }
            Ok(UpdatePhysics::Camera(Camera::Quaternion(q))) => {
                // TODO: Collision detection
                if let Err(e) = uniforms.send(uniforms::Event::UpdateCameraQuaternion(q)) {
                    log::error!("Failed to send new camera quaternion: {e}");
                    break;
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Character control channel has disconnected");
                break;
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
