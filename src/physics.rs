use cgmath::Matrix4;

use crate::{character, primitives::Primitive, uniforms};
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

pub fn thread(
    update_period: Duration,
    should_threads_die: Arc<RwLock<bool>>,
    update_scene_channel: mpsc::Receiver<UpdateScene>,
    character_channel: mpsc::Receiver<character::Orientation>,
    update_acceleration_structure: mpsc::Sender<UpdateScene>,
    uniforms: mpsc::Sender<uniforms::Event>,
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
        match character_channel.try_recv() {
            Ok(v) => {
                // TODO: Update character position
                if let Err(e) = uniforms.send(uniforms::Event::UpdateCamera) {
                    log::error!("Failed to send a camera update: {e}");
                    break;
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Character control channel has disconnected");
                break;
            }
        }
        match update_scene_channel.try_recv() {
            Ok(v) => {
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

        // TODO: Call user's function

        physics.tick();
        // TODO: update_acceleration_structure if something changed

        // Only wait if we are running on time
        if let Some(sleep_time) = update_period.checked_sub(last_run_time.elapsed()) {
            std::thread::sleep(sleep_time);
        } else {
            log::warn!("Physics loop is running late, trying to loop every {:?} but actually looping every {:?}", update_period, last_run_time.elapsed());
        }
        last_run_time = Instant::now();
    }
}

struct Physics {}

impl Physics {
    fn new() -> Self {
        Self {}
    }
    fn tick(&mut self) {}
}
