use cgmath::Quaternion;
use std::{
    sync::{mpsc, Arc, RwLock},
    time::Duration,
};

use crate::vec::Vec3;

#[derive(Debug, Clone)]
pub struct Orientation {
    pub pos: Vec3,
    pub quat: Quaternion<f32>,
    // TODO: Store a shape to represent the character model (AABB? Capsule?)
}
impl Default for Orientation {
    fn default() -> Self {
        Self {
            pos: Vec3::default(),
            quat: Quaternion::new(0.0, 1.0, 0.0, 0.0),
        }
    }
}

pub enum Controls {
    /// Move in x and y
    Move(f32, f32),
    /// Changes about x and y axes in radians(?)
    Look(f32, f32),
}

pub fn thread(
    initial_orientation: Orientation,
    should_threads_die: Arc<RwLock<bool>>,
    control_channel: mpsc::Receiver<Controls>,
    physics_channel: mpsc::Sender<Orientation>,
) {
    let mut character = Character::new(initial_orientation);
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
        match control_channel.recv_timeout(Duration::new(0, 10_000_000)) {
            Ok(Controls::Move(dx, dy)) => {
                if let Err(e) = physics_channel.send(character.translate(dx, dy)) {
                    log::error!("Failed to send character move update: {e}");
                }
            }
            Ok(Controls::Look(dx, dy)) => {
                if let Err(e) = physics_channel.send(character.look(dx, dy)) {
                    log::error!("Failed to send character look update: {e}");
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::error!("Character control channel has disconnected");
                break;
            }
        }
    }
}

/// Kinematic Character Controller
struct Character {
    camera: Orientation,
}

impl Character {
    pub fn new(initial_orientation: Orientation) -> Self {
        Self {
            camera: initial_orientation,
        }
    }
    pub fn translate(&mut self, dx: f32, dy: f32) -> Orientation {
        self.camera.pos.set_x(self.camera.pos.x() + dx);
        self.camera.pos.set_z(self.camera.pos.z() + dy);
        self.camera.clone()
    }
    pub fn look(&mut self, dx: f32, dy: f32) -> Orientation {
        // TODO: do properly
        self.camera.quat.v.x += dx;
        self.camera.quat.v.y += dy;
        self.camera.clone()
    }
}
