pub mod lentil;
pub mod model;
pub mod sphere;

use ash::vk;

#[cfg(not(target_arch = "wasm32"))]
use crate::vec::Vec3;

pub trait Extrema {
    fn get_extrema(&self) -> (Vec3, Vec3);
}

#[derive(Debug)]
pub struct PrimitiveAddresses {
    pub primitive: vk::DeviceAddress,
}

pub trait Addressable {
    fn get_addresses(&self, device: &ash::Device) -> PrimitiveAddresses;
}

#[derive(Clone, Copy, PartialEq)]
pub enum ObjectType {
    Triangle = 0,
    Sphere = 1,
    Lentil = 2,
}

#[derive(Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new<T: Extrema>(obj: &T) -> Self {
        Self {
            min: obj.get_extrema().0,
            max: obj.get_extrema().1,
        }
    }
}

pub struct Scene {
    pub models: Vec<model::Model>,
    pub spheres: Vec<sphere::Sphere>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            models: vec![],
            spheres: vec![],
        }
    }
}
