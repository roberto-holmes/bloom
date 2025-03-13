pub mod lentil;
pub mod model;
pub mod sphere;

use anyhow::Result;
use ash::vk;

use crate::vec::Vec3;

pub enum Primitive {
    Model(model::Model),
    Sphere(sphere::Sphere),
    Lentil(lentil::Lentil),
}

impl Addressable for Primitive {
    fn get_addresses(&self, device: &ash::Device) -> Result<PrimitiveAddresses> {
        match self {
            Primitive::Model(v) => v.get_addresses(device),
            Primitive::Sphere(v) => v.get_addresses(device),
            Primitive::Lentil(v) => v.get_addresses(device),
        }
    }
    fn free(&mut self) {
        match self {
            Primitive::Model(v) => v.free(),
            Primitive::Sphere(v) => v.free(),
            Primitive::Lentil(v) => v.free(),
        };
    }
}

pub trait Extrema {
    fn get_extrema(&self) -> (Vec3, Vec3);
}

#[derive(Debug)]
pub struct PrimitiveAddresses {
    pub primitive: vk::DeviceAddress,
}

pub trait Addressable {
    fn get_addresses(&self, device: &ash::Device) -> Result<PrimitiveAddresses>;
    fn free(&mut self);
}

pub(crate) trait Objectionable {
    fn allocate(&mut self, allocator: &vk_mem::Allocator, device: &ash::Device) -> Result<()>;
}

#[repr(u64)]
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
