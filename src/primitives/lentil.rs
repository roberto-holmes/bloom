use anyhow::{anyhow, Result};
use ash::vk;
use hecs::Entity;

use crate::{vec::Vec3, vulkan};

use super::{Addressable, Extrema, ObjectType, Objectionable, PrimitiveAddresses};

#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy)]
/// Describe a cylindrical aspherical lens with two lens surface profiles
pub struct LentilData {
    pub object_type: u64,
    // pub center: Vec3,
    pub radius: f32,
    pub length: f32, // Direction and magnitude of longitudinal axis
    pub curvature_a: f32,
    pub curvature_b: f32,
    pub kappa_a: f32,
    pub kappa_b: f32,
    material: u32,
    pub is_selected: u32,
}

impl LentilData {
    fn new(radius: f32, length: f32, is_selected: bool) -> Self {
        Self {
            object_type: ObjectType::Lentil as _,
            radius,
            length,
            curvature_a: 1.5,
            curvature_b: 1.5,
            kappa_a: 1.0,
            kappa_b: 1.0,
            material: 0,
            is_selected: if is_selected { 1 } else { 0 },
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Lentil {
    data: LentilData,
    material: Entity,
    data_buffer: Option<vulkan::Buffer>,
}

impl Lentil {
    pub fn new(radius: f32, length: f32, material: Entity) -> Result<Self> {
        let data = LentilData::new(radius, length, false);
        // TODO: Check that the given values are valid (r is less than r_max)

        Ok(Self {
            data,
            material,
            data_buffer: None,
        })
    }
}

impl Extrema for Lentil {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        // An AABB that can fit the lentil in any orientation
        let min = Vec3::all(-self.data.length) - Vec3::all(self.data.radius);
        let max = Vec3::all(self.data.length) + Vec3::all(self.data.radius);
        // TODO: Account for lenses that stick out
        (min, max)
    }
}

impl Addressable for Lentil {
    fn get_addresses(&self, device: &ash::Device) -> Result<PrimitiveAddresses> {
        match &self.data_buffer {
            Some(b) => Ok(PrimitiveAddresses {
                primitive: b.get_device_address(device),
            }),
            None => Err(anyhow!("Lentil does not have a buffer allocated")),
        }
    }
    fn free(&mut self) {
        self.data_buffer = None;
    }
}

impl Objectionable for Lentil {
    fn set_materials(&mut self, map: &std::collections::HashMap<Entity, usize>) {
        match map.get(&self.material) {
            Some(&m) => self.data.material = m as u32,
            None => self.data.material = 0,
        }
    }
    fn allocate(
        &mut self,
        allocator: &vk_mem::Allocator,
        _device: &ash::Device,
        _command_pool: vk::CommandPool,
        _queue: vk::Queue,
    ) -> Result<()> {
        self.data_buffer = Some(vulkan::Buffer::new_populated(
            allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &self.data,
            1,
            "lentil",
        )?);
        Ok(())
    }
}
