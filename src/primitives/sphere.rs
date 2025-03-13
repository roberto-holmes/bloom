use anyhow::{anyhow, Result};
use ash::vk;

use crate::{vec::Vec3, vulkan};

use super::{Addressable, Extrema, ObjectType, Objectionable, PrimitiveAddresses};

#[repr(C)]
pub struct SphereData {
    pub object_type: u64,
    pub radius: f32,
    _material: u32,
    pub is_selected: u32,
}

impl SphereData {
    fn new(radius: f32, material: u32, is_selected: bool) -> Self {
        Self {
            object_type: ObjectType::Sphere as _,
            radius,
            _material: material,
            is_selected: if is_selected { 1 } else { 0 },
        }
    }
}

pub struct Sphere {
    data: SphereData,
    data_buffer: Option<vulkan::Buffer>,
}

impl Sphere {
    pub fn new(radius: f32, material: u32) -> Result<Self> {
        let data = SphereData::new(radius, material, false);

        Ok(Self {
            data,
            data_buffer: None,
        })
    }
}

impl Extrema for Sphere {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        let min = -Vec3::all(self.data.radius);
        let max = Vec3::all(self.data.radius);
        (min, max)
    }
}

impl Objectionable for Sphere {
    fn allocate(&mut self, allocator: &vk_mem::Allocator, _device: &ash::Device) -> Result<()> {
        self.data_buffer = Some(vulkan::Buffer::new_populated(
            allocator,
            size_of::<SphereData>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &self.data,
            1,
        )?);
        Ok(())
    }
}

impl Addressable for Sphere {
    fn get_addresses(&self, device: &ash::Device) -> Result<PrimitiveAddresses> {
        match &self.data_buffer {
            Some(b) => Ok(PrimitiveAddresses {
                primitive: b.get_device_address(device),
            }),
            None => Err(anyhow!("Sphere does not have a buffer allocated")),
        }
    }
    fn free(&mut self) {
        self.data_buffer = None;
    }
}
