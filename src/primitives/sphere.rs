use anyhow::Result;
use ash::vk;

use crate::{vec::Vec3, vulkan};

use super::{Addressable, Extrema, ObjectType, PrimitiveAddresses};

#[repr(C)]
pub struct SphereData {
    pub object_type: u32,
    pub radius: f32,
    _material: u32,
    pub is_selected: u32,
}

impl SphereData {
    fn new(radius: f32, material: u32, is_selected: bool) -> Self {
        Self {
            object_type: ObjectType::Sphere as u32,
            radius,
            _material: material,
            is_selected: if is_selected { 1 } else { 0 },
        }
    }
}

pub struct Sphere {
    pub transformation: cgmath::Matrix4<f32>,
    data: SphereData,
    data_buffer: vulkan::Buffer,
}

impl Sphere {
    pub fn new(
        allocator: &vk_mem::Allocator,
        radius: f32,
        transformation: cgmath::Matrix4<f32>,
        material: u32,
    ) -> Result<Self> {
        let data = SphereData::new(radius, material, false);

        let data_buffer = vulkan::Buffer::new_populated(
            allocator,
            size_of::<SphereData>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &data,
            1,
        )?;

        Ok(Self {
            transformation,
            data,
            data_buffer,
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

impl Addressable for Sphere {
    fn get_addresses(&self, device: &ash::Device) -> PrimitiveAddresses {
        PrimitiveAddresses {
            primitive: self.data_buffer.get_device_address(device),
        }
    }
}
