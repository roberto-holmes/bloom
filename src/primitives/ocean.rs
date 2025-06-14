use anyhow::anyhow;
use ash::vk;
use hecs::Entity;

use crate::{vec::Vec3, vulkan};

use super::{Addressable, Extrema, ObjectType, Objectionable, PrimitiveAddresses};

#[derive(Debug, PartialEq, Clone)]
pub struct OceanData {
    pub object_type: u64,
    pub material: u32,
    pub wind_speed: f32, // At 10m in m/s
    pub wind_angle: f32,
    pub depth: f32,
    pub leeward_fetch: f32,     // Distance from shore in m
    pub size: u32,              // Number of pixels squared of maps
    pub map: vk::DeviceAddress, // Image containing height and normal maps (maybe something else too?)
}

impl OceanData {
    fn new() -> Self {
        Self {
            object_type: ObjectType::Ocean as _,
            material: 0,
            wind_speed: 3.4,
            wind_angle: 3.4,
            depth: 3.4,
            leeward_fetch: 1000.0,
            size: 0,
            map: 0,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Ocean {
    data: OceanData,
    material: Entity,
    data_buffer: Option<vulkan::Buffer>,
    // map: Option<vulkan::Image<'a>>,
}

impl Ocean {
    pub fn new(material: Entity) -> Self {
        let data = OceanData::new();

        Self {
            data,
            material,
            data_buffer: None,
            // map: None,
        }
    }
}

impl Extrema for Ocean {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        let min = Vec3::all(-1.0);
        let max = Vec3::all(1.0);
        (min, max)
    }
}

impl Objectionable for Ocean {
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
        _command_pool: ash::vk::CommandPool,
        _queue: ash::vk::Queue,
    ) -> anyhow::Result<()> {
        // TODO: Create image
        self.data_buffer = Some(vulkan::Buffer::new_populated(
            allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &self.data,
            1,
        )?);
        Ok(())
    }
}

impl Addressable for Ocean {
    fn free(&mut self) {
        self.data_buffer = None;
    }
    fn get_addresses(&self, device: &ash::Device) -> anyhow::Result<super::PrimitiveAddresses> {
        match &self.data_buffer {
            Some(b) => Ok(PrimitiveAddresses {
                primitive: b.get_device_address(device),
            }),
            None => Err(anyhow!("Ocean does not have a buffer allocated")),
        }
    }
}
