use anyhow::anyhow;
use ash::vk;
use hecs::Entity;

use crate::{oceans::OCEAN_RESOLUTION, vec::Vec3, vulkan};

use super::{Addressable, Extrema, ObjectType, Objectionable, PrimitiveAddresses};

#[derive(Debug, PartialEq, Clone)]
pub struct HitData {
    pub object_type: u64,
    pub material: u32,
    pub size: u32, // Number of pixels squared of maps
}

impl HitData {
    fn new() -> Self {
        Self {
            object_type: ObjectType::Ocean as _,
            material: 0,
            size: OCEAN_RESOLUTION,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct GenData {
    pub wind_speed: f32, // At 10m in m/s
    pub wind_angle: f32,
    pub leeward_fetch: f32, // Distance from shore in m
    pub depth: f32,
    pub length_scale: f32,
    pub(crate) size: u32, // Number of pixels squared of maps
}

impl GenData {
    fn new(
        wind_speed: f32,
        wind_angle: f32,
        leeward_fetch: f32,
        depth: f32,
        length_scale: f32,
    ) -> Self {
        Self {
            wind_speed,
            wind_angle,
            leeward_fetch,
            depth,
            length_scale,
            size: OCEAN_RESOLUTION,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Ocean {
    hit_data: HitData,
    pub params: GenData,
    material: Entity,
    data_buffer: Option<vulkan::Buffer>,
}

impl Ocean {
    pub fn new(
        material: Entity,
        wind_speed: f32,
        wind_angle: f32,
        leeward_fetch: f32,
        depth: f32,
        length_scale: f32,
    ) -> Self {
        let hit_data = HitData::new();
        let params = GenData::new(wind_speed, wind_angle, leeward_fetch, depth, length_scale);

        Self {
            hit_data,
            params,
            material,
            data_buffer: None,
        }
    }
}

impl Extrema for Ocean {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        // TODO: Figure out how we are going to describe the dimensions of the ocean
        let min = Vec3::new(-10.0, -1.0, -10.0);
        let max = Vec3::new(10.0, 1.0, 10.0);
        // let min = Vec3::all(-1.0);
        // let max = Vec3::all(1.0);
        (min, max)
    }
}

impl Objectionable for Ocean {
    fn set_materials(&mut self, map: &std::collections::HashMap<Entity, usize>) {
        match map.get(&self.material) {
            Some(&m) => self.hit_data.material = m as u32,
            None => self.hit_data.material = 0,
        }
    }
    fn allocate(
        &mut self,
        allocator: &vk_mem::Allocator,
        _device: &ash::Device,
        _command_pool: ash::vk::CommandPool,
        _queue: ash::vk::Queue,
    ) -> anyhow::Result<()> {
        self.data_buffer = Some(vulkan::Buffer::new_populated(
            allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &self.hit_data,
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
