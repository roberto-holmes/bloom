use anyhow::Result;
use ash::vk;

use crate::{vec::Vec3, vulkan};

use super::{Addressable, Extrema, ObjectType, PrimitiveAddresses};

#[repr(C)]
/// Describe a cylindrical aspherical lens with two lens surface profiles
pub struct LentilData {
    pub object_type: u32,
    // pub center: Vec3,
    pub radius: f32,
    pub length: f32, // Direction and magnitude of longitudinal axis
    pub curvature_a: f32,
    pub curvature_b: f32,
    pub kappa_a: f32,
    pub kappa_b: f32,
    _material: u32,
    pub is_selected: u32,
}

impl LentilData {
    fn new(radius: f32, length: f32, material: u32, is_selected: bool) -> Self {
        Self {
            object_type: ObjectType::Lentil as u32,
            radius,
            length,
            curvature_a: 1.0,
            curvature_b: 1.0,
            kappa_a: 1.0,
            kappa_b: 1.0,
            _material: material,
            is_selected: if is_selected { 1 } else { 0 },
        }
    }
}

pub struct Lentil {
    pub transformation: cgmath::Matrix4<f32>,
    data: LentilData,
    data_buffer: vulkan::Buffer,
}

impl Lentil {
    pub fn new(
        allocator: &vk_mem::Allocator,
        radius: f32,
        length: f32,
        transformation: cgmath::Matrix4<f32>,
        material: u32,
    ) -> Result<Self> {
        let data = LentilData::new(radius, length, material, false);
        // TODO: Check that the given values are valid (r is less than r_max)
        let data_buffer = vulkan::Buffer::new_populated(
            allocator,
            size_of::<LentilData>() as u64,
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

impl Extrema for Lentil {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        // let center = Vec3::from(
        //     self.transformation
        //         * cgmath::Vector4 {
        //             x: 0.0,
        //             y: 0.0,
        //             z: 0.0,
        //             w: 1.0,
        //         },
        // );
        // let axial = Vec3::from(
        //     self.transformation
        //         * cgmath::Vector4 {
        //             x: self.data.length,
        //             y: 0.0,
        //             z: 0.0,
        //             w: 1.0,
        //         },
        // );
        let axial = Vec3([self.data.length, 0.0, 0.0]);
        let negative = -axial;
        let min = axial.min_extrema(&negative) - Vec3::all(self.data.radius);
        let max = axial.max_extrema(&negative) + Vec3::all(self.data.radius);
        // TODO: Account for lenses that stick out
        (min, max)
    }
}

impl Addressable for Lentil {
    fn get_addresses(&self, device: &ash::Device) -> PrimitiveAddresses {
        PrimitiveAddresses {
            primitive: self.data_buffer.get_device_address(device),
        }
    }
}
