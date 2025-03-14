use anyhow::{anyhow, Result};
use ash::vk;

use crate::{vec::Vec3, vulkan};

use super::{Addressable, ObjectType, Objectionable, PrimitiveAddresses};

#[repr(C)]
#[derive(Debug)]
pub struct MeshData {
    pub object_type: u64,
    pub vertices: vk::DeviceAddress,
    pub indices: vk::DeviceAddress,
    pub material_indices: vk::DeviceAddress,
}

#[derive(Debug)]
pub struct Vertex {
    _pos: Vec3,
    _pad1: u32,
    _normal: Vec3,
    _pad2: u32,
}

impl Vertex {
    fn new(pos: Vec3, normal: Vec3) -> Self {
        Self {
            _pos: pos,
            _pad1: 0,
            _normal: normal,
            _pad2: 0,
        }
    }
}

#[derive(Debug)]
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_indices: Vec<u32>,
    pub vertex_buffer: Option<vulkan::Buffer>, // Buffer containing all the vertices that make up the object
    pub index_buffer: Option<vulkan::Buffer>,  // How to connect up the vertices into triangles
    pub mat_index_buffer: Option<vulkan::Buffer>, // What material each triangle maps to
    pub primitive_data: Option<MeshData>,
    main_buffer: Option<vulkan::Buffer>,
}

impl Model {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, material_indices: Vec<u32>) -> Self {
        Self {
            vertices,
            indices,
            material_indices,
            vertex_buffer: None,
            index_buffer: None,
            mat_index_buffer: None,
            primitive_data: None,
            main_buffer: None,
        }
    }
    pub fn new_cube(material: u32) -> Result<Self> {
        #[rustfmt::skip]
        let vertices = vec![
            Vertex::new(Vec3([ 0.5,  0.5,  0.5]), Vec3([ 0.0,  1.0,  0.0])),    // Top
            Vertex::new(Vec3([-0.5,  0.5,  0.5]), Vec3([ 0.0,  1.0,  0.0])),
            Vertex::new(Vec3([ 0.5,  0.5, -0.5]), Vec3([ 0.0,  1.0,  0.0])),
            Vertex::new(Vec3([-0.5,  0.5, -0.5]), Vec3([ 0.0,  1.0,  0.0])),
            Vertex::new(Vec3([ 0.5, -0.5,  0.5]), Vec3([ 0.0, -1.0,  0.0])),    // Bottom
            Vertex::new(Vec3([-0.5, -0.5,  0.5]), Vec3([ 0.0, -1.0,  0.0])),
            Vertex::new(Vec3([ 0.5, -0.5, -0.5]), Vec3([ 0.0, -1.0,  0.0])),
            Vertex::new(Vec3([-0.5, -0.5, -0.5]), Vec3([ 0.0, -1.0,  0.0])),
            Vertex::new(Vec3([ 0.5,  0.5,  0.5]), Vec3([ 1.0,  0.0,  0.0])),    // Right
            Vertex::new(Vec3([ 0.5,  0.5, -0.5]), Vec3([ 1.0,  0.0,  0.0])),
            Vertex::new(Vec3([ 0.5, -0.5, -0.5]), Vec3([ 1.0,  0.0,  0.0])),
            Vertex::new(Vec3([ 0.5, -0.5,  0.5]), Vec3([ 1.0,  0.0,  0.0])),
            Vertex::new(Vec3([-0.5,  0.5,  0.5]), Vec3([-1.0,  0.0,  0.0])),    // Left
            Vertex::new(Vec3([-0.5,  0.5, -0.5]), Vec3([-1.0,  0.0,  0.0])),
            Vertex::new(Vec3([-0.5, -0.5, -0.5]), Vec3([-1.0,  0.0,  0.0])),
            Vertex::new(Vec3([-0.5, -0.5,  0.5]), Vec3([-1.0,  0.0,  0.0])),
            Vertex::new(Vec3([-0.5,  0.5,  0.5]), Vec3([ 0.0,  0.0,  1.0])),   // Front
            Vertex::new(Vec3([ 0.5,  0.5,  0.5]), Vec3([ 0.0,  0.0,  1.0])),
            Vertex::new(Vec3([ 0.5, -0.5,  0.5]), Vec3([ 0.0,  0.0,  1.0])),
            Vertex::new(Vec3([-0.5, -0.5,  0.5]), Vec3([ 0.0,  0.0,  1.0])),
            Vertex::new(Vec3([-0.5,  0.5, -0.5]), Vec3([ 0.0,  0.0, -1.0])),   // Back
            Vertex::new(Vec3([ 0.5,  0.5, -0.5]), Vec3([ 0.0,  0.0, -1.0])),
            Vertex::new(Vec3([ 0.5, -0.5, -0.5]), Vec3([ 0.0,  0.0, -1.0])),
            Vertex::new(Vec3([-0.5, -0.5, -0.5]), Vec3([ 0.0,  0.0, -1.0])),
            ];
        #[rustfmt::skip]
        let indices= vec![
                0,  1,  2,  1,  2,  3, // Top
                4,  5,  6,  5,  6,  7, // Bottom
                8,  9, 10,  8, 10, 11, // Right
            12, 13, 14, 12, 14, 15, // Left
            16, 17, 18, 16, 18, 19, // Front
            20, 21, 22, 20, 22, 23, // Back
            ];
        let material_indices = vec![material; 12];

        Ok(Self {
            vertices,
            indices,
            material_indices,
            primitive_data: None,
            main_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
            mat_index_buffer: None,
        })
    }
    pub fn new_mirror(material: u32) -> Result<Self> {
        #[rustfmt::skip]
        let vertices = vec![
            Vertex::new(Vec3([ 0.5,  0.5,  0.5]), Vec3([ 0.0,  1.0,  0.0])),    // Top
            Vertex::new(Vec3([-0.5,  0.5,  0.5]), Vec3([ 0.0,  1.0,  0.0])),
            Vertex::new(Vec3([ 0.5,  0.5, -0.5]), Vec3([ 0.0,  1.0,  0.0])),
            Vertex::new(Vec3([-0.5,  0.5, -0.5]), Vec3([ 0.0,  1.0,  0.0])),
            Vertex::new(Vec3([ 0.5, -0.5,  0.5]), Vec3([ 0.0, -1.0,  0.0])),    // Bottom
            Vertex::new(Vec3([-0.5, -0.5,  0.5]), Vec3([ 0.0, -1.0,  0.0])),
            Vertex::new(Vec3([ 0.5, -0.5, -0.5]), Vec3([ 0.0, -1.0,  0.0])),
            Vertex::new(Vec3([-0.5, -0.5, -0.5]), Vec3([ 0.0, -1.0,  0.0])),
            Vertex::new(Vec3([ 0.5,  0.5,  0.5]), Vec3([ 1.0,  0.0,  0.0])),    // Right
            Vertex::new(Vec3([ 0.5,  0.5, -0.5]), Vec3([ 1.0,  0.0,  0.0])),
            Vertex::new(Vec3([ 0.5, -0.5, -0.5]), Vec3([ 1.0,  0.0,  0.0])),
            Vertex::new(Vec3([ 0.5, -0.5,  0.5]), Vec3([ 1.0,  0.0,  0.0])),
            Vertex::new(Vec3([-0.5,  0.5,  0.5]), Vec3([-1.0,  0.0,  0.0])),    // Left
            Vertex::new(Vec3([-0.5,  0.5, -0.5]), Vec3([-1.0,  0.0,  0.0])),
            Vertex::new(Vec3([-0.5, -0.5, -0.5]), Vec3([-1.0,  0.0,  0.0])),
            Vertex::new(Vec3([-0.5, -0.5,  0.5]), Vec3([-1.0,  0.0,  0.0])),
            Vertex::new(Vec3([-0.5,  0.5,  0.5]), Vec3([ 0.0,  0.0,  1.0])),   // Front
            Vertex::new(Vec3([ 0.5,  0.5,  0.5]), Vec3([ 0.0,  0.0,  1.0])),
            Vertex::new(Vec3([ 0.5, -0.5,  0.5]), Vec3([ 0.0,  0.0,  1.0])),
            Vertex::new(Vec3([-0.5, -0.5,  0.5]), Vec3([ 0.0,  0.0,  1.0])),
            Vertex::new(Vec3([-0.5,  0.5, -0.5]), Vec3([ 0.0,  0.0, -1.0])),   // Back
            Vertex::new(Vec3([ 0.5,  0.5, -0.5]), Vec3([ 0.0,  0.0, -1.0])),
            Vertex::new(Vec3([ 0.5, -0.5, -0.5]), Vec3([ 0.0,  0.0, -1.0])),
            Vertex::new(Vec3([-0.5, -0.5, -0.5]), Vec3([ 0.0,  0.0, -1.0])),
            ];
        #[rustfmt::skip]
        let indices= vec![
                0,  1,  2,  1,  2,  3, // Top
                4,  5,  6,  5,  6,  7, // Bottom
                8,  9, 10,  8, 10, 11, // Right
            12, 13, 14, 12, 14, 15, // Left
            16, 17, 18, 16, 18, 19, // Front
            20, 21, 22, 20, 22, 23, // Back
            ];
        let material_indices = vec![material; 12];

        Ok(Self {
            vertices,
            indices,
            material_indices,
            primitive_data: None,
            main_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
            mat_index_buffer: None,
        })
    }
    pub fn new_plane(material: u32) -> Result<Self> {
        #[rustfmt::skip]
       let vertices= vec![
            Vertex::new(Vec3([ 1.0, 0.0,  1.0]), Vec3([0.0, 1.0, 0.0])),
            Vertex::new(Vec3([-1.0, 0.0,  1.0]), Vec3([0.0, 1.0, 0.0])),
            Vertex::new(Vec3([ 1.0, 0.0, -1.0]), Vec3([0.0, 1.0, 0.0])),
            Vertex::new(Vec3([-1.0, 0.0, -1.0]), Vec3([0.0, 1.0, 0.0])),
            ];
        #[rustfmt::skip]
       let indices= vec![
             0,  1,  2,  1,  2,  3,
        ];
        let material_indices = vec![material; 2];

        Ok(Self {
            vertices,
            indices,
            material_indices,
            primitive_data: None,
            main_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
            mat_index_buffer: None,
        })
    }
    pub fn get_device_address(&self, device: &ash::Device) -> Result<vk::DeviceAddress> {
        match &self.main_buffer {
            Some(v) => Ok(v.get_device_address(device)),
            None => Err(anyhow!(
                "Tried to get address of a model that hadn't been allocated"
            )),
        }
    }
}

impl Objectionable for Model {
    fn allocate(&mut self, allocator: &vk_mem::Allocator, device: &ash::Device) -> Result<()> {
        let vertex_buffer_size = self.vertices.len() * size_of::<Vertex>();
        let index_buffer_size = self.indices.len() * size_of::<u32>();
        let mat_index_buffer_size = self.material_indices.len() * size_of::<i32>();

        // let max_material_index = 1;
        // TODO: Make sure material indices never exceed the number of available materials?
        // for index in &mut material_indices {
        //     *index = std::cmp::min(max_material_index, *index);
        // }

        // TODO: Add a staging buffer so these buffers can live on the GPU
        self.vertex_buffer = Some(vulkan::Buffer::new_populated(
            allocator,
            vertex_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            self.vertices.as_ptr(),
            self.vertices.len(),
        )?);
        self.index_buffer = Some(vulkan::Buffer::new_populated(
            allocator,
            index_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            self.indices.as_ptr(),
            self.indices.len(),
        )?);
        // log::debug!("Cube materials: {:?}", self.material_indices);
        self.mat_index_buffer = Some(vulkan::Buffer::new_populated(
            allocator,
            mat_index_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            self.material_indices.as_ptr(),
            self.material_indices.len(),
        )?);

        self.primitive_data = Some(MeshData {
            object_type: ObjectType::Triangle as _,
            vertices: self
                .vertex_buffer
                .as_ref()
                .unwrap()
                .get_device_address(device),
            indices: self
                .index_buffer
                .as_ref()
                .unwrap()
                .get_device_address(device),
            material_indices: self
                .mat_index_buffer
                .as_ref()
                .unwrap()
                .get_device_address(device),
        });

        self.main_buffer = Some(vulkan::Buffer::new_populated(
            allocator,
            size_of::<MeshData>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            self.primitive_data.as_ref().unwrap(),
            1,
        )?);

        Ok(())
    }
}

impl Addressable for Model {
    fn get_addresses(&self, device: &ash::Device) -> Result<PrimitiveAddresses> {
        Ok(PrimitiveAddresses {
            primitive: self.get_device_address(device)?,
        })
    }
    fn free(&mut self) {
        self.vertex_buffer = None;
        self.index_buffer = None;
        self.mat_index_buffer = None;
        self.primitive_data = None;
        self.main_buffer = None;
    }
}
