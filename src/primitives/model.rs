use anyhow::{anyhow, Result};
use ash::vk;

use crate::{vec::Vec3, vulkan};

use super::{Addressable, Extrema, ObjectType, Objectionable, PrimitiveAddresses};

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
    pos: Vec3,
    _pad1: u32,
    _normal: Vec3,
    _pad2: u32,
}

impl Vertex {
    fn new(pos: Vec3, normal: Vec3) -> Self {
        Self {
            pos,
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
    pub fn new_gltf_primitive(
        primitive: gltf::Primitive,
        buffers: &Vec<gltf::buffer::Data>,
        material_id: u32,
    ) -> Self {
        let r = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let mut indices = Vec::new();
        if let Some(gltf::mesh::util::ReadIndices::U16(gltf::accessor::Iter::Standard(iter))) =
            r.read_indices()
        {
            for v in iter {
                indices.push(v as u32);
            }
        }
        let mut positions = Vec::new();
        if let Some(iter) = r.read_positions() {
            for v in iter {
                positions.push(v);
            }
        }
        let mut textures = Vec::new();
        if let Some(gltf::mesh::util::ReadTexCoords::F32(gltf::accessor::Iter::Standard(iter))) =
            r.read_tex_coords(0)
        {
            for v in iter {
                textures.push(v);
            }
        }
        let mut normals = Vec::new();
        if let Some(iter) = r.read_normals() {
            for v in iter {
                normals.push(v);
            }
        }
        let mut joints = Vec::new();
        if let Some(gltf::mesh::util::ReadJoints::U8(gltf::accessor::Iter::Standard(iter))) =
            r.read_joints(0)
        {
            for v in iter {
                joints.push(v);
            }
        }
        let mut weights = Vec::new();
        if let Some(gltf::mesh::util::ReadWeights::F32(gltf::accessor::Iter::Standard(iter))) =
            r.read_weights(0)
        {
            for v in iter {
                weights.push(v);
            }
        }
        let mut vertices = Vec::with_capacity(positions.len());

        // Ensure we have a normal for each vertex
        assert_eq!(positions.len(), normals.len());

        // Populate the vertices with positions and normals
        for (i, p) in positions.iter().enumerate() {
            vertices.push(Vertex::new(Vec3(*p), Vec3(normals[i])));
        }

        let material_indices = vec![material_id; indices.len() / 3];

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
    fn allocate(
        &mut self,
        allocator: &vk_mem::Allocator,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<()> {
        // let max_material_index = 1;
        // TODO: Make sure material indices never exceed the number of available materials?
        // for index in &mut material_indices {
        //     *index = std::cmp::min(max_material_index, *index);
        // }

        // TODO: Add a staging buffer so these buffers can live on the GPU
        self.vertex_buffer = Some(vulkan::Buffer::new_populated_staged(
            device,
            command_pool,
            queue,
            allocator,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            self.vertices.as_ptr(),
            self.vertices.len(),
        )?);
        self.index_buffer = Some(vulkan::Buffer::new_populated_staged(
            device,
            command_pool,
            queue,
            allocator,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            self.indices.as_ptr(),
            self.indices.len(),
        )?);
        self.mat_index_buffer = Some(vulkan::Buffer::new_populated_staged(
            device,
            command_pool,
            queue,
            allocator,
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

        self.main_buffer = Some(vulkan::Buffer::new_populated_staged(
            device,
            command_pool,
            queue,
            allocator,
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

impl Extrema for Model {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        if self.vertices.len() == 0 {
            log::warn!("Trying to check the size of an empty model");
            return (Vec3::zero(), Vec3::zero());
        }
        let mut min = self.vertices[0].pos;
        let mut max = self.vertices[0].pos;

        for v in &self.vertices {
            min = min.min_extrema(&v.pos);
            max = max.max_extrema(&v.pos);
        }

        (min, max)
    }
}
