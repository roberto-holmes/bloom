use anyhow::Result;
use ash::vk;

#[cfg(not(target_arch = "wasm32"))]
use crate::{vec::Vec3, vulkan};

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
}

pub trait Extrema {
    fn get_extrema(&self) -> (Vec3, Vec3);
    fn get_center(&self) -> Vec3;
}

#[repr(C)]
pub struct SphereData {
    pub object_type: u32,
    pub center: Vec3,
    pub radius: f32,
    _material: u32,
    pub is_selected: u32,
}

impl SphereData {
    fn new(center: Vec3, radius: f32, material: u32, is_selected: bool) -> Self {
        Self {
            object_type: ObjectType::Sphere as u32,
            center,
            radius,
            _material: material,
            is_selected: if is_selected { 1 } else { 0 },
        }
    }
}

pub struct Sphere {
    data: SphereData,
    data_buffer: vulkan::Buffer,
}

impl Sphere {
    pub fn new(
        allocator: &vk_mem::Allocator,
        center: Vec3,
        radius: f32,
        material: u32,
    ) -> Result<Self> {
        let data = SphereData::new(center, radius, material, false);

        let data_buffer = vulkan::Buffer::new_populated(
            allocator,
            size_of::<SphereData>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &data,
            1,
        )?;

        Ok(Self { data, data_buffer })
    }
}

impl Extrema for Sphere {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        let min = self.data.center - Vec3::all(self.data.radius);
        let max = self.data.center + Vec3::all(self.data.radius);
        (min, max)
    }
    fn get_center(&self) -> Vec3 {
        self.data.center
    }
}

impl Addressable for Sphere {
    fn get_addresses(&self, device: &ash::Device) -> PrimitiveAddresses {
        PrimitiveAddresses {
            primitive: self.data_buffer.get_device_address(device),
        }
    }
}

// #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MeshData {
    pub object_type: u32,
    pub vertices: vk::DeviceAddress,
    pub indices: vk::DeviceAddress,
    pub material_indices: vk::DeviceAddress,
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

pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_indices: Vec<u32>,
    // pub vertex_count: u32, // Total number of vertices that make up this model
    // pub index_count: u32,
    // pub mesh: Mesh,
    pub vertex_buffer: vulkan::Buffer, // Buffer containing all the vertices that make up the object
    pub index_buffer: vulkan::Buffer,  // How to connect up the vertices into triangles
    pub mat_index_buffer: vulkan::Buffer, // What material each triangle maps to
    // pub materials_buffer: vulkan::Buffer, // TODO: Make the materials stored by the scene, not each object
    pub primitive_data: MeshData,
    main_buffer: vulkan::Buffer,
}

impl Model {
    pub fn new_cube(allocator: &vk_mem::Allocator, device: &ash::Device) -> Result<Self> {
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
        let material_indices = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];

        let (vertex_buffer, index_buffer, mat_index_buffer, primitive_data, main_buffer) =
            Model::allocate(allocator, device, &vertices, &indices, &material_indices)?;

        Ok(Self {
            vertices,
            indices,
            material_indices,
            primitive_data,
            main_buffer,
            vertex_buffer,
            index_buffer,
            mat_index_buffer,
        })
    }
    pub fn new_mirror(allocator: &vk_mem::Allocator, device: &ash::Device) -> Result<Self> {
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
        let material_indices = vec![7; 12];

        let (vertex_buffer, index_buffer, mat_index_buffer, primitive_data, main_buffer) =
            Model::allocate(allocator, device, &vertices, &indices, &material_indices)?;

        Ok(Self {
            vertices,
            indices,
            material_indices,
            primitive_data,
            main_buffer,
            vertex_buffer,
            index_buffer,
            mat_index_buffer,
        })
    }
    pub fn new_plane(allocator: &vk_mem::Allocator, device: &ash::Device) -> Result<Self> {
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
        let material_indices = vec![6, 6];

        let (vertex_buffer, index_buffer, mat_index_buffer, primitive_data, main_buffer) =
            Model::allocate(allocator, device, &vertices, &indices, &material_indices)?;

        Ok(Self {
            vertices,
            indices,
            material_indices,
            primitive_data,
            main_buffer,
            vertex_buffer,
            index_buffer,
            mat_index_buffer,
        })
    }
    fn allocate(
        allocator: &vk_mem::Allocator,
        device: &ash::Device,
        vertices: &Vec<Vertex>,
        indices: &Vec<u32>,
        material_indices: &Vec<u32>,
    ) -> Result<(
        vulkan::Buffer, // Vertices
        vulkan::Buffer, // Indices
        vulkan::Buffer, // Material Indices
        MeshData,
        vulkan::Buffer,
    )> {
        let vertex_buffer_size = vertices.len() * size_of::<Vertex>();
        let index_buffer_size = indices.len() * size_of::<u32>();
        let mat_index_buffer_size = material_indices.len() * size_of::<i32>();

        // let max_material_index = 1;
        // TODO: Make sure material indices never exceed the number of available materials?
        // for index in &mut material_indices {
        //     *index = std::cmp::min(max_material_index, *index);
        // }

        // TODO: Add a staging buffer so these buffers can live on the GPU
        let vertex_buffer = vulkan::Buffer::new_populated(
            allocator,
            vertex_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vertices.as_ptr(),
            vertices.len(),
        )?;
        let index_buffer = vulkan::Buffer::new_populated(
            allocator,
            index_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            indices.as_ptr(),
            indices.len(),
        )?;
        let mat_index_buffer = vulkan::Buffer::new_populated(
            allocator,
            mat_index_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            material_indices.as_ptr(),
            material_indices.len(),
        )?;

        let object_type = ObjectType::Triangle as u32;
        let addresses = MeshData {
            object_type,
            vertices: vertex_buffer.get_device_address(device),
            indices: index_buffer.get_device_address(device),
            material_indices: mat_index_buffer.get_device_address(device),
        };

        let main_buffer = vulkan::Buffer::new_populated(
            allocator,
            size_of::<MeshData>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &addresses,
            1,
        )?;

        Ok((
            vertex_buffer,
            index_buffer,
            mat_index_buffer,
            addresses,
            main_buffer,
        ))
    }
    pub fn get_device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        self.main_buffer.get_device_address(device)
    }
}

impl Addressable for Model {
    fn get_addresses(&self, device: &ash::Device) -> PrimitiveAddresses {
        PrimitiveAddresses {
            primitive: self.get_device_address(device),
        }
    }
}
