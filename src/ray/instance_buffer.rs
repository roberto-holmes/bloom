use anyhow::Result;
use ash::vk;
use hecs::Entity;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{physics, ray::RESERVED_SIZE, vulkan};

#[repr(C)]
struct Instance {
    pub asi: vk::AccelerationStructureInstanceKHR,
    pub entity: physics::EntityData,
}

impl Instance {
    pub const fn size() -> u64 {
        size_of::<Self>() as u64
    }
}

/// Store the Bottom Level Acceleration Structures (BLAS)
pub struct InstanceBuffer {
    /// Store all of the instances in the scene in a sparcely populated GPU only buffer
    buffer: vulkan::Buffer,
    /// The TLAS will want an array of addresses fo each instance
    address_buffer: vulkan::Buffer,
    /// Keep track of where each entity is in the big buffer
    entity_location: HashMap<Entity, u64>, // TODO: Figure out how to give this data to the GPU
    /// Note where the gaps aer in the sparse buffer so that they can be filled in
    empty_indices: Vec<u64>,
    // Store the total number of instances in the buffer (ignoring gaps in the buffer)
    pub instance_count: usize,
    /// Keep track of whether we have had any instances added or removed since the last time we generated the list of addresses
    has_changed: bool,

    orphans: Vec<Entity>,
}

impl InstanceBuffer {
    pub fn new(allocator: &Arc<vk_mem::Allocator>) -> Result<Self> {
        Ok(Self {
            buffer: vulkan::Buffer::new_gpu(
                &allocator,
                Instance::size() * RESERVED_SIZE as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?,
            address_buffer: vulkan::Buffer::new_gpu(
                &allocator,
                (size_of::<vk::DeviceAddress>() * RESERVED_SIZE) as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?,
            empty_indices: Vec::with_capacity(RESERVED_SIZE),
            entity_location: HashMap::with_capacity(RESERVED_SIZE),
            instance_count: 0,
            has_changed: false,
            orphans: Vec::with_capacity(RESERVED_SIZE),
        })
    }

    /// Create a tightly packed array of device addresses to all of the instances of Bottom Level Acceleration Structures
    pub fn get_address_array(
        &mut self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        allocator: &Arc<vk_mem::Allocator>,
    ) -> Result<vk::DeviceAddress> {
        // We only need to change the addresses visible to the TLAS if there are new or missing instances
        if !self.has_changed {
            return Ok(self.address_buffer.get_device_address(device));
        }
        // Create a vector of the addresses to each instance
        let base = self.buffer.get_device_address(device);
        let mut v = Vec::with_capacity(self.instance_count);
        for (_, index) in &self.entity_location {
            v.push(base + index * Instance::size());
        }

        log::debug!("BLAS Addresses: {:?}", v);

        if v.len() != self.instance_count {
            log::warn!(
                "Number of addresses {} != number of instances {}",
                v.len(),
                self.instance_count
            );
        }

        // Fill the address buffer with the data from the vector, allowing it to grow if necessary
        // TODO: Consider shrinking the array if the data is much smaller than the allocated memory?
        if !self
            .address_buffer
            .check_total_space::<vk::DeviceAddress>(v.len())
        {
            self.address_buffer = vulkan::Buffer::new_populated_staged(
                device,
                command_pool,
                queue,
                allocator,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::UNIFORM_BUFFER,
                v.as_ptr(),
                v.len(),
                v.len() + RESERVED_SIZE,
            )?;
        } else {
            self.address_buffer.populate_staged(
                device,
                command_pool,
                queue,
                allocator,
                v.as_ptr(),
                v.len(),
            )?;
        }
        self.has_changed = false;
        let address = self.address_buffer.get_device_address(device);

        log::debug!("Buffer of BLAS Addresses address: {address}");

        Ok(address)
    }

    /// Add instances to the buffer, taking care to fill in any spaces left by previously removed instances.
    /// Will return false if the entity is already present
    pub fn try_add(
        &mut self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        allocator: &Arc<vk_mem::Allocator>,
        entity: Entity,
        new_as_instance: vk::AccelerationStructureInstanceKHR,
        base_transform: cgmath::Matrix4<f32>,
        world_transform: cgmath::Matrix4<f32>,
    ) -> Result<bool> {
        if self.entity_location.contains_key(&entity) {
            return Ok(false);
        }
        // Ensure the buffer is big enough for the new instance
        self.grow(device, command_pool, queue, allocator)?;

        let new_instance = Instance {
            asi: new_as_instance,
            entity: physics::EntityData {
                base_transform: base_transform,
                world_transform: world_transform,
                entity: entity.id(),
                pad: [0; 3],
            },
        };

        // Try to add the new instance into a gap in the buffer, if possible
        match self.empty_indices.pop() {
            Some(i) => {
                log::debug!("Inserting instance into index {i}");
                self.buffer.insert_staged(
                    device,
                    command_pool,
                    queue,
                    allocator,
                    i,
                    &new_instance,
                    1,
                )?;
                self.entity_location.insert(entity, i);
            }
            None => {
                log::debug!(
                    "Appending entity {} to instance {}",
                    entity.id(),
                    self.instance_count
                );
                log::warn!(
                    "Instance has transform {:?}",
                    new_instance.asi.transform.matrix
                );
                self.buffer.append_staged(
                    device,
                    command_pool,
                    queue,
                    allocator,
                    &new_instance,
                    1,
                )?;
                self.entity_location
                    .insert(entity, self.instance_count as u64);
            }
        }
        self.instance_count += 1;
        self.has_changed = true;
        Ok(true)
    }

    pub fn remove_orphans(&mut self, active_entities: &HashSet<Entity>) -> usize {
        let all_entities: HashSet<Entity> = self.entity_location.keys().cloned().collect();
        let orphans: HashSet<&Entity> = all_entities.difference(active_entities).collect();

        for &o in &orphans {
            log::trace!("Removing instance orphan {:?}", o);
            self.orphans.push(*o);
            self.remove(o);
        }
        orphans.len()
    }

    /// Removes an entity from the instance buffer
    pub fn remove(&mut self, e: &Entity) {
        // TODO: Consider blanking the memory
        match self.entity_location.remove_entry(e) {
            Some((_, v)) => {
                self.empty_indices.push(v);
                self.instance_count -= 1;
            }
            None => {
                log::warn!("Tried to remove a nonexistant instance")
            }
        }
        self.has_changed = true;
    }

    /// If necessary, allocates new memory and copies over the old buffer
    fn grow(
        &mut self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        allocator: &Arc<vk_mem::Allocator>,
    ) -> Result<()> {
        // We only need to reallocate the buffer if the current one is not big enough to deal with all the instances we have
        if !self.buffer.check_available_space::<Instance>(1) {
            log::debug!(
                "Resizing instance buffer to {} B",
                Instance::size() * (self.instance_count + RESERVED_SIZE) as u64
            );
            // Create a new (larger) buffer
            let mut new_buffer = vulkan::Buffer::new_gpu(
                allocator,
                Instance::size() * (self.instance_count + RESERVED_SIZE) as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;

            new_buffer.copy_from::<Instance>(
                device,
                command_pool,
                queue,
                &self.buffer,
                self.instance_count,
            )?;

            // Drop and replace old buffer
            self.buffer = new_buffer;
        }
        self.has_changed = true;
        Ok(())
    }
}
