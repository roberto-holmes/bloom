use anyhow::Result;
use ash::{RawPtr, vk};

use crate::vulkan;

pub struct AccelerationStructure {
    device: vk::Device,
    destructor: vk::PFN_vkDestroyAccelerationStructureKHR,

    name: &'static str,

    pub handle: vk::AccelerationStructureKHR,
    pub device_address: vk::DeviceAddress,
    pub buffer: Option<vulkan::Buffer>,
}

impl AccelerationStructure {
    pub fn new(device: &ash::khr::acceleration_structure::Device, name: &'static str) -> Self {
        Self {
            device: device.device(),
            destructor: device.fp().destroy_acceleration_structure_khr,

            name,

            handle: vk::AccelerationStructureKHR::null(),
            device_address: 0,
            buffer: None,
        }
    }
    pub fn create_buffer(
        &mut self,
        allocator: &vk_mem::Allocator,
        build_size_info: vk::AccelerationStructureBuildSizesInfoKHR,
    ) -> Result<()> {
        log::trace!("Creating AS buffer");
        self.buffer = Some(vulkan::Buffer::new_gpu(
            allocator,
            build_size_info.acceleration_structure_size * 2,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            self.name,
        )?);
        Ok(())
    }
    pub fn grow_buffer(
        &mut self,
        allocator: &vk_mem::Allocator,
        build_size_info: vk::AccelerationStructureBuildSizesInfoKHR,
    ) -> Result<bool> {
        match self.buffer.as_mut() {
            None => {
                log::trace!("Growing AS buffer from 0");
                self.create_buffer(allocator, build_size_info)?;
                return Ok(true);
            }
            Some(b) => {
                // If the current buffer doesn't have enough space for the new acceleration structure,
                // create a new buffer to overwrite it
                if !b.check_available_space::<u8>(
                    build_size_info.acceleration_structure_size as usize,
                ) {
                    log::trace!("Growing AS buffer");
                    self.create_buffer(allocator, build_size_info)?;
                    return Ok(true);
                }
            }
        }
        // The current buffer is good enough
        Ok(false)
    }
    pub fn get_buffer(&self) -> vk::Buffer {
        match &self.buffer {
            Some(v) => v.get(),
            None => {
                log::error!("Acceleration Structure Buffer was not initialised prior to calling");
                vk::Buffer::null()
            }
        }
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        log::trace!("Dropping Acceleration Structure {:?}", self.handle);
        unsafe { (self.destructor)(self.device, self.handle, None.as_raw_ptr()) };
    }
}
