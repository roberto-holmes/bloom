use ash::vk;

use crate::{
    core::{aligned_size, create_bindless_descriptor_set_layout},
    error::{Result, raise},
    vulkan::{self, Destructor},
};

pub struct Descriptor {
    pub layout: Destructor<vk::DescriptorSetLayout>,
    pub buffer: vulkan::Buffer,
    offset: vk::DeviceSize,
}

impl Descriptor {
    pub fn new(
        binding: u32,
        allocator: &vk_mem::Allocator,
        device: &ash::Device,
        db_device: &ash::ext::descriptor_buffer::Device,
        descriptor_type: vk::DescriptorType,
        descriptor_count: u32,
        stage_flags: vk::ShaderStageFlags,
        descriptor_buffer_properties: &vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    ) -> Result<Self> {
        let layout = create_bindless_descriptor_set_layout(
            device,
            binding,
            descriptor_type,
            descriptor_count,
            stage_flags,
        )
        .unwrap(); // TODO: Errors
        // Compute and align the sizes of the sets so that we can compute the offsets
        let size = aligned_size(
            unsafe { db_device.get_descriptor_set_layout_size(layout.get()) },
            descriptor_buffer_properties.descriptor_buffer_offset_alignment,
        );
        // Get the offsets of the descriptor bindings of each set layout as they don't necessarily start at
        // the beginning of the buffer (driver could add metadata at the start or something)
        let offset = unsafe { db_device.get_descriptor_set_layout_binding_offset(layout.get(), 0) };

        let mut usage = vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        if descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
            usage |= vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
        };

        let buffer = vulkan::Buffer::new_aligned(
            allocator,
            size,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::MAPPED //? Does it need to be mapped or just HOST_VISIBLE?
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM, // TODO: Random, sequential, or neither?
            // | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE, // https://www.khronos.org/blog/vk-ext-descriptor-buffer says we can use this flag if we "ensure multiple writes are written contiguously"
            usage,
            descriptor_buffer_properties.descriptor_buffer_offset_alignment, // TODO: Does this buffer need to be aligned?
            "Descriptor Data",
        )
        .unwrap(); // TODO: Errors
        Ok(Self {
            layout,
            buffer,
            offset,
        })
    }

    pub fn get_descriptor_image(
        &mut self,
        db_device: &ash::ext::descriptor_buffer::Device,
        image_view: vk::ImageView,
        offset: usize,
        descriptor_buffer_properties: &vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    ) -> Result<()> {
        // Create a new descriptor pointing at the new image
        let image_descriptor = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: image_view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        let image_descriptor_info = vk::DescriptorGetInfoEXT {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            data: vk::DescriptorDataEXT {
                p_storage_image: &image_descriptor,
            },
            ..Default::default()
        };
        log::info!(
            "Offsetting data by {}",
            self.offset
                + (descriptor_buffer_properties.storage_image_descriptor_size * offset)
                    as vk::DeviceSize
        );
        // Copy the desciptor into the descriptor buffer
        unsafe {
            let mapped_data = self
                .buffer
                .get_mapped_data(
                    descriptor_buffer_properties.storage_image_descriptor_size as vk::DeviceSize,
                    self.offset
                        + (descriptor_buffer_properties.storage_image_descriptor_size * offset)
                            as vk::DeviceSize,
                )
                .unwrap(); // TODO: Errors

            db_device.get_descriptor(&image_descriptor_info, mapped_data);
        };
        Ok(())
    }

    pub fn get_descriptor_combined(
        &mut self,
        db_device: &ash::ext::descriptor_buffer::Device,
        sampler: vk::Sampler,
        image_view: vk::ImageView,
        offset: usize,
        descriptor_buffer_properties: &vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    ) -> Result<()> {
        // Create a new descriptor pointing at the new image
        let image_descriptor = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::READ_ONLY_OPTIMAL,
        };
        let image_descriptor_info = vk::DescriptorGetInfoEXT {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            data: vk::DescriptorDataEXT {
                p_combined_image_sampler: &image_descriptor,
            },
            ..Default::default()
        };

        // Copy the desciptor into the descriptor buffer
        unsafe {
            let mapped_data = self
                .buffer
                .get_mapped_data(
                    descriptor_buffer_properties.combined_image_sampler_descriptor_size
                        as vk::DeviceSize,
                    self.offset
                        + (descriptor_buffer_properties.combined_image_sampler_descriptor_size
                            * offset) as vk::DeviceSize,
                )
                .unwrap(); // TODO: Errors

            db_device.get_descriptor(&image_descriptor_info, mapped_data);
        };
        Ok(())
    }

    pub fn get_descriptor_tlas(
        &mut self,
        db_device: &ash::ext::descriptor_buffer::Device,
        tlas_handle: vk::DeviceAddress,
        offset: usize,
        descriptor_buffer_properties: &vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    ) -> Result<()> {
        let image_descriptor_info = vk::DescriptorGetInfoEXT {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            data: vk::DescriptorDataEXT {
                acceleration_structure: tlas_handle,
            },
            ..Default::default()
        };
        // Copy the desciptor into the descriptor buffer
        unsafe {
            let mapped_data = self
                .buffer
                .get_mapped_data(
                    descriptor_buffer_properties.acceleration_structure_descriptor_size
                        as vk::DeviceSize,
                    self.offset
                        + (descriptor_buffer_properties.acceleration_structure_descriptor_size
                            * offset) as vk::DeviceSize,
                )
                .unwrap(); // TODO: Errors

            db_device.get_descriptor(&image_descriptor_info, mapped_data);
        };
        Ok(())
    }
}
