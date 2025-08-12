use ash::vk;
use hecs::Entity;
use std::{cmp::max, collections::HashMap, path::Path, sync::Arc};

use crate::{
    core::{begin_single_time_commands, end_single_time_command},
    error::{raise, raise_root_error, Error, Result},
    ray::{descriptor::Descriptor, RESERVED_SIZE},
    vulkan::{self},
};

/// Store the Bottom Level Acceleration Structures (BLAS)
pub struct TextureContainer<'a> {
    /// Keep track of where each entity is in the big vector
    texture_location: HashMap<Entity, u64>, // TODO: Figure out how to give this data to the GPU
    /// Note where the gaps are in the sparse vector so that they can be filled in
    empty_indices: Vec<u64>,
    // Store the total number of instances in the vector (ignoring gaps in the vector)
    pub texture_count: usize,
    /// Keep track of whether we have had any instances added or removed since the last time we generated the list of addresses
    has_changed: bool,

    textures: Vec<Option<vulkan::Image<'a>>>,

    pub descriptor: Descriptor,
}

impl<'a> TextureContainer<'a> {
    pub fn new(
        device: &ash::Device,
        db_device: &ash::ext::descriptor_buffer::Device,
        device_properties: vk::PhysicalDeviceProperties2<'a>,
        descriptor_buffer_properties: &vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'a>,
        allocator: &Arc<vk_mem::Allocator>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let mut descriptor = Descriptor::new(
            0,
            allocator,
            device,
            db_device,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            device_properties
                .properties
                .limits
                .max_descriptor_set_sampled_images, // TODO: Figure out how many textures to support
            vk::ShaderStageFlags::RAYGEN_KHR // TODO: Remove Raygen (this is just for debug viewing the image directly)
                    | vk::ShaderStageFlags::MISS_KHR // For the skybox
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            descriptor_buffer_properties,
        )?;

        // Create a placeholder texture so that we can always fallback to a texture if something goes wrong
        const PLACEHOLDER_SIZE: u32 = 256;
        let placeholder_mip_levels = ((::std::cmp::max(PLACEHOLDER_SIZE, PLACEHOLDER_SIZE) as f32)
            .log2()
            .floor() as u32)
            + 1;

        let mut placeholder_texture = create_placeholder_texture(
            &device,
            device_properties,
            &allocator,
            PLACEHOLDER_SIZE,
            placeholder_mip_levels,
            command_pool,
            queue,
        )?;
        placeholder_texture
            .create_sampler(placeholder_mip_levels)
            .unwrap();
        placeholder_texture
            .generate_mipmaps(device, command_pool, queue, placeholder_mip_levels)
            .unwrap();
        // Populate the descriptor with the image and its sampler
        descriptor.get_descriptor_combined(
            &db_device,
            placeholder_texture.sampler().unwrap(),
            placeholder_texture.view(),
            0,
            descriptor_buffer_properties,
        )?;

        let mut textures = Vec::with_capacity(RESERVED_SIZE);
        textures.push(Some(placeholder_texture));

        Ok(Self {
            descriptor,
            empty_indices: Vec::with_capacity(RESERVED_SIZE),
            texture_location: HashMap::with_capacity(RESERVED_SIZE),
            texture_count: 1,
            has_changed: false,
            textures,
        })
    }
    /// Add instances to the buffer, taking care to fill in any spaces left by previously removed instances.
    /// Will return false if the entity is already present
    pub fn try_add(
        &mut self,
        device: &ash::Device,
        device_properties: vk::PhysicalDeviceProperties2<'a>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        allocator: &Arc<vk_mem::Allocator>,
        entity: Entity,
        texture_path: &Path,
    ) -> Result<bool> {
        if self.texture_location.contains_key(&entity) {
            return Ok(false);
        }
        // Create new image with sampler
        let texture = create_texture(
            &device,
            device_properties,
            &allocator,
            command_pool,
            queue,
            texture_path,
        )?;

        // Try to add the new texture into a gap in the vector, if possible
        match self.empty_indices.pop() {
            Some(i) => {
                log::debug!("Inserting texture into index {i}");
                self.textures[i as usize] = Some(texture);
                self.texture_location.insert(entity, i);
            }
            None => {
                log::debug!(
                    "Appending entity {} to texture {}",
                    entity.id(),
                    self.texture_count
                );
                // Reserve some additional space if we are going to be reallocating anyway
                if self.textures.capacity() == self.textures.len() {
                    log::debug!("Reallocating texture vector");
                    self.textures.reserve(RESERVED_SIZE);
                }
                self.textures.push(Some(texture));
                self.texture_location
                    .insert(entity, self.texture_count as u64);
            }
        }

        self.texture_count += 1;
        self.has_changed = true;
        Ok(true)
    }

    /// Removes an entity from the instance buffer
    pub fn remove(&mut self, e: &Entity) {
        // TODO: Consider blanking the memory
        match self.texture_location.remove_entry(e) {
            Some((_, v)) => {
                self.textures[v as usize] = None;
                self.empty_indices.push(v);
                self.texture_count -= 1;
            }
            None => {
                log::warn!("Tried to remove a nonexistant instance")
            }
        }
        self.has_changed = true;
    }

    pub fn update_descriptor(
        &mut self,
        db_device: &ash::ext::descriptor_buffer::Device,
        descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'a>,
    ) -> Result<()> {
        if !self.has_changed {
            return Ok(());
        }
        // TODO: Figure out how to modify the descriptor_count field in the descriptor layout
        let mut i = 0;
        // TODO: Somehow connect this to the entity hash map
        for t in &self.textures {
            if let Some(t) = t {
                self.descriptor.get_descriptor_combined(
                    db_device,
                    t.sampler().unwrap(),
                    t.view(),
                    i,
                    &descriptor_buffer_properties,
                )?;
                i += 1;
            }
        }
        self.has_changed = false;
        Ok(())
    }
}

pub fn create_placeholder_texture<'a>(
    device: &ash::Device,
    device_properties: vk::PhysicalDeviceProperties2<'a>,
    allocator: &Arc<vk_mem::Allocator>,
    size: u32,
    mip_levels: u32,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<vulkan::Image<'a>> {
    let command_buffer = begin_single_time_commands(device, command_pool).unwrap(); // TODO: Replace with new errors
                                                                                    // Create the image in memory
    let image = vulkan::Image::new(
        format!("Placeholder Texture"),
        device,
        device_properties,
        allocator,
        vk_mem::MemoryUsage::AutoPreferDevice,
        vk_mem::AllocationCreateFlags::empty(),
        size,
        size,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        mip_levels,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
    )
    .unwrap();

    let subresource_range = [vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mip_levels,
        base_array_layer: 0,
        layer_count: 1,
    }];

    let clear_barrier = [vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image.get())
        .subresource_range(subresource_range[0])
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)];
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&clear_barrier);

    // Clear the old ray image so we can start accumulating again
    unsafe {
        device.cmd_pipeline_barrier2(command_buffer, &dependency);

        let clear_value = vk::ClearColorValue {
            float32: [1.0, 0.5, 0.0, 1.0],
        };

        device.cmd_clear_color_image(
            command_buffer,
            image.get(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &clear_value,
            &subresource_range,
        );
    }
    // We want to stay as TRANSFER DST for the mipmap creation

    end_single_time_command(device, command_pool, queue, command_buffer).unwrap(); // TODO: Replace with new errors
    Ok(image)
}

pub fn create_texture<'a>(
    device: &ash::Device,
    device_properties: vk::PhysicalDeviceProperties2<'a>,
    allocator: &Arc<vk_mem::Allocator>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    texture_path: &Path,
) -> Result<vulkan::Image<'a>> {
    let image_reader = image::ImageReader::open(texture_path);
    if let Err(e) = &image_reader {
        raise(
            format!("Failed to open image {}", texture_path.display()),
            e,
        )?;
    }
    let image_decoder = image_reader.unwrap().decode(); // Apparently this function is slow in debug mode
    if let Err(e) = &image_decoder {
        raise(
            format!("Failed to decode image {}", texture_path.display()),
            e,
        )?
    }
    let mut image_object = image_decoder.unwrap();
    image_object = image_object.flipv();
    let (width, height) = (image_object.width(), image_object.height());
    let image_size = (std::mem::size_of::<u8>() as u32 * width * height * 4) as vk::DeviceSize;
    let image_data = image_object.to_rgba8().into_raw();

    // TODO: Do we always want to create all the mip maps
    let mip_levels = ((max(width, height) as f32).log2().floor() as u32) + 1;

    log::debug!(
        "Texture {} has {mip_levels} mip levels",
        texture_path.display()
    );

    if image_size <= 0 {
        raise_root_error(format!(
            "Failed to load texture image {}",
            texture_path.display()
        ))?;
    }

    if image_data.len() as u64 != image_size {
        raise_root_error(format!(
            "Image size != width x height x 4 Bytes ({} != {width}x{height})",
            image_data.len()
        ))?;
    }

    let mut image = vulkan::Image::new_populated(
        format!("{}", texture_path.display()),
        device,
        allocator,
        device_properties,
        command_pool,
        queue,
        image_data.as_ptr(),
        image_size,
        width,
        height,
        mip_levels,
        vk::Format::R8G8B8A8_SRGB,
    )
    .unwrap(); // Todo: Replace with new errors

    image
        .generate_mipmaps(device, command_pool, queue, mip_levels)
        .unwrap(); // TODO: new error

    image.create_sampler(mip_levels).unwrap();

    Ok(image)
}
