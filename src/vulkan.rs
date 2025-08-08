use std::sync::Arc;

use anyhow::{anyhow, Result};
use ash::{vk, RawPtr};
use colored::Colorize;
use vk_mem::{self, Alloc};

use crate::core;

#[derive(Debug)]
pub struct Destructor<T: Copy + Default> {
    device: vk::Device,
    member: T,
    destructor: unsafe extern "system" fn(vk::Device, T, *const vk::AllocationCallbacks),
}
impl<T: Copy + Default> Destructor<T> {
    pub fn new(
        device: &ash::Device,
        member: T,
        destructor: unsafe extern "system" fn(vk::Device, T, *const vk::AllocationCallbacks),
    ) -> Self {
        Self {
            device: device.handle(),
            member,
            destructor,
        }
    }
    pub fn new_raw(
        device: vk::Device,
        member: T,
        destructor: unsafe extern "system" fn(vk::Device, T, *const vk::AllocationCallbacks),
    ) -> Self {
        Self {
            device,
            member,
            destructor,
        }
    }
    pub fn get(&self) -> T {
        self.member
    }
    #[track_caller]
    fn clean(&mut self) {
        log::trace!("Dropping {}", std::any::type_name::<T>());
        unsafe { (self.destructor)(self.device, self.member, None.as_raw_ptr()) };
    }
    #[track_caller]
    pub fn empty(&mut self) {
        self.clean();
        self.member = T::default();
    }
}
impl<T: Copy + Default> Drop for Destructor<T> {
    fn drop(&mut self) {
        self.clean();
    }
}

/// RAII wrapper for the ash::Device object
pub struct Device {
    device: ash::Device,
}

impl Device {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        create_info: vk::DeviceCreateInfo,
    ) -> Result<Self> {
        Ok(Self {
            device: unsafe { instance.create_device(physical_device, &create_info, None)? },
        })
    }
    pub fn get(&self) -> &ash::Device {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        log::trace!("Dropping Device");
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

/// RAII wrapper for the ash::Instance object
pub struct Instance {
    instance: ash::Instance,
}

impl Instance {
    pub fn new(entry: &ash::Entry, create_info: vk::InstanceCreateInfo) -> Result<Self> {
        Ok(Self {
            instance: unsafe { entry.create_instance(&create_info, None)? },
        })
    }
    pub fn get(&self) -> &ash::Instance {
        &self.instance
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        log::trace!("Dropping Instance");
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

/// RAII wrapper for the vk::SwapchainKHR object
pub struct Swapchain {
    pub loader: ash::khr::swapchain::Device,
    member: vk::SwapchainKHR,
}

impl Swapchain {
    pub fn new(
        loader: ash::khr::swapchain::Device,
        create_info: vk::SwapchainCreateInfoKHR,
    ) -> Result<Self> {
        let member = unsafe { loader.create_swapchain(&create_info, None)? };
        Ok(Self { loader, member })
    }
    pub fn get(&self) -> vk::SwapchainKHR {
        self.member
    }
    fn clean(&mut self) {
        log::trace!("Dropping Swapchain");
        unsafe {
            self.loader.destroy_swapchain(self.member, None);
        }
    }
    pub fn empty(&mut self) {
        self.clean();
        self.member = vk::SwapchainKHR::null();
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.clean();
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Buffer {
    allocator: vk_mem::ffi::VmaAllocator,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    populated_size: vk::DeviceSize,
    total_size: vk::DeviceSize,
    type_name: &'static str,
    name: Option<&'static str>,
}

impl Buffer {
    #[allow(unused)]
    #[track_caller]
    pub fn new_cpu(
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::new_generic(
            allocator,
            size,
            vk_mem::MemoryUsage::AutoPreferHost,
            vk_mem::AllocationCreateFlags::empty(),
            usage,
        )
    }
    #[allow(unused)]
    #[track_caller]
    pub fn new_gpu(
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::new_generic(
            allocator,
            size,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
            usage,
        )
    }
    #[allow(unused)]
    #[track_caller]
    pub fn new(
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::new_generic(
            allocator,
            size,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::empty(),
            usage,
        )
    }
    #[track_caller]
    pub fn new_mapped(
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let new_buffer = Self::new_generic(
            allocator,
            size,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage,
        )?;
        Ok(new_buffer)
    }
    #[track_caller]
    pub fn new_generic(
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
        location: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        // log::trace!("Creating VMA allocated buffer");
        if size == 0 {
            return Err(anyhow!("Tried to create a 0 size buffer"));
        }
        let create_info = vk_mem::AllocationCreateInfo {
            usage: location,
            flags,
            ..Default::default()
        };
        let (buffer, allocation) = unsafe {
            allocator.create_buffer(
                &ash::vk::BufferCreateInfo::default().size(size).usage(usage),
                &create_info,
            )?
        };
        let allocation_info = allocator.get_allocation_info(&allocation);
        Ok(Self {
            allocator: allocator.internal,
            buffer,
            allocation,
            populated_size: 0,
            total_size: size,
            allocation_info,
            type_name: "generic",
            name: None,
        })
    }
    #[track_caller]
    pub fn new_aligned(
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
        location: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
        usage: vk::BufferUsageFlags,
        alignment: vk::DeviceSize,
        name: &'static str,
    ) -> Result<Self> {
        // log::trace!("Creating Aligned VMA allocated buffer");
        if size == 0 {
            return Err(anyhow!("Tried to create a 0 size buffer"));
        }
        let create_info = vk_mem::AllocationCreateInfo {
            usage: location,
            flags,
            ..Default::default()
        };
        let (buffer, allocation) = unsafe {
            allocator.create_buffer_with_alignment(
                &ash::vk::BufferCreateInfo::default().size(size).usage(usage),
                &create_info,
                alignment,
            )?
        };
        let allocation_info = allocator.get_allocation_info(&allocation);
        Ok(Self {
            allocator: allocator.internal,
            buffer,
            allocation,
            populated_size: 0,
            total_size: size,
            allocation_info,
            type_name: "aligned",
            name: Some(name),
        })
    }
    #[track_caller]
    pub fn new_populated<T>(
        allocator: &vk_mem::Allocator,
        usage: vk::BufferUsageFlags,
        data: *const T,
        data_len: usize,
    ) -> Result<Self> {
        let mut buffer = Self::new_generic(
            allocator,
            (size_of::<T>() * data_len) as u64,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage,
        )?;
        buffer.type_name = std::any::type_name::<T>();
        buffer.populate(data, data_len)?;
        // log::trace!("Created buffer of {}", buffer.type_name);
        Ok(buffer)
    }
    #[track_caller]
    pub fn new_populated_staged<T>(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        allocator: &vk_mem::Allocator,
        usage: vk::BufferUsageFlags,
        data: *const T,
        data_len: usize,
        reserved_len: usize,
    ) -> Result<Self> {
        let size = (size_of::<T>() * data_len) as u64;
        let mut reserved_size = (size_of::<T>() * reserved_len) as u64;
        reserved_size = reserved_size.max(size);

        let staging_buffer = Self::new_populated(
            allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            data,
            data_len,
        )?;
        let mut buffer = Self::new_gpu(
            allocator,
            reserved_size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        core::copy_buffer(
            device,
            staging_buffer.get(),
            buffer.get(),
            size,
            command_pool,
            queue,
        )?;

        buffer.populated_size = size;

        buffer.type_name = std::any::type_name::<T>();

        Ok(buffer)
    }
    #[track_caller]
    pub fn populate<T>(&mut self, data: *const T, size: usize) -> Result<()> {
        if !self.total_size < (size_of::<T>() * size) as u64 {
            return Err(anyhow!(
                "Tried to populate a buffer with too much data ({}>{})",
                (size_of::<T>() * size),
                self.total_size
            ));
        }
        unsafe {
            let mut mapped_data: *mut ::std::os::raw::c_void = ::std::ptr::null_mut();
            vk_mem::ffi::vmaMapMemory(self.allocator, self.allocation.0, &mut mapped_data)
                .result()?;

            (mapped_data as *mut T).copy_from_nonoverlapping(data, size);

            vk_mem::ffi::vmaUnmapMemory(self.allocator, self.allocation.0);
        }
        self.populated_size = (size_of::<T>() * size) as vk::DeviceSize;
        self.type_name = std::any::type_name::<T>();
        Ok(())
    }
    pub fn populate_mapped<T>(&mut self, data: *const T, size: usize) -> Result<()> {
        unsafe {
            if self.allocation_info.mapped_data.is_null() {
                return Err(anyhow::anyhow!(
                    "Tried to copy data into an unmapped buffer"
                ));
            }
            (self.allocation_info.mapped_data as *mut T).copy_from_nonoverlapping(data, size);
        }
        self.populated_size = (size_of::<T>() * size) as vk::DeviceSize;
        self.type_name = std::any::type_name::<T>();
        Ok(())
    }
    #[track_caller]
    pub fn populate_staged<T>(
        &mut self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        allocator: &vk_mem::Allocator,
        data: *const T,
        data_len: usize,
    ) -> Result<()> {
        if !self.check_total_space::<T>(data_len) {
            log::error!("Tried to populate a buffer with too much data");
            return Err(anyhow!("Buffer Overrun"));
        }
        let size = (size_of::<T>() * data_len) as u64;

        let staging_buffer = Self::new_populated(
            allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            data,
            data_len,
        )?;

        let command_buffer = core::begin_single_time_commands(device, command_pool)?;
        unsafe {
            let copy_region = [vk::BufferCopy::default().size(size)];
            device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.get(),
                self.buffer,
                &copy_region,
            );
        }
        core::end_single_time_command(device, command_pool, queue, command_buffer)?;

        self.populated_size = size;

        self.type_name = std::any::type_name::<T>();

        Ok(())
    }
    #[track_caller]
    pub fn copy_from<T>(
        &mut self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        src: &Buffer,
        data_len: usize,
    ) -> Result<()> {
        if !self.check_total_space::<T>(data_len) {
            log::error!("Tried to populate a buffer with too much data");
            return Err(anyhow!("Buffer Overrun"));
        }
        let size = (size_of::<T>() * data_len) as u64;

        let command_buffer = core::begin_single_time_commands(device, command_pool)?;
        unsafe {
            let copy_region = [vk::BufferCopy::default().size(size)];
            device.cmd_copy_buffer(command_buffer, src.get(), self.buffer, &copy_region);
        }
        core::end_single_time_command(device, command_pool, queue, command_buffer)?;

        self.populated_size = size;

        self.type_name = std::any::type_name::<T>();

        Ok(())
    }
    #[track_caller]
    pub fn append_staged<T>(
        &mut self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        allocator: &vk_mem::Allocator,
        data: *const T,
        data_len: usize,
    ) -> Result<()> {
        let size = (size_of::<T>() * data_len) as u64;
        let available_space = self.total_size - self.populated_size;
        if size > available_space {
            log::error!(
                "Tried to append data to a buffer that is too full ({size}B > {available_space}B)"
            );
            return Err(anyhow!("Buffer overrun"));
        }
        let staging_buffer = Self::new_populated(
            allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            data,
            data_len,
        )?;

        let command_buffer = core::begin_single_time_commands(device, command_pool)?;
        unsafe {
            let copy_region = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: self.populated_size,
                size,
            }];
            device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.get(),
                self.buffer,
                &copy_region,
            );
        }
        core::end_single_time_command(device, command_pool, queue, command_buffer)?;

        self.populated_size += size;

        Ok(())
    }
    #[track_caller]
    pub fn insert_staged<T>(
        &mut self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        allocator: &vk_mem::Allocator,
        offset: u64,
        data: *const T,
        data_len: usize,
    ) -> Result<()> {
        let staging_buffer = Self::new_populated(
            allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            data,
            data_len,
        )?;

        let command_buffer = core::begin_single_time_commands(device, command_pool)?;
        unsafe {
            let copy_region = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: offset * size_of::<T>() as u64,
                size: (size_of::<T>() * data_len) as u64,
            }];
            device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.get(),
                self.buffer,
                &copy_region,
            );
        }
        core::end_single_time_command(device, command_pool, queue, command_buffer)?;
        Ok(())
    }
    /// Returns true if the passed data length would fit in the unused portion of the buffer
    #[track_caller]
    pub fn check_available_space<T>(&self, data_len: usize) -> bool {
        if self.total_size < self.populated_size {
            log::warn!(
                "Populated size {} is somehow bigger than total size {}, needs investigation",
                self.populated_size,
                self.total_size
            );
            return false;
        }
        (size_of::<T>() * data_len) as u64 <= (self.total_size - self.populated_size)
    }
    /// Returns true if the passed data length would fit in the buffer
    #[track_caller]
    pub fn check_total_space<T>(&self, data_len: usize) -> bool {
        (size_of::<T>() * data_len) as u64 <= self.total_size
    }
    #[track_caller]
    pub fn create_descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: 0,
            range: self.total_size,
        }
    }
    #[track_caller]
    pub fn get_device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        let buffer_device_ai = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { device.get_buffer_device_address(&buffer_device_ai) }
    }
    pub fn get(&self) -> vk::Buffer {
        self.buffer
    }
    pub unsafe fn get_mapped_data(
        &self,
        size: vk::DeviceSize,
        offset: vk::DeviceSize,
    ) -> Result<&mut [u8]> {
        if self.allocation_info.mapped_data.is_null() {
            log::error!("Tried to access an unmapped buffer");
            return Err(anyhow::anyhow!("Invalid buffer configuration"));
        }
        Ok(std::slice::from_raw_parts_mut(
            self.allocation_info
                .mapped_data
                .byte_offset(offset as isize) as *mut u8,
            size as usize,
        ))
    }
}

impl Drop for Buffer {
    #[track_caller]
    fn drop(&mut self) {
        match self.name {
            Some(n) => log::trace!(
                "Destroying {} buffer [{}]",
                n.cyan(),
                self.type_name.green()
            ),
            None => log::trace!("Destroying {} buffer", self.type_name.green()),
        }
        unsafe {
            vk_mem::ffi::vmaDestroyBuffer(self.allocator, self.buffer, self.allocation.0);
        }
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

pub struct Image<'a> {
    device: vk::Device,
    image: Option<vk::Image>,
    view: Option<Destructor<vk::ImageView>>,
    allocation: vk_mem::Allocation,
    // allocation: vk_mem::ffi::VmaAllocationWrapper,
    alloc_create_info: vk_mem::AllocationCreateInfo,
    image_create_info: vk::ImageCreateInfo<'a>,
    view_create_info: vk::ImageViewCreateInfo<'a>,

    pub width: u32,
    pub height: u32,

    allocator_pool: vk_mem::AllocatorPool,
    create_view: vk::PFN_vkCreateImageView,
    destroy_view: vk::PFN_vkDestroyImageView,
}

impl<'a> Image<'a> {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<vk_mem::Allocator>,
        location: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<Self> {
        let alloc_create_info = vk_mem::AllocationCreateInfo {
            usage: location,
            flags,
            ..Default::default()
        };
        let image_create_info = ash::vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            format,
            tiling,
            initial_layout: vk::ImageLayout::UNDEFINED,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let (image, allocation) =
            unsafe { allocator.create_image(&image_create_info, &alloc_create_info)? };
        let view_create_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        Ok(Self {
            device: device.handle(),
            // allocator: vk_mem::ffi::VmaAllocatorWrapper(allocator.internal),
            image: Some(image),
            view: Some(Destructor::new(
                device,
                unsafe { device.create_image_view(&view_create_info, None)? },
                device.fp_v1_0().destroy_image_view,
            )),
            allocation: allocation,
            alloc_create_info,
            image_create_info,
            view_create_info,
            create_view: device.fp_v1_0().create_image_view,
            destroy_view: device.fp_v1_0().destroy_image_view,

            width,
            height,
            allocator_pool: vk_mem::AllocatorPool {
                allocator: Arc::clone(allocator),
                pool: allocator.pool(),
            },
        })
    }
    // pub fn new_populated<T>(
    //     device: &ash::Device,
    //     allocator: &Arc<vk_mem::Allocator>,
    //     command_pool: vk::CommandPool,
    //     queue: vk::Queue,
    //     location: vk_mem::MemoryUsage,
    //     flags: vk_mem::AllocationCreateFlags,
    //     data: *const T,
    //     width: u32,
    //     height: u32,
    //     format: vk::Format,
    // ) -> Result<Self> {
    //     let image = Self::new(
    //         device,
    //         allocator,
    //         location,
    //         flags,
    //         width,
    //         height,
    //         format,
    //         vk::ImageTiling::OPTIMAL,
    //         vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
    //         vk::ImageAspectFlags::COLOR,
    //     )?;

    //     let size = (width * height) as u64;

    //     // Use a staging buffer so that we don't need the image to be able to map to CPU memory
    //     let mut staging_buffer = Buffer::new_mapped(
    //         allocator,
    //         size * size_of::<T>() as u64,
    //         vk::BufferUsageFlags::TRANSFER_SRC,
    //     )?;
    //     staging_buffer.populate_mapped(data, size as usize)?;

    //     // Copy image data into the image proper
    //     let command_buffer = core::begin_single_time_commands(device, command_pool)?;

    //     // Convert image layout to be able to copy into it
    //     let barriers = [vk::ImageMemoryBarrier2::default()
    //         .old_layout(vk::ImageLayout::UNDEFINED)
    //         .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
    //         .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
    //         .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
    //         .image(image.get())
    //         .subresource_range(vk::ImageSubresourceRange {
    //             aspect_mask: vk::ImageAspectFlags::COLOR,
    //             base_mip_level: 0,
    //             level_count: 1,
    //             base_array_layer: 0,
    //             layer_count: 1,
    //         })
    //         .src_stage_mask(vk::PipelineStageFlags2::NONE)
    //         .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
    //         .src_access_mask(vk::AccessFlags2::NONE)
    //         .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)];

    //     let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    //     unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };

    //     unsafe {
    //         let copy_region = [vk::BufferImageCopy {
    //             image_extent: vk::Extent3D {
    //                 width,
    //                 height,
    //                 depth: 1,
    //             },
    //             image_subresource: vk::ImageSubresourceLayers {
    //                 aspect_mask: vk::ImageAspectFlags::COLOR,
    //                 layer_count: 1,
    //                 ..Default::default()
    //             },
    //             ..Default::default()
    //         }];
    //         device.cmd_copy_buffer_to_image(
    //             command_buffer,
    //             staging_buffer.get(),
    //             image.image.unwrap(),
    //             vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    //             &copy_region,
    //         );
    //     }

    //     // Convert image layout to be readable by the shaders
    //     let barriers = [vk::ImageMemoryBarrier2::default()
    //         .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
    //         .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
    //         .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
    //         .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
    //         .image(image.get())
    //         .subresource_range(vk::ImageSubresourceRange {
    //             aspect_mask: vk::ImageAspectFlags::COLOR,
    //             base_mip_level: 0,
    //             level_count: 1,
    //             base_array_layer: 0,
    //             layer_count: 1,
    //         })
    //         .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
    //         .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
    //         .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
    //         .dst_access_mask(vk::AccessFlags2::SHADER_READ)];

    //     let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    //     unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };

    //     core::end_single_time_command(device, command_pool, queue, command_buffer)?;

    //     Ok(image)
    // }
    fn create_image(&mut self) -> Result<()> {
        unsafe {
            let mut create_info: vk_mem::ffi::VmaAllocationCreateInfo =
                (&self.alloc_create_info).into();
            create_info.pool = self.allocator_pool.pool.0;
            // TODO: Enable `VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT` for large images/buffers
            // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
            // create_info.flags =
            //     vk_mem::ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
            //         as u32;
            let mut image = vk::Image::null();
            let mut allocation: vk_mem::ffi::VmaAllocation = std::mem::zeroed();
            vk_mem::ffi::vmaCreateImage(
                self.allocator_pool.allocator.internal,
                &self.image_create_info,
                &create_info,
                &mut image,
                &mut allocation,
                std::ptr::null_mut(),
            )
            .result()?;
            self.image = Some(image);
            self.allocation = vk_mem::Allocation(allocation);
        }
        Ok(())
    }
    fn create_view(&mut self) -> Result<()> {
        self.view_create_info.image = self.image.unwrap();
        let mut image_view = std::mem::MaybeUninit::uninit();

        self.view = Some(Destructor::new_raw(
            self.device,
            unsafe {
                (self.create_view)(
                    self.device,
                    &self.view_create_info,
                    None.as_raw_ptr(),
                    image_view.as_mut_ptr(),
                )
                .assume_init_on_success(image_view)?
            },
            self.destroy_view,
        ));
        Ok(())
    }
    pub fn resize(&mut self, width: u32, height: u32) -> Result<bool> {
        if self.is_correct_size(width, height) {
            return Ok(false);
        }
        unsafe { self.clean() };
        self.image_create_info.extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        self.create_image()?;
        self.create_view()?;
        self.width = width;
        self.height = height;
        Ok(true)
    }
    pub fn view(&self) -> vk::ImageView {
        self.view.as_ref().unwrap().get()
    }
    pub fn get(&self) -> vk::Image {
        self.image.unwrap()
    }
    pub fn is_correct_size(&self, width: u32, height: u32) -> bool {
        width == self.width && height == self.height
    }
    unsafe fn clean(&mut self) {
        log::trace!("Destroying VMA allocated buffer");
        self.view = None;
        if self.image.is_some() {
            vk_mem::ffi::vmaDestroyImage(
                self.allocator_pool.allocator.internal,
                self.image.unwrap(),
                self.allocation.0,
            );
        }
        self.image = None;
    }
}

impl<'a> Drop for Image<'a> {
    fn drop(&mut self) {
        unsafe { self.clean() };
    }
}
