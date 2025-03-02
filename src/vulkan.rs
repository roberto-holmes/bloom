use std::sync::Arc;

use anyhow::Result;
use ash::{vk, RawPtr};
use vk_mem::{self, Alloc};

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
    fn clean(&mut self) {
        log::trace!("Dropping {}", std::any::type_name::<T>());
        unsafe { (self.destructor)(self.device, self.member, None.as_raw_ptr()) };
    }
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

pub struct Buffer {
    allocator: vk_mem::ffi::VmaAllocator,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    size: vk::DeviceSize,
}

impl Buffer {
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
        // new_buffer.allocation_info.mapped_data // TODO: Do we return the mapped data or should this struct hide it
        Ok(new_buffer)
    }
    pub fn new_generic(
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
        location: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        log::trace!("Creating VMA allocated buffer");
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
            size,
            allocation_info,
        })
    }
    pub fn populate<T>(&self, data: *const T, size: usize) -> Result<()> {
        unsafe {
            let mut mapped_data: *mut ::std::os::raw::c_void = ::std::ptr::null_mut();
            vk_mem::ffi::vmaMapMemory(self.allocator, self.allocation.0, &mut mapped_data)
                .result()?;

            (mapped_data as *mut T).copy_from_nonoverlapping(data, size);

            vk_mem::ffi::vmaUnmapMemory(self.allocator, self.allocation.0);
        }
        Ok(())
    }
    pub fn populate_mapped<T>(&self, data: *const T, size: usize) -> Result<()> {
        unsafe {
            if self.allocation_info.mapped_data.is_null() {
                return Err(anyhow::anyhow!(
                    "Tried to copy data into an unmapped buffer"
                ));
            }
            (self.allocation_info.mapped_data as *mut T).copy_from_nonoverlapping(data, size);
        }
        Ok(())
    }
    pub fn new_populated<T>(
        allocator: &vk_mem::Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        data: *const T,
        data_size: usize,
    ) -> Result<Self> {
        let buffer = Self::new_generic(
            allocator,
            size,
            vk_mem::MemoryUsage::Auto,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage,
        )?;
        buffer.populate(data, data_size)?;
        Ok(buffer)
    }
    pub fn create_descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: 0,
            range: self.size,
        }
    }
    pub fn get_device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        let buffer_device_ai = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { device.get_buffer_device_address(&buffer_device_ai) }
    }
    pub fn get(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        log::trace!("Destroying VMA allocated buffer");
        unsafe {
            vk_mem::ffi::vmaDestroyBuffer(self.allocator, self.buffer, self.allocation.0);
        }
    }
}

pub struct Image<'a> {
    device: vk::Device,
    image: Option<vk::Image>,
    view: Option<Destructor<vk::ImageView>>,
    allocation: vk_mem::Allocation,
    // allocation: vk_mem::ffi::VmaAllocationWrapper,
    alloc_create_info: vk_mem::AllocationCreateInfo,
    image_create_info: vk::ImageCreateInfo<'a>,
    view_create_info: vk::ImageViewCreateInfo<'a>,

    width: u32,
    height: u32,

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
    fn create_image(&mut self) -> Result<()> {
        unsafe {
            let mut create_info: vk_mem::ffi::VmaAllocationCreateInfo =
                (&self.alloc_create_info).into();
            create_info.pool = self.allocator_pool.pool.0;
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
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe { self.clean() };
        self.image_create_info.extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        self.create_image()?;
        self.create_view()?;
        Ok(())
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
