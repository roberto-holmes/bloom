use anyhow::Result;
use ash::{vk, RawPtr};

pub struct Destructor<T: Copy> {
    device: vk::Device,
    member: T,
    destructor: unsafe extern "system" fn(vk::Device, T, *const vk::AllocationCallbacks),
}
impl<T: Copy> Destructor<T> {
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
    pub fn get(&self) -> T {
        self.member
    }
}
impl<T: Copy> Drop for Destructor<T> {
    fn drop(&mut self) {
        log::trace!("Dropping {}", std::any::type_name::<T>());
        unsafe { (self.destructor)(self.device, self.member, None.as_raw_ptr()) };
    }
}

/// RAII wrapper for the ash::Device object
pub struct Device {
    device: ash::Device,
}

impl Device {
    pub fn new(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        create_info: vk::DeviceCreateInfo,
    ) -> Result<Self> {
        Ok(Self {
            device: unsafe { instance.create_device(*physical_device, &create_info, None)? },
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
