use ash::vk;
use macros::Index;

use crate::structures::SurfaceStuff;

extern crate proc_macro;

#[derive(Debug, Clone, Copy)]
pub struct QueueIndex {
    /// Has a suitable queue been found and are the indices thus valid
    is_found: bool,
    /// Which queue family the queue belongs to
    pub family_index: u32,
    /// Which queue inside the family we are
    pub index: u32,
    /// Which QueueFlags the queue needs to be able to do its job
    flags: vk::QueueFlags,
    /// Does the queue need to be able to support presentation to a surface
    needs_present: bool,
}

impl QueueIndex {
    pub fn new(flags: vk::QueueFlags, needs_present: bool) -> Self {
        let saved_flags = if flags == vk::QueueFlags::TRANSFER {
            // Transfer is implied by the other two flags and is not mandatory to announce
            // https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkQueueFlagBits.html
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER
        } else {
            flags
        };
        Self {
            is_found: false,
            family_index: 0,
            index: 0,
            flags: saved_flags,
            needs_present,
        }
    }
    pub fn is_suitable(
        &self,
        device: vk::PhysicalDevice,
        surface_stuff: &SurfaceStuff,
        family_index: u32,
        properties: &vk::QueueFamilyProperties2,
    ) -> bool {
        // First we need to make sure the queue contains the flags we need
        if !properties
            .queue_family_properties
            .queue_flags
            .intersects(self.flags)
        {
            // We can early return if we don't
            return false;
        }

        // If the queue needs to be able to present then we can look that up
        if self.needs_present {
            return unsafe {
                surface_stuff
                    .surface_loader
                    .get_physical_device_surface_support(
                        device,
                        family_index,
                        surface_stuff.surface,
                    )
                    .expect("Failed to check device for presentation support")
            };
        }
        true
    }
    pub fn assign(&mut self, family_index: u32, queue_index: u32) {
        self.is_found = true;
        self.family_index = family_index;
        self.index = queue_index;
    }
}

impl PartialEq for QueueIndex {
    fn eq(&self, other: &Self) -> bool {
        self.family_index == other.family_index && self.index == other.index
    }
}

#[derive(Debug, Clone, Copy, Index)]
// Stores the family and index of each queue that we want
pub struct QueueIndices {
    // Graphics queues
    pub viewport: QueueIndex,
    // Ray, Physics, and Ocean are all compute queues
    pub ray: QueueIndex,
    pub physics: QueueIndex,
    pub ocean: QueueIndex,
    // Sync is the only transfer queue
    pub sync: QueueIndex,
}

impl QueueIndices {
    pub fn new() -> QueueIndices {
        QueueIndices {
            viewport: QueueIndex::new(vk::QueueFlags::GRAPHICS, true),
            ray: QueueIndex::new(vk::QueueFlags::COMPUTE, false),
            physics: QueueIndex::new(vk::QueueFlags::COMPUTE, false),
            ocean: QueueIndex::new(vk::QueueFlags::COMPUTE, false),
            sync: QueueIndex::new(vk::QueueFlags::TRANSFER, false),
        }
    }

    pub fn is_complete(&self) -> bool {
        // Check that every queue has its `is_found` field set
        for q in self {
            if !q.is_found {
                return false;
            }
        }
        true
    }
}

impl<'a> IntoIterator for &'a QueueIndices {
    type Item = QueueIndex;
    type IntoIter = QueueIndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        QueueIndicesIterator {
            queue_indices: self,
            index: 0,
        }
    }
}

pub struct QueueIndicesIterator<'a> {
    queue_indices: &'a QueueIndices,
    index: usize,
}

impl<'a> Iterator for QueueIndicesIterator<'a> {
    type Item = QueueIndex;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.queue_indices.len() {
            return None;
        }
        let result = self.queue_indices[self.index];
        self.index += 1;
        Some(result)
    }
}
