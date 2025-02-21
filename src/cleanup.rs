use std::collections::VecDeque;

/// Provide RAII style destructor for a struct building vulkan objects.
/// Requires pushing vulkan destructors as object are created.
/// e.g
//  cleanup.push(Box::new(|s: &Compute| unsafe {
//      s.device.destroy_descriptor_pool(s.descriptor_pool, None)
//  }));
pub struct Cleanup<'a, T> {
    parent: Option<&'a T>,
    deletion_queue: VecDeque<Box<dyn Fn(&T)>>, // TODO: Is this the best data type for this?
}

impl<'a, T> Cleanup<'a, T> {
    pub fn new() -> Self {
        Self {
            parent: None,
            deletion_queue: VecDeque::with_capacity(20),
        }
    }
    pub fn add_parent(&mut self, parent: &'a T) {
        self.parent = Some(parent);
    }
    /// Adds a destructor to the front of the destruction queue
    pub fn push(&mut self, f: Box<dyn Fn(&T)>) {
        self.deletion_queue.push_front(f);
    }
    pub fn cleanup(&mut self) {
        // We don't care if a parent is assigned if there is nothing to delete
        if self.deletion_queue.is_empty() {
            return;
        }
        match self.parent {
            Some(p) => {
                for v in self.deletion_queue.iter() {
                    v(p);
                }
                self.deletion_queue.clear();
            }
            None => {
                log::error!("Tried to perform a cleanup but no parent object was assigned")
            }
        }
    }
}

impl<'a, T> Drop for Cleanup<'a, T> {
    fn drop(&mut self) {
        self.cleanup();
    }
}
