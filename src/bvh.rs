use std::u32;

use bytemuck::Zeroable;

use crate::{algebra::Vec3, primitives::Scene, MAX_OBJECT_COUNT};

const AABB_PADDING_SIZE: f32 = 0.0001;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AABB {
    // Axis Aligned Bounding Box
    min: Vec3,
    left_child_index: u32,
    max: Vec3,
    right_child_index: u32,
    object_type: u32, // 0 - Sphere, 1 - Quad
    object_index: u32,
    is_populated: u32,
    _pad: [u32; 1],
}

impl AABB {
    pub fn new((min, max): (Vec3, Vec3)) -> Self {
        Self {
            min,
            max,
            object_type: u32::MAX,
            object_index: u32::MAX,
            left_child_index: 0,
            right_child_index: 0,
            is_populated: 1,
            _pad: [0; 1],
        }
    }
    fn grow(&mut self, (min, max): (Vec3, Vec3)) {
        self.min = self.min.min_extrema(&min);
        self.max = self.max.max_extrema(&max);
    }
    fn sort_longest_axis(&self, scene: &mut Scene, start: usize, end: usize) {
        // Get longest axis
        let size = self.max - self.min;

        // Sort the relevant part scene by the longest axis so that we can split it by selecting the middle object
        if size.x() > size.y() && size.x() > size.z() {
            scene.sort_x(start, end);
        } else if size.y() > size.z() {
            scene.sort_y(start, end);
        } else {
            scene.sort_z(start, end);
        }
    }
    fn pad(&mut self, padding_size: f32) {
        // Make sure the box isn't smaller than padding size in any dimension
        let delta = self.max - self.min;
        if delta.x() < padding_size {
            self.min.set_x(self.min.x() - padding_size);
            self.max.set_x(self.max.x() + padding_size);
        }
        if delta.y() < padding_size {
            self.min.set_y(self.min.y() - padding_size);
            self.max.set_y(self.max.y() + padding_size);
        }
        if delta.z() < padding_size {
            self.min.set_z(self.min.z() - padding_size);
            self.max.set_z(self.max.z() + padding_size);
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct BVH {
    // Bounding Volume Hierarchy
    nodes: [AABB; 2 * MAX_OBJECT_COUNT - 1],
}

impl BVH {
    fn populate(&mut self, scene: &mut Scene, node_index: &mut usize, start: usize, end: usize) {
        // There is only one object left to sort so we will make this node a leaf
        if end - start == 1 {
            self.nodes[*node_index].object_type = scene.get_type_of(start) as u32;
            self.nodes[*node_index].object_index = scene.get_index_of(start) as u32;
            self.nodes[*node_index].pad(AABB_PADDING_SIZE);
            *node_index += 1;
            return;
        }

        for i in start..end {
            self.nodes[*node_index].grow(scene.get_extrema_of(i));
        }

        self.nodes[*node_index].sort_longest_axis(scene, start, end);

        let halfway_point = (end - start) / 2 + start;
        let parent_index = *node_index;
        *node_index += 1;

        // Deal with the left child
        self.nodes[parent_index].left_child_index = (*node_index) as u32;

        self.nodes[*node_index] = AABB::new(scene.get_extrema_of(start));
        self.populate(scene, node_index, start, halfway_point);

        // Deal with the right child
        self.nodes[parent_index].right_child_index = (*node_index) as u32;
        self.nodes[*node_index] = AABB::new(scene.get_extrema_of(halfway_point));
        self.populate(scene, node_index, halfway_point, end);
    }

    fn new(scene: &mut Scene) -> Self {
        let mut nodes = [AABB::zeroed(); 2 * MAX_OBJECT_COUNT - 1];

        nodes[0] = AABB::new(scene.get_extrema_of(0));

        let mut bvh = Self { nodes };
        let mut index = 0;
        bvh.populate(scene, &mut index, 0, scene.len());
        bvh
    }
}

pub fn create_bvh(scene: &mut Scene) -> [AABB; 2 * MAX_OBJECT_COUNT - 1] {
    BVH::new(scene).nodes
}
