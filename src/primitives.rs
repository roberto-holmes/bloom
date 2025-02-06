use bytemuck::Zeroable;

#[cfg(not(target_arch = "wasm32"))]
use rand::rngs::ThreadRng;

use crate::{
    algebra::Vec3, Material, MAX_MATERIAL_COUNT, MAX_QUAD_COUNT, MAX_SPHERE_COUNT,
    MAX_TRIANGLE_COUNT,
};

#[derive(Clone, Copy)]
pub enum ObjectType {
    Sphere = 0,
    Quad = 1,
    Triangle = 2,
}

pub trait Extrema {
    fn get_extrema(&self) -> (Vec3, Vec3);
    fn get_center(&self) -> Vec3;
}

pub struct Object {
    object_type: ObjectType,
    index: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    material: u32,
    pub is_selected: u32,
    _pad: [u32; 2],
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32, material: u32) -> Self {
        Self {
            center,
            radius,
            material,
            is_selected: 0,
            _pad: [0; 2],
        }
    }
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            center: Vec3::new(0., 0., 0.),
            radius: 0.5,
            material: 0,
            is_selected: 0,
            _pad: [0; 2],
        }
    }
}

impl Extrema for Sphere {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        let min = self.center - Vec3::all(self.radius);
        let max = self.center + Vec3::all(self.radius);
        (min, max)
    }
    fn get_center(&self) -> Vec3 {
        self.center
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Quad {
    q: Vec3,
    is_selected: u32,
    u: Vec3,
    _pad0: u32,
    v: Vec3,
    _pad1: u32,
    normal: Vec3,
    d: f32,
    w: Vec3,
    material: u32,
}

impl Quad {
    pub fn new(q: Vec3, u: Vec3, v: Vec3, material: u32) -> Self {
        let n = u.cross(&v);
        let normal = n.normalized();
        Self {
            q,
            u,
            v,
            is_selected: 0,
            material,
            normal,
            d: normal.dot(&q),
            w: n / (n.dot(&n)),
            _pad0: 0,
            _pad1: 0,
        }
    }
}

impl Default for Quad {
    fn default() -> Self {
        let q = Vec3::new(0., 0., 0.);
        let u = Vec3::new(0., 1., 0.);
        let v = Vec3::new(1., 1., 0.);
        let n = u.cross(&v);
        let normal = n.normalized();
        Self {
            q,
            u,
            v,
            normal,
            d: normal.dot(&q),
            w: n / (n.dot(&n)),
            material: 0,
            is_selected: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

impl Extrema for Quad {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        let min = self.q.min_extrema_4(
            &(self.q + self.u),
            &(self.q + self.v),
            &(self.q + self.u + self.v),
        );
        let max = self.q.max_extrema_4(
            &(self.q + self.u),
            &(self.q + self.v),
            &(self.q + self.u + self.v),
        );

        (min, max)
    }
    fn get_center(&self) -> Vec3 {
        self.q + 0.5 * (self.u + self.v)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle {
    a: Vec3,
    is_selected: u32,
    b: Vec3,
    material: u32,
    c: Vec3,
    _pad0: u32,
}

impl Triangle {
    pub fn new(a: Vec3, b: Vec3, c: Vec3, material: u32) -> Self {
        Self {
            a,
            b,
            c,
            is_selected: 0,
            material,
            // normal: (b - a).cross(&(c - a)).normalized(),
            _pad0: 0,
        }
    }
}

impl Default for Triangle {
    fn default() -> Self {
        Self::new(
            Vec3::new(0., 0., 0.),
            Vec3::new(1., 0., 1.),
            Vec3::new(1., 1., 0.),
            0,
        )
    }
}

impl Extrema for Triangle {
    fn get_extrema(&self) -> (Vec3, Vec3) {
        let min = self.a.min_extrema_3(&(self.b), &(self.c));
        let max = self.a.max_extrema_3(&(self.b), &(self.c));

        (min, max)
    }
    fn get_center(&self) -> Vec3 {
        // Centroid of a triangle
        (self.a + self.b + self.c) / 3.
    }
}

pub struct Scene {
    pub scene_vec: Vec<Object>,
    mat_arr: [Material; MAX_MATERIAL_COUNT],
    sphere_arr: [Sphere; MAX_SPHERE_COUNT],
    quad_arr: [Quad; MAX_QUAD_COUNT],
    triangle_arr: [Triangle; MAX_TRIANGLE_COUNT],
    last_material_index: usize,
    last_sphere_index: usize,
    last_quad_index: usize,
    last_triangle_index: usize,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            scene_vec: Vec::new(),
            mat_arr: [Material::default(); MAX_MATERIAL_COUNT],
            sphere_arr: [Sphere::zeroed(); MAX_SPHERE_COUNT],
            quad_arr: [Quad::zeroed(); MAX_QUAD_COUNT],
            triangle_arr: [Triangle::zeroed(); MAX_TRIANGLE_COUNT],
            last_material_index: 0,
            last_sphere_index: 0,
            last_quad_index: 0,
            last_triangle_index: 0,
        }
    }
    pub fn add_material(&mut self, mat: Material) -> u32 {
        self.last_material_index += 1; // Makes sure 0 is always available as the default
        if self.last_material_index >= MAX_MATERIAL_COUNT {
            log::warn!("Trying to add too many materials");
            return 0;
        }
        self.mat_arr[self.last_material_index] = mat;
        self.last_material_index as u32
    }
    pub fn add_sphere(&mut self, sphere: Sphere) {
        if self.last_sphere_index >= MAX_SPHERE_COUNT {
            log::warn!("Trying to add too many spheres");
            return;
        }
        self.sphere_arr[self.last_sphere_index] = sphere;
        self.scene_vec.push(Object {
            object_type: ObjectType::Sphere,
            index: self.last_sphere_index,
        });
        self.last_sphere_index += 1;
    }
    pub fn add_quad(&mut self, quad: Quad) {
        if self.last_quad_index >= MAX_QUAD_COUNT {
            log::warn!("Trying to add too many quads");
            return;
        }
        self.quad_arr[self.last_quad_index] = quad;
        self.scene_vec.push(Object {
            object_type: ObjectType::Quad,
            index: self.last_quad_index,
        });
        self.last_quad_index += 1;
    }
    pub fn add_triangle(&mut self, triangle: Triangle) {
        if self.last_quad_index >= MAX_QUAD_COUNT {
            log::warn!("Trying to add too many triangles");
            return;
        }
        self.triangle_arr[self.last_triangle_index] = triangle;
        self.scene_vec.push(Object {
            object_type: ObjectType::Triangle,
            index: self.last_triangle_index,
        });
        self.last_triangle_index += 1;
    }

    pub fn len(&self) -> usize {
        self.scene_vec.len()
    }

    pub fn get_material_arr(&self) -> &[Material; MAX_MATERIAL_COUNT] {
        &self.mat_arr
    }
    pub fn get_sphere_arr(&self) -> &[Sphere; MAX_SPHERE_COUNT] {
        &self.sphere_arr
    }
    pub fn get_sphere_arr_mut(&mut self) -> &mut [Sphere; MAX_SPHERE_COUNT] {
        &mut self.sphere_arr
    }
    pub fn get_quad_arr(&self) -> &[Quad; MAX_QUAD_COUNT] {
        &self.quad_arr
    }
    pub fn get_triangle_arr(&self) -> &[Triangle; MAX_TRIANGLE_COUNT] {
        &self.triangle_arr
    }

    pub fn get_extrema_of(&self, index: usize) -> (Vec3, Vec3) {
        let o = &self.scene_vec[index];
        match o.object_type {
            ObjectType::Quad => self.quad_arr[o.index].get_extrema(),
            ObjectType::Sphere => self.sphere_arr[o.index].get_extrema(),
            ObjectType::Triangle => self.triangle_arr[o.index].get_extrema(),
        }
    }

    pub fn get_type_of(&self, index: usize) -> ObjectType {
        self.scene_vec[index].object_type
    }

    pub fn get_index_of(&self, index: usize) -> usize {
        self.scene_vec[index].index
    }

    pub fn sort_x(&mut self, start: usize, end: usize) {
        self.scene_vec[start..end].sort_by(|o1, o2| {
            (match o1.object_type {
                ObjectType::Quad => self.quad_arr[o1.index].get_center().x(),
                ObjectType::Sphere => self.sphere_arr[o1.index].get_center().x(),
                ObjectType::Triangle => self.triangle_arr[o1.index].get_center().x(),
            })
            .partial_cmp(
                &(match o2.object_type {
                    ObjectType::Quad => self.quad_arr[o2.index].get_center().x(),
                    ObjectType::Sphere => self.sphere_arr[o2.index].get_center().x(),
                    ObjectType::Triangle => self.triangle_arr[o2.index].get_center().x(),
                }),
            )
            .expect("Sort x")
        });
    }

    pub fn sort_y(&mut self, start: usize, end: usize) {
        self.scene_vec[start..end].sort_by(|o1, o2| {
            (match o1.object_type {
                ObjectType::Quad => self.quad_arr[o1.index].get_center().y(),
                ObjectType::Sphere => self.sphere_arr[o1.index].get_center().y(),
                ObjectType::Triangle => self.triangle_arr[o1.index].get_center().y(),
            })
            .partial_cmp(
                &(match o2.object_type {
                    ObjectType::Quad => self.quad_arr[o2.index].get_center().y(),
                    ObjectType::Sphere => self.sphere_arr[o2.index].get_center().y(),
                    ObjectType::Triangle => self.triangle_arr[o2.index].get_center().y(),
                }),
            )
            .expect("Sort y")
        });
    }

    pub fn sort_z(&mut self, start: usize, end: usize) {
        self.scene_vec[start..end].sort_by(|o1, o2| {
            (match o1.object_type {
                ObjectType::Quad => self.quad_arr[o1.index].get_center().z(),
                ObjectType::Sphere => self.sphere_arr[o1.index].get_center().z(),
                ObjectType::Triangle => self.triangle_arr[o1.index].get_center().z(),
            })
            .partial_cmp(
                &(match o2.object_type {
                    ObjectType::Quad => self.quad_arr[o2.index].get_center().z(),
                    ObjectType::Sphere => self.sphere_arr[o2.index].get_center().z(),
                    ObjectType::Triangle => self.triangle_arr[o2.index].get_center().z(),
                }),
            )
            .expect("Sort z")
        });
    }

    #[cfg(target_arch = "wasm32")]
    pub fn get_random_material(&self, _rng: &mut u32) -> u32 {
        use web_sys::js_sys::Math::random;
        (random() as f32 * (self.last_material_index + 1) as f32) as u32
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_random_material(&self, rng: &mut ThreadRng) -> u32 {
        use rand::Rng;
        (rng.gen::<f32>() * (self.last_material_index + 1) as f32) as u32
    }
}
