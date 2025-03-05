use cgmath::{dot, Matrix3, Vector3};
use winit::dpi::PhysicalPosition;

use crate::{core::UniformBufferObject, primitives::Sphere, MAX_SPHERE_COUNT};

struct Ray {
    origin: Vector3<f32>,
    direction: Vector3<f32>,
}

impl Ray {
    pub fn new(origin: &Vector3<f32>, direction: &Vector3<f32>) -> Self {
        Self {
            origin: *origin,
            direction: *direction,
        }
    }
}

impl Ray {
    pub fn distance_to(&self, t: f32) -> f32 {
        dot(t * self.direction, t * self.direction).sqrt()
    }
}

fn intersect_sphere(ray: &Ray, sphere: Sphere) -> f32 {
    let v = ray.origin - Into::<Vector3<f32>>::into(sphere.center);
    let a = dot(ray.direction, ray.direction);
    let b = dot(v, ray.direction);
    let c = dot(v, v) - sphere.radius * sphere.radius;

    // Find roots for the quadratic
    let d = b * b - a * c;

    // If no roots are found, the ray does not intersect with the sphere
    if d < 0. {
        return f32::MAX;
    }

    // If there is a real solution, find the time at which it takes place
    let sqrt_d = d.sqrt();
    let recip_a = 1. / a;
    let mb = -b;
    let t1 = (mb - sqrt_d) * recip_a;
    let t2 = (mb + sqrt_d) * recip_a;

    if t1 > 0. {
        return t1;
    } else if t2 > 0. {
        return t2;
    } else {
        // Check if the solution is for time = 0
        return f32::MAX;
    }
}

fn intersect_scene(ray: &Ray, scene: &[Sphere; MAX_SPHERE_COUNT]) -> (usize, f32) {
    let mut closest_hit: f32 = f32::MAX;
    let mut hit_object_num: usize = 0;
    for i in 0..scene.len() {
        if scene[i].radius <= 0. {
            continue;
        }
        // Loop through each object
        let hit = intersect_sphere(ray, scene[i]);
        if hit > 0. && hit < closest_hit {
            closest_hit = hit;
            hit_object_num = i;
        }
    }
    if closest_hit < f32::MAX {
        (hit_object_num, ray.distance_to(closest_hit))
    } else {
        (usize::MAX, f32::MAX)
    }
}

pub fn get_selected_object(
    pos: &PhysicalPosition<f64>,
    uniforms: &UniformBufferObject,
    scene: &[Sphere; MAX_SPHERE_COUNT],
) -> (usize, f32) {
    let mut u = (pos.x / (uniforms.width - 1) as f64) as f32;
    let mut v = (pos.y / (uniforms.height - 1) as f64) as f32;

    let viewport_scale_factor =
        2. * uniforms.camera.focal_distance * (uniforms.camera.vfov_rad / 2.).tan();

    u = (2. * u - 1.)
        * ((uniforms.width as f32) / (uniforms.height as f32))
        * viewport_scale_factor;
    v = (2. * v - 1.) * -1. * viewport_scale_factor;

    let camera_rotation = Matrix3::<f32> {
        x: uniforms.camera.u.into(),
        y: uniforms.camera.v.into(),
        z: uniforms.camera.w.into(),
    };
    let direction = camera_rotation * Vector3::new(u, v, uniforms.camera.focal_distance);

    let ray = Ray::new(&uniforms.camera.origin.into(), &direction);

    intersect_scene(&ray, scene)
}

#[allow(unused)]
pub fn clear_all_selections(scene: &mut [Sphere; MAX_SPHERE_COUNT]) {
    for object in scene {
        object.is_selected = 0;
    }
}

#[allow(unused)]
pub fn add_selection(object_num: usize, scene: &mut [Sphere; MAX_SPHERE_COUNT]) {
    scene[object_num].is_selected = 1;
}

#[allow(unused)]
pub fn remove_selection(object_num: usize, scene: &mut [Sphere; MAX_SPHERE_COUNT]) {
    scene[object_num].is_selected = 0;
}
