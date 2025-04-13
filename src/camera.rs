use std::f32::consts::{FRAC_PI_2, PI};

use bytemuck::{Pod, Zeroable};
use cgmath::Quaternion;

use crate::{
    api::{DOF_SCALE, FOCAL_DISTANCE, VFOV_DEG},
    vec::Vec3,
};

#[derive(Debug, Copy, Clone)]
pub struct CameraOrientation {
    pub pos: Vec3,
    pub quat: Quaternion<f32>,
}

impl Default for CameraOrientation {
    fn default() -> Self {
        Self {
            pos: Vec3::default(),
            quat: Quaternion::new(0.0, 1.0, 0.0, 0.0),
        }
    }
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct CameraUniforms {
    pub origin: Vec3,
    pub focal_distance: f32,
    pub u: Vec3,
    pub vfov_rad: f32,
    pub v: Vec3,
    pub dof_scale: f32,
    pub w: Vec3,
    _padding0: u32, //  Vectors need to be aligned to 32 Bytes
}

#[derive(Debug)]
pub struct Camera {
    pub uniforms: CameraUniforms,
    center: Vec3,
    up: Vec3,
    distance: f32,
    azimuth: f32,
    altitude: f32,
}

impl Camera {
    pub fn with_spherical_coords(
        center: Vec3,
        up: Vec3,
        distance: f32,
        azimuth: f32,
        altitude: f32,
        focal_distance: f32,
        vfov_deg: f32,
        dof_scale: f32,
    ) -> Camera {
        let mut camera = Camera {
            uniforms: CameraUniforms::zeroed(),
            center,
            up,
            distance,
            azimuth,
            altitude,
        };
        camera.calculate_uniforms();
        camera.uniforms.focal_distance = focal_distance;
        camera.uniforms.vfov_rad = vfov_deg.to_radians();
        camera.uniforms.dof_scale = dof_scale;
        camera
    }

    fn calculate_uniforms(&mut self) {
        let w = {
            let (y, xz_scale) = self.altitude.sin_cos();
            let (x, z) = self.azimuth.sin_cos();
            -Vec3::new(x * xz_scale, y, z * xz_scale)
        };
        let origin = self.center - self.distance * w;
        let u = w.cross(&self.up).normalized();
        let v = u.cross(&w);
        self.uniforms.origin = origin;
        self.uniforms.u = u;
        self.uniforms.v = v;
        self.uniforms.w = w;
    }

    pub fn uniforms(&self) -> &CameraUniforms {
        &self.uniforms
    }

    pub fn zoom(&mut self, displacement: f32) {
        self.distance = (self.distance - displacement).max(0.0); // Prevent negative distance
        self.uniforms.origin = self.center - self.distance * self.uniforms.w;
    }

    pub fn pan(&mut self, du: f32, dv: f32) {
        let pan = du * self.uniforms.u + dv * self.uniforms.v;
        self.center += pan;
        self.uniforms.origin += pan;
    }

    pub fn orbit(&mut self, du: f32, dv: f32) {
        const MAX_ALT: f32 = FRAC_PI_2 - 1e-6;
        self.altitude = (self.altitude + dv).clamp(-MAX_ALT, MAX_ALT);
        self.azimuth += du;
        self.azimuth %= 2. * PI;
        self.calculate_uniforms();
    }

    #[allow(unused)]
    pub fn look_at(
        source: Vec3,
        dest: Vec3,
        up: Vec3,
        focal_distance: f32,
        vfov_deg: f32,
        dof_scale: f32,
    ) -> Camera {
        let center_to_origin = source - dest;
        let distance = center_to_origin.length().max(0.01); // Prevent distance of 0
        let neg_w = center_to_origin.normalized();
        let azimuth = neg_w.x().atan2(neg_w.z());
        let altitude = neg_w.y().asin();
        Self::with_spherical_coords(
            dest,
            up,
            distance,
            azimuth,
            altitude,
            focal_distance,
            vfov_deg,
            dof_scale,
        )
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera::look_at(
            Vec3::new(9., 6., 9.),
            Vec3::new(0., 0., 0.),
            Vec3::new(0., 1., 0.),
            FOCAL_DISTANCE,
            VFOV_DEG,
            DOF_SCALE,
        )
    }
}
