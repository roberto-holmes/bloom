use std::path::PathBuf;

use crate::vec::Vec3;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct MaterialData {
    albedo: Vec3,
    alpha: f32,             // 0.0 = Transparent (Dielectric), 1.0 = Opaque
    refractive_index: f32,  // Relative from air into material (glass is ~1.0/1.5)
    smoothness: f32,        // 0.0 = Matte (Lambertian), 1.0 = Mirror (Specular)
    emissivity: f32,        // 0.0 = No emission, 1.0 = every ray will add light to the scene
    emission_strength: f32, // > 0
    emitted_colour: Vec3,
    texture_index: u32,
}

pub struct Material {
    data: MaterialData,
    pub(crate) texture_path: Option<PathBuf>,
}

impl Material {
    pub fn new(
        albedo: Vec3,
        smoothness: f32,
        alpha: f32,
        refraction_index: f32,
        emissivity: f32,
        emission_strength: f32,
        emitted_colour: Vec3,
    ) -> Self {
        Self {
            texture_path: None,
            data: MaterialData {
                albedo,
                smoothness,
                alpha,
                refractive_index: refraction_index,
                emissivity,
                emission_strength,
                emitted_colour,
                texture_index: 0,
            },
        }
    }
    #[allow(unused)]
    pub fn new_basic(albedo: Vec3, smoothness: f32) -> Self {
        let mut material = Material::default();
        material.data.albedo = albedo;
        material.data.smoothness = smoothness;
        material
    }
    #[allow(unused)]
    pub fn new_clear(albedo: Vec3) -> Self {
        let mut material = Material::default();
        material.data.albedo = albedo;
        material.data.alpha = 0.;
        material
    }
    #[allow(unused)]
    pub fn new_emissive(emitted_colour: Vec3, emission_strength: f32) -> Self {
        let mut material = Material::default();
        material.data.emissivity = 1.;
        material.data.emission_strength = emission_strength;
        material.data.emitted_colour = emitted_colour;
        material
    }
    #[allow(unused)]
    pub fn new_textured(path: PathBuf) -> Self {
        let mut material = Material::default();
        material.texture_path = Some(path);
        material
    }
    pub(crate) fn get_data(&self) -> MaterialData {
        // TODO: how textures?
        self.data
    }
    pub(crate) fn update_texture_index(&mut self, texture_index: u32) {
        self.data.texture_index = texture_index;
    }
    // pub fn new_random() -> Self {
    //     Self::new(
    //         Vec3::new(
    //             get_random(&mut self.rng) as f32,
    //             get_random(&mut self.rng) as f32,
    //             get_random(&mut self.rng) as f32,
    //         )
    //         .normalized(),
    //         get_random(&mut self.rng) as f32,
    //         get_random(&mut self.rng) as f32,
    //         1. / 1.5,
    //         get_random(&mut self.rng) as f32,
    //         get_random(&mut self.rng) as f32,
    //         Vec3::new(
    //             get_random(&mut self.rng) as f32,
    //             get_random(&mut self.rng) as f32,
    //             get_random(&mut self.rng) as f32,
    //         ),
    //     )
    // }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            texture_path: None,
            data: MaterialData {
                albedo: Vec3::new(1.0, 1.0, 1.0),
                smoothness: 0.0,
                alpha: 1.0,
                refractive_index: 1.4,
                emissivity: 0.0,
                emission_strength: 0.0,
                emitted_colour: Vec3::new(1., 1., 1.),
                texture_index: 0,
            },
        }
    }
}
