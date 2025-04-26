use crate::vec::Vec3;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Material {
    albedo: Vec3,
    alpha: f32,             // 0.0 = Transparent (Dielectric), 1.0 = Opaque
    refractive_index: f32,  // Relative from air into material (glass is ~1.0/1.5)
    smoothness: f32,        // 0.0 = Matte (Lambertian), 1.0 = Mirror (Specular)
    emissivity: f32,        // 0.0 = No emission, 1.0 = every ray will add light to the scene
    emission_strength: f32, // > 0
    emitted_colour: Vec3,
    _pad: [u32; 1],
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
            albedo,
            smoothness,
            alpha,
            refractive_index: refraction_index,
            emissivity,
            emission_strength,
            emitted_colour,
            _pad: [0; 1],
        }
    }
    #[allow(unused)]
    pub fn new_basic(albedo: Vec3, smoothness: f32) -> Self {
        let mut material = Material::default();
        material.albedo = albedo;
        material.smoothness = smoothness;
        material
    }
    #[allow(unused)]
    pub fn new_clear(albedo: Vec3) -> Self {
        let mut material = Material::default();
        material.albedo = albedo;
        material.alpha = 0.;
        material
    }
    #[allow(unused)]
    pub fn new_emissive(emitted_colour: Vec3, emission_strength: f32) -> Self {
        let mut material = Material::default();
        material.emissivity = 1.;
        material.emission_strength = emission_strength;
        material.emitted_colour = emitted_colour;
        material
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
            albedo: Vec3::new(1.0, 1.0, 1.0),
            smoothness: 0.0,
            alpha: 1.0,
            refractive_index: 1.4,
            emissivity: 0.0,
            emission_strength: 0.0,
            emitted_colour: Vec3::new(1., 1., 1.),
            _pad: [0; 1],
        }
    }
}
