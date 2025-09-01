use std::{
    path::PathBuf,
    sync::{RwLock, RwLockWriteGuard},
    time::Duration,
};

use anyhow::Result;
use cgmath::{Matrix4, Rad, SquareMatrix};
use hecs::Entity;

use crate::{
    material::Material,
    primitives::{
        model::{self, Model},
        Extrema, Primitive, AABB,
    },
    quaternion::Quaternion,
    structures::Cubemap,
    vec::Vec3,
};

pub const FOCAL_DISTANCE: f32 = 4.5;
pub const VFOV_DEG: f32 = 40.;
pub const DOF_SCALE: f32 = 0.0;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Event {
    PrePhysics,  // Allow the user to modify anything
    Physics,     // Compute collisions etc.
    PostPhysics, // Update UBO and Acceleration structure

    PreRay,
    Ray,
    PostRay,

    GraphicsUpdate,

    Input,
}

#[derive(Debug, Clone)]
pub struct Skybox {
    pub(crate) cubemap: Cubemap<PathBuf>,
}
impl Skybox {
    pub fn new(
        px: PathBuf,
        nx: PathBuf,
        py: PathBuf,
        ny: PathBuf,
        pz: PathBuf,
        nz: PathBuf,
    ) -> Self {
        Skybox {
            cubemap: Cubemap {
                px,
                nx,
                py,
                ny,
                pz,
                nz,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Collider {
    pub aabb: AABB,
    pub last_pos: Vec3,
    pub fixed: bool,
}
impl Collider {
    pub fn new<T: Extrema>(obj: &T, fixed: bool) -> Self {
        Self {
            aabb: AABB::new(obj),
            last_pos: Vec3::zero(),
            fixed,
        }
    }
}

#[derive(Debug)]
pub struct Orientation {
    pub pos: Vec3,
    pub quat: Quaternion,
    pub(crate) transformation: Matrix4<f32>,
}

impl Orientation {
    pub fn new(pos: Vec3, quat: Quaternion) -> Self {
        let transformation = Self::calc_transformation(pos, quat);
        Self {
            pos,
            quat,
            transformation,
        }
    }
    pub fn update(&mut self) {
        self.transformation = Self::calc_transformation(self.pos, self.quat);
    }
    fn calc_transformation(pos: Vec3, quat: Quaternion) -> Matrix4<f32> {
        // TODO: Calculate matrix from pos and quat
        let (axis, angle) = quat.to_axis_angle_rad();
        Matrix4::from_translation(pos.into()) * Matrix4::from_axis_angle(axis.into(), Rad(angle))
    }
}

impl Default for Orientation {
    fn default() -> Self {
        Self {
            pos: Vec3::zero(),
            quat: Quaternion::identity(),
            transformation: Matrix4::identity(),
        }
    }
}

/// Component of entities that are positioned relative to a parent entity
#[derive(Debug)]
pub struct Child {
    /// Parent entity
    pub parent: Entity,
    /// Converts child-relative coordinates to parent-relative coordinates
    pub offset_pos: Vec3,
    pub offset_quat: Quaternion,
}

#[derive(Debug)]
pub struct Instance {
    pub primitive: Entity,
    pub base_transform: Matrix4<f32>,
    pub initial_transform: Matrix4<f32>,
}

impl Instance {
    pub fn new(primitive: Entity) -> Self {
        Self {
            primitive,
            base_transform: Matrix4::identity(),
            initial_transform: Matrix4::identity(),
        }
    }
}

#[derive(Debug)]
pub struct Camera {
    pub vfov_rad: f32,
    pub focal_distance: f32,
    pub dof_scale: f32,
    pub enabled: u32,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            focal_distance: FOCAL_DISTANCE,
            vfov_rad: VFOV_DEG.to_radians(),
            dof_scale: DOF_SCALE,
            enabled: 0,
        }
    }
}

pub trait Bloomable {
    fn get_active_camera(&self) -> Option<Entity>; // How should we deal with no camera in the scene?
    fn get_physics_update_period(&self) -> Duration;
    /// Init Window is the only function that will be called before the object starts getting cloned and
    /// everythin that wants to be used by different functions will need Arc<RwLock<>>>
    fn init_window(&mut self, window: std::sync::Arc<RwLock<winit::window::Window>>);
    fn init(&mut self, world: &std::sync::Arc<RwLock<hecs::World>>) -> Result<()>;
    fn input(
        &mut self,
        event: winit::event::WindowEvent,
        world: &std::sync::Arc<RwLock<hecs::World>>,
    );
    fn raw_input(
        &mut self,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
        world: &std::sync::Arc<RwLock<hecs::World>>,
    );
    fn resize(&mut self, width: u32, height: u32, world: &std::sync::Arc<RwLock<hecs::World>>);
    fn display_tick(&mut self, delta_time: Duration, world: &std::sync::Arc<RwLock<hecs::World>>);
    fn physics_tick(&mut self, delta_time: Duration, world: &std::sync::Arc<RwLock<hecs::World>>);
}

pub fn import_gltf(world: &mut RwLockWriteGuard<hecs::World>, path: &str) -> Vec<Entity> {
    let mut imported_models = Vec::new();
    let (document, buffers, images) = gltf::import(path).unwrap(); // TODO: Error handling

    let mut textures = Vec::with_capacity(images.len());
    // Add textures to world
    for (i, _) in images.iter().enumerate() {
        textures.push(world.spawn((Material::from_model(PathBuf::from(path), i),)));
    }

    for m in document.meshes() {
        for p in m.primitives() {
            if p.mode() != gltf::mesh::Mode::Triangles {
                log::warn!(
                    "Primitive is of type {:?} instead of triangles. Unsure what to do.",
                    p.mode()
                );
            }
            // TODO: PBR
            let material_entity = if let Some(base_colour_texture) =
                p.material().pbr_metallic_roughness().base_color_texture()
            {
                // TODO: Figure out base_colour_texture.tex_coord()
                if base_colour_texture.texture().index() >= textures.len() {
                    log::warn!(
                        "Trying to index non-existant texture (index {} of {} textures)",
                        base_colour_texture.texture().index(),
                        textures.len()
                    );
                    continue;
                } else {
                    textures[base_colour_texture.texture().index()]
                    // TODO: Consider using sampler from file
                    // let sampler = base_colour_texture.texture().sampler();
                }
            } else {
                let colour = p.material().pbr_metallic_roughness().base_color_factor();
                world.spawn((Material::new_basic(
                    Vec3(colour[..3].try_into().unwrap()),
                    0.0,
                ),))
            };

            let r = p.reader(|buffer| Some(&buffers[buffer.index()]));

            let mut indices = Vec::new();
            if let Some(gltf::mesh::util::ReadIndices::U16(gltf::accessor::Iter::Standard(iter))) =
                r.read_indices()
            {
                for i in iter {
                    indices.push(i as u32);
                }
            }

            let mut positions = Vec::new();
            if let Some(iter) = r.read_positions() {
                for v in iter {
                    positions.push(v);
                }
            }
            let mut tex_coords = Vec::new();
            if let Some(gltf::mesh::util::ReadTexCoords::F32(gltf::accessor::Iter::Standard(
                iter,
            ))) = r.read_tex_coords(0)
            {
                for t in iter {
                    tex_coords.push(t);
                }
            }
            let mut normals = Vec::new();
            if let Some(iter) = r.read_normals() {
                for v in iter {
                    normals.push(v);
                }
            }

            if positions.len() != tex_coords.len() {
                log::error!(
                    "Primitive vertices don't all have a texture coordinate ({} vs {})",
                    positions.len(),
                    tex_coords.len()
                );
                continue;
            }
            if positions.len() != normals.len() {
                log::error!(
                    "Primitive vertices don't all have a normal ({} vs {})",
                    positions.len(),
                    tex_coords.len()
                );
                continue;
            }

            let mut vertices = Vec::with_capacity(positions.len());

            // Populate the vertices with positions and normals
            for (i, p) in positions.iter().enumerate() {
                vertices.push(model::Vertex::new(
                    Vec3(*p),
                    Vec3(normals[i]),
                    tex_coords[i],
                ));
            }

            let material_count = indices.len() / 3;
            let material_entities = vec![material_entity; material_count];

            let model = Model::new(vertices, indices, material_entities);

            imported_models.push(world.spawn((Primitive::Model(model),)));
        }
    }

    imported_models
}
