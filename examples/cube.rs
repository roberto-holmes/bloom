use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Result;
use bloom::api::{Camera, Instance, Orientation, Skybox};
use bloom::material::Material;
use bloom::primitives::model::Model;
use bloom::primitives::Primitive;
use bloom::quaternion::Quaternion;
use bloom::vec::Vec3;
use bloom::{self, api::Bloomable};
use cgmath::{Matrix4, SquareMatrix};
use hecs::Entity;
use winit::window::Window;

fn main() {
    let demo = Demo::new();
    bloom::run(demo);
}

#[derive(Debug, Clone)]
struct Demo {
    camera: Arc<RwLock<Option<Entity>>>,
}

unsafe impl Send for Demo {}
unsafe impl Sync for Demo {}

impl Demo {
    fn new() -> Self {
        Self {
            camera: Arc::new(RwLock::new(None)),
        }
    }
}

impl Bloomable for Demo {
    fn get_active_camera(&self) -> Option<Entity> {
        *self.camera.read().unwrap()
    }
    fn get_physics_update_period(&self) -> Duration {
        Duration::from_millis(2)
    }
    fn init_window(&mut self, _: Arc<RwLock<Window>>) {}
    fn init(&mut self, world: &Arc<RwLock<hecs::World>>) -> Result<()> {
        let mut w = world.write().unwrap();
        // let green = w.spawn((Material::new_basic(Vec3::new(0.0, 1.0, 0.5), 0.0),));
        let tex = w.spawn((Material::new_textured(PathBuf::from("textures/statue.jpg")),));
        let cube = w.spawn((Primitive::Model(Model::new_cube(tex)?),));
        let _ = w.spawn((Instance {
            primitive: cube,
            base_transform: Matrix4::<f32>::identity(),
            initial_transform: Matrix4::<f32>::identity(),
        },));

        let _ = w.spawn((Skybox::new(
            PathBuf::from("textures/skybox/px.png"),
            PathBuf::from("textures/skybox/nx.png"),
            PathBuf::from("textures/skybox/py.png"),
            PathBuf::from("textures/skybox/ny.png"),
            PathBuf::from("textures/skybox/pz.png"),
            PathBuf::from("textures/skybox/nz.png"),
        ),));

        let camera = w.spawn((
            Camera::default(),
            Orientation::new(Vec3::new(0.0, 0.0, -5.0), Quaternion::identity()),
        ));
        self.camera = Arc::new(RwLock::new(Some(camera)));
        Ok(())
    }
    fn resize(&mut self, _width: u32, _height: u32, _world: &std::sync::Arc<RwLock<hecs::World>>) {}
    fn raw_input(
        &mut self,
        _: winit::event::DeviceId,
        _: winit::event::DeviceEvent,
        _: &std::sync::Arc<RwLock<hecs::World>>,
    ) {
    }

    fn input(&mut self, _: winit::event::WindowEvent, _: &std::sync::Arc<RwLock<hecs::World>>) {}
    fn display_tick(&mut self, _: Duration, _: &std::sync::Arc<RwLock<hecs::World>>) {}
    fn physics_tick(&mut self, _: Duration, _: &std::sync::Arc<RwLock<hecs::World>>) {}
}
