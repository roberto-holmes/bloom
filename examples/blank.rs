use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Result;
use bloom::{self, api::Bloomable};
use hecs::Entity;
use winit::window::Window;

fn main() {
    let demo = Demo::new();
    bloom::run(demo);
}

#[derive(Debug, Clone)]
struct Demo {}

unsafe impl Send for Demo {}
unsafe impl Sync for Demo {}

impl Demo {
    fn new() -> Self {
        Self {}
    }
}

impl Bloomable for Demo {
    fn get_active_camera(&self) -> Option<Entity> {
        None
    }
    fn get_physics_update_period(&self) -> Duration {
        Duration::from_millis(2)
    }
    fn init_window(&mut self, _: Arc<RwLock<Window>>) {}
    fn init(&mut self, _world: &Arc<RwLock<hecs::World>>) -> Result<()> {
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
