use std::collections::HashSet;
use std::f32;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use anyhow::Result;
use bloom::api::{Instance, Orientation};
use bloom::material::Material;
use bloom::primitives::ocean::Ocean;
use bloom::quaternion::Quaternion;
use bloom::{
    self,
    api::{Bloomable, Camera},
    primitives::Primitive,
    vec::Vec3,
};
use hecs::Entity;
use winit::event::DeviceEvent;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

fn main() {
    let demo = Demo::new();
    bloom::run(demo);
}

#[derive(Debug, Clone)]
struct Demo {
    window: Option<Arc<RwLock<Window>>>,
    mouse_state: Arc<RwLock<MouseState>>,
    keyboard_state: Arc<RwLock<KeyboardState>>,
    is_mouse_grabbed: Arc<RwLock<bool>>,
    are_angles_updated: Arc<Mutex<bool>>,
    pitch_rad: Arc<Mutex<f32>>,
    yaw_rad: Arc<Mutex<f32>>,

    camera: Arc<RwLock<Option<Entity>>>,
}

unsafe impl Send for Demo {}
unsafe impl Sync for Demo {}

impl Demo {
    fn new() -> Self {
        Self {
            // api: None,
            window: None,
            camera: Arc::new(RwLock::new(None)),
            mouse_state: Arc::new(RwLock::new(MouseState::default())),
            keyboard_state: Arc::new(RwLock::new(KeyboardState::default())),
            is_mouse_grabbed: Arc::new(RwLock::new(false)),
            are_angles_updated: Arc::new(Mutex::new(false)),
            pitch_rad: Arc::new(Mutex::new(0.0)),
            yaw_rad: Arc::new(Mutex::new(0.0)),
        }
    }
    pub fn translate(&mut self, dx: f32, dz: f32, dy: f32, world: &Arc<RwLock<hecs::World>>) {
        let delta: Vec3 = Vec3::new(dx, dy, dz);

        let mut w = world.write().unwrap();
        // Get player orientation
        let player_ori = w
            .query_one_mut::<&mut Orientation>(self.camera.read().unwrap().unwrap())
            .unwrap();

        player_ori.pos += player_ori.quat.apply(delta);
    }
    pub fn look(&mut self, dx_rad: f32, dy_rad: f32, _world: &Arc<RwLock<hecs::World>>) {
        let mut yaw_rad = self.yaw_rad.lock().unwrap();
        let mut pitch_rad = self.pitch_rad.lock().unwrap();
        // Apply yaw to the player and pitch to the camera
        *yaw_rad += dx_rad;
        *pitch_rad += dy_rad;

        if *pitch_rad > std::f32::consts::FRAC_PI_2 {
            *pitch_rad = std::f32::consts::FRAC_PI_2;
        } else if *pitch_rad < -std::f32::consts::FRAC_PI_2 {
            *pitch_rad = -std::f32::consts::FRAC_PI_2;
        }

        if *yaw_rad > std::f32::consts::PI * 2.0 {
            *yaw_rad -= std::f32::consts::PI * 2.0;
        } else if *yaw_rad < -std::f32::consts::PI * 2.0 {
            *yaw_rad += std::f32::consts::PI * 2.0;
        }
        *self.are_angles_updated.lock().unwrap() = true;
    }
}

impl Bloomable for Demo {
    fn get_active_camera(&self) -> Option<Entity> {
        *self.camera.read().unwrap()
    }
    fn get_physics_update_period(&self) -> Duration {
        Duration::from_millis(2)
    }
    fn init_window(&mut self, window: Arc<RwLock<Window>>) {
        self.window = Some(window)
    }
    fn init(&mut self, world: &Arc<RwLock<hecs::World>>) -> Result<()> {
        let mut w = world.write().unwrap();

        let blu = w.spawn((Material::new_basic(Vec3::new(0.0, 0.0, 1.0), 0.),));

        // Spawn in ocean
        let o = w.spawn((Primitive::Ocean(Ocean::new(blu)),));
        let _ = w.spawn((Instance::new(o), Orientation::default()));

        let camera = w.spawn((Camera::default(), Orientation::default()));
        self.camera = Arc::new(RwLock::new(Some(camera)));

        Ok(())
    }
    fn resize(&mut self, _width: u32, _height: u32, _world: &std::sync::Arc<RwLock<hecs::World>>) {}
    fn raw_input(
        &mut self,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
        world: &std::sync::Arc<RwLock<hecs::World>>,
    ) {
        match event {
            DeviceEvent::MouseMotion { mut delta } => {
                delta.0 *= 0.01;
                delta.1 *= 0.01;
                if *self.is_mouse_grabbed.read().unwrap() {
                    let _ = self.look(delta.0 as f32, delta.1 as f32, world);
                }
            }
            _ => {}
        }
    }

    fn input(
        &mut self,
        event: winit::event::WindowEvent,
        _world: &std::sync::Arc<RwLock<hecs::World>>,
    ) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_state.write().unwrap().update_position(position);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.mouse_state
                    .write()
                    .unwrap()
                    .update_button(button, state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let _ = match delta {
                    MouseScrollDelta::PixelDelta(delta) => 0.001 * delta.y as f32,
                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                };
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyG),
                        ..
                    },
                ..
            } => {
                if let Some(win) = self.window.as_ref() {
                    match win.write() {
                        Ok(w) => {
                            let mut is_grabbed = self.is_mouse_grabbed.write().unwrap();
                            if *is_grabbed {
                                log::info!("Releasing mouse");
                                match w.set_cursor_grab(winit::window::CursorGrabMode::None) {
                                    Ok(()) => {}
                                    Err(e) => log::error!("Failed to release mouse: {e}"),
                                };
                                w.set_cursor_visible(true);
                            } else {
                                log::info!("Grabbing mouse");
                                match w.set_cursor_grab(winit::window::CursorGrabMode::Confined) {
                                    Ok(()) => {}
                                    Err(e) => log::error!("Failed to grab mouse: {e}"),
                                };
                                w.set_cursor_visible(false);
                            }
                            *is_grabbed = !*is_grabbed;
                        }
                        Err(e) => log::error!("Window is poisoned: {e}"),
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyF),
                        ..
                    },
                ..
            } => {
                if let Some(win) = self.window.as_ref() {
                    match win.write() {
                        Ok(w) => match w.fullscreen() {
                            None => {
                                w.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
                            }
                            Some(_) => w.set_fullscreen(None),
                        },
                        Err(e) => log::error!("Window is poisoned: {e}"),
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: s,
                        physical_key: PhysicalKey::Code(v),
                        ..
                    },
                ..
            } => {
                match s {
                    ElementState::Pressed => self.keyboard_state.write().unwrap().add(v),
                    ElementState::Released => self.keyboard_state.write().unwrap().remove(v),
                };
            }
            _ => (),
        }
    }
    fn display_tick(&mut self, _delta: Duration, _world: &std::sync::Arc<RwLock<hecs::World>>) {}
    fn physics_tick(&mut self, delta_time: Duration, world: &std::sync::Arc<RwLock<hecs::World>>) {
        let delta = 0.005 * delta_time.as_millis() as f32;
        let mut movement = (0.0, 0.0, 0.0);
        if self
            .keyboard_state
            .read()
            .unwrap()
            .is_pressed(KeyCode::KeyW)
        {
            movement.1 += delta;
        }
        if self
            .keyboard_state
            .read()
            .unwrap()
            .is_pressed(KeyCode::KeyS)
        {
            movement.1 -= delta;
        }
        if self
            .keyboard_state
            .read()
            .unwrap()
            .is_pressed(KeyCode::KeyA)
        {
            movement.0 -= delta;
        }
        if self
            .keyboard_state
            .read()
            .unwrap()
            .is_pressed(KeyCode::KeyD)
        {
            movement.0 += delta;
        }
        if self
            .keyboard_state
            .read()
            .unwrap()
            .is_pressed(KeyCode::Space)
        {
            movement.2 += delta;
        }
        if self
            .keyboard_state
            .read()
            .unwrap()
            .is_pressed(KeyCode::ControlLeft)
        {
            movement.2 -= delta;
        }
        self.translate(movement.0, movement.1, movement.2, world);
        let mut is_updated = self.are_angles_updated.lock().unwrap();
        if *is_updated == true {
            world
                .write()
                .unwrap()
                .query_one_mut::<&mut Orientation>(self.camera.read().unwrap().unwrap())
                .unwrap()
                .quat = Quaternion::from_euler(
                *self.pitch_rad.lock().unwrap(),
                0.0,
                *self.yaw_rad.lock().unwrap(),
            );
            *is_updated = false;
        }
    }
}

#[derive(Debug, Default)]
struct MouseState {
    left_pressed: bool,
    right_pressed: bool,
    middle_pressed: bool,
    forward_pressed: bool,
    backward_pressed: bool,

    click_position: PhysicalPosition<f64>,

    last_position: PhysicalPosition<f64>,
    current_position: PhysicalPosition<f64>,
}

impl MouseState {
    pub fn update_button(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Left => {
                self.left_pressed = state.is_pressed();
                if state.is_pressed() {
                    self.click_position = self.current_position
                }
            }
            MouseButton::Right => self.right_pressed = state.is_pressed(),
            MouseButton::Middle => self.middle_pressed = state.is_pressed(),
            MouseButton::Forward => self.forward_pressed = state.is_pressed(),
            MouseButton::Back => self.backward_pressed = state.is_pressed(),
            MouseButton::Other(v) => {
                log::warn!("Ignoring mouse button {}", v)
            }
        }
    }
    pub fn update_position(&mut self, position: PhysicalPosition<f64>) {
        self.last_position = self.current_position;
        self.current_position = position;
    }
}

#[derive(Debug, Default)]
struct KeyboardState {
    pressed_keys: HashSet<KeyCode>,
}

impl KeyboardState {
    fn add(&mut self, key: KeyCode) {
        // We don't care if it was already pressed
        let _ = self.pressed_keys.insert(key);
    }
    fn remove(&mut self, key: KeyCode) {
        // We don't care if it wasn't previously pressed
        let _ = self.pressed_keys.remove(&key);
    }
    fn is_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }
}
