use std::collections::HashSet;
use std::f32;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Result;
use bloom::{
    self,
    api::{BloomAPI, Bloomable},
    material,
    primitives::{lentil::Lentil, model::Model, sphere::Sphere, Primitive},
    quaternion::Quaternion,
    vec::Vec3,
};
use cgmath::Matrix4;
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

struct Demo {
    api: Option<BloomAPI>,
    window: Option<Arc<RwLock<Window>>>,
    mouse_state: MouseState,
    keyboard_state: KeyboardState,

    is_mouse_grabbed: bool,
    is_quaternion_updated: bool,

    pitch_rad: f32,
    yaw_rad: f32,

    pos: Vec3,
    quat: Quaternion,
}

impl Demo {
    fn new() -> Self {
        Self {
            api: None,
            window: None,
            mouse_state: MouseState::default(),
            keyboard_state: KeyboardState::default(),
            is_mouse_grabbed: false,
            is_quaternion_updated: false,

            pitch_rad: 0.0,
            yaw_rad: 0.0,

            pos: Vec3::default(),
            quat: Quaternion::identity(),
        }
    }
    fn get_api(&mut self) -> &mut BloomAPI {
        self.api.as_mut().unwrap()
    }
    pub fn translate(&mut self, dx: f32, dy: f32) {
        let delta = Vec3::new(dx, 0.0, dy);
        let quat = Quaternion::from_euler(0.0, 0.0, self.yaw_rad);
        self.pos += quat.apply(delta);
        self.api.as_ref().unwrap().update_camera_position(self.pos);
    }
    pub fn look(&mut self, dx_rad: f32, dy_rad: f32) {
        self.yaw_rad += dx_rad;
        self.pitch_rad += dy_rad;

        if self.pitch_rad > std::f32::consts::FRAC_PI_2 {
            self.pitch_rad = std::f32::consts::FRAC_PI_2;
        } else if self.pitch_rad < -std::f32::consts::FRAC_PI_2 {
            self.pitch_rad = -std::f32::consts::FRAC_PI_2;
        }

        if self.yaw_rad > std::f32::consts::PI * 2.0 {
            self.yaw_rad -= std::f32::consts::PI * 2.0;
        } else if self.yaw_rad < -std::f32::consts::PI * 2.0 {
            self.yaw_rad += std::f32::consts::PI * 2.0;
        }
        log::debug!(
            "looking by pitch:{:.2}° [{dy_rad}], yaw:{:.2}° [{dx_rad}]",
            self.pitch_rad.to_degrees(),
            self.yaw_rad.to_degrees()
        );
        self.quat = Quaternion::from_euler(self.pitch_rad, 0.0, self.yaw_rad);

        self.is_quaternion_updated = true;
    }
}

impl Bloomable for Demo {
    fn init_window(&mut self, window: Arc<RwLock<Window>>) {
        self.window = Some(window)
    }
    fn init(&mut self, api_in: BloomAPI) -> Result<()> {
        self.api = Some(api_in);
        let scene = self.get_api();
        // let mut api = scene.lock().unwrap();

        // let scene = &mut api.scene;

        let red =
            scene.add_material(material::Material::new_basic(Vec3::new(1.0, 0.0, 0.0), 0.))?;
        let grey =
            scene.add_material(material::Material::new_basic(Vec3::new(0.7, 0.7, 0.7), 0.1))?;
        let yellow =
            scene.add_material(material::Material::new_basic(Vec3::new(1.0, 0.8, 0.0), 0.))?;
        let glass = scene.add_material(material::Material::new_clear(Vec3::new(0.9, 1.0, 1.0)))?;
        let mirror =
            scene.add_material(material::Material::new_basic(Vec3::new(1.0, 1.0, 0.9), 1.0))?;
        let light = scene.add_material(material::Material::new_emissive(
            Vec3::new(0.9, 0.3, 0.1),
            1.0,
        ))?;

        let (document, buffers, _) = gltf::import("models/Duck.glb").unwrap();
        for m in document.meshes() {
            for p in m.primitives() {
                let m = Model::new_gltf_primitive(p, &buffers, yellow);
                let m_id = scene.add_obj(Primitive::Model(m))?;
                let _ = scene.add_instance(
                    m_id,
                    Matrix4::<f32>::from_translation(cgmath::Vector3::new(-50.0, -1.0, 0.0))
                        * Matrix4::<f32>::from_scale(1.0 / 100.0),
                );
            }
        }
        log::info!("Material has id {yellow}");
        let cube = Model::new_cube(red)?;
        let cube_id = scene.add_obj(Primitive::Model(cube))?;

        let plane_id = scene.add_obj(Primitive::Model(Model::new_plane(grey)?))?;
        let mirror_id = scene.add_obj(Primitive::Model(Model::new_mirror(mirror)?))?;

        let _ = scene.add_instance(
            cube_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: 2.0,
                y: 3.0,
                z: -1.0,
            }),
        )?;
        let _ = scene.add_instance(
            plane_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            }) * cgmath::Matrix4::from_nonuniform_scale(1.0, 1.0, 15.0),
        );
        let _ = scene.add_instance(
            mirror_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: 0.0,
                y: 0.0,
                z: -7.0,
            }) * cgmath::Matrix4::from_nonuniform_scale(3.0, 5.0, 0.1),
        );
        let _ = scene.add_instance(
            mirror_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: 0.0,
                y: 0.0,
                z: 7.0,
            }) * cgmath::Matrix4::from_nonuniform_scale(3.0, 5.0, 0.1),
        );

        let sphere = Sphere::new(1.0, light).unwrap();
        let sphere_id = scene.add_obj(Primitive::Sphere(sphere))?;
        let _ = scene.add_instance(
            sphere_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: -1.0,
                y: 5.0,
                z: 3.0,
            }),
        );

        let lentil = Lentil::new(1.0, 1.0, glass).unwrap();
        let lentil_id = scene.add_obj(Primitive::Lentil(lentil))?;
        let _ = scene.add_instance(
            lentil_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: 2.0,
                y: 0.0,
                z: 0.0,
            }),
        );
        let _ = scene.add_instance(
            lentil_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: -2.0,
                y: 0.0,
                z: 0.0,
            }),
        );
        #[rustfmt::skip]
        let _ = scene.add_instance(
            lentil_id,
            Matrix4::from_translation(cgmath::Vector3 {
                    x: 2.0,
                    y: 0.0,
                    z: -3.0,
                }) * cgmath::Matrix4::new(
                    0.9182397, 0.3893702,-0.0722967, 0.0,
                   -0.1784285, 0.5697361, 0.8022245, 0.0,
                   -0.3535523,-0.7237346, 0.5926290 , 0.0,
                    0.0      , 0.0      , 0.0      , 1.0,
                ),
        );
        Ok(())
    }
    fn resize(&mut self, _width: u32, _height: u32) {}
    fn input(&mut self, event: winit::event::WindowEvent) {
        // let api = self.get_api();
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_state.update_position(position);
                let (mut dx, mut dy) = self.mouse_state.get_pos_delta();
                dx *= 0.01;
                dy *= 0.01;

                if self.is_mouse_grabbed {
                    let _ = self.look(dx, dy);
                }
                // if self.mouse_state.left_pressed {
                // scene.api.camera.orbit(dx, dy);
                // api.uniform.reset_samples();
                // }
                // else if self.mouse_state.middle_pressed {
                //     api.camera.pan(dx, dy);
                //     api.uniform.reset_samples();
                // } else if self.mouse_state.right_pressed {
                //     api.camera.zoom(-dy);
                //     api.uniform.reset_samples();
                // }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.mouse_state.update_button(button, state);

                // if button == MouseButton::Left
                //     && !state.is_pressed()
                //     && self.mouse_state.get_click_delta() < 5.0
                // {
                //     let (hit_object, dist_to_object) = select::get_selected_object(
                //         &self.mouse_state.current_position,
                //         &api.uniform,
                //         &api.scene.get_sphere_arr(),
                //     );
                //     if hit_object == usize::MAX {
                //         api.camera.uniforms.dof_scale = 0.;
                //     } else {
                //         api.camera.uniforms.focal_distance = dist_to_object;
                //         api.camera.uniforms.dof_scale = DOF_SCALE;
                //     }
                //     api.uniform.reset_samples();
                // }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let _ = match delta {
                    MouseScrollDelta::PixelDelta(delta) => 0.001 * delta.y as f32,
                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                };
                // api.camera.zoom(delta);
                // api.uniform.reset_samples();
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
                            if self.is_mouse_grabbed {
                                log::info!("Releasing mouse");
                                // match w.set_cursor_grab(winit::window::CursorGrabMode::None) {
                                //     Ok(()) => {}
                                //     Err(e) => log::error!("Failed to ungrab mouse: {e}"),
                                // };
                                w.set_cursor_visible(true);
                            } else {
                                log::info!("Grabbing mouse");
                                // match w.set_cursor_grab(winit::window::CursorGrabMode::Confined) {
                                //     Ok(()) => {}
                                //     Err(e) => log::error!("Failed to grab mouse: {e}"),
                                // };
                                w.set_cursor_visible(false);
                            }
                            self.is_mouse_grabbed = !self.is_mouse_grabbed;
                        }
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
                    ElementState::Pressed => self.keyboard_state.add(v),
                    ElementState::Released => self.keyboard_state.remove(v),
                };
            }
            _ => (),
        }
    }
    fn display_tick(&mut self) {}
    fn physics_tick(&mut self, delta_time: Duration) {
        let delta = 0.001 * delta_time.as_millis() as f32;
        let mut movement = (0.0, 0.0);
        if self.keyboard_state.is_pressed(KeyCode::KeyW) {
            movement.1 += delta;
        }
        if self.keyboard_state.is_pressed(KeyCode::KeyS) {
            movement.1 -= delta;
        }
        if self.keyboard_state.is_pressed(KeyCode::KeyA) {
            movement.0 -= delta;
        }
        if self.keyboard_state.is_pressed(KeyCode::KeyD) {
            movement.0 += delta;
        }
        // We only need to actually update the position if we've (tried to) move
        if movement.0 != 0.0 || movement.1 != 0.0 {
            self.translate(movement.0, movement.1);
        }
        if self.is_quaternion_updated == true {
            self.api
                .as_ref()
                .unwrap()
                .update_camera_quaternion(self.quat);
            self.is_quaternion_updated = false;
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
    pub fn get_pos_delta(&self) -> (f32, f32) {
        (
            (self.current_position.x - self.last_position.x) as f32,
            (self.current_position.y - self.last_position.y) as f32,
        )
    }
    pub fn get_click_delta(&self) -> f32 {
        // We just need an approximation to decide if the mouse has moved too much to ignore the click
        ((self.current_position.x - self.click_position.x).abs()
            + (self.current_position.y - self.click_position.y).abs()) as f32
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
