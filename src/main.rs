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
use cgmath::{Matrix4, SquareMatrix};
use maze_generator::prelude::Generator;
use rand::Rng;
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

struct Demo {
    api: Option<BloomAPI>,
    window: Option<Arc<RwLock<Window>>>,
    mouse_state: MouseState,
    keyboard_state: KeyboardState,

    is_mouse_grabbed: bool,
    are_angles_updated: bool,

    pitch_rad: f32,
    yaw_rad: f32,

    goal: u64,
    goal_pos: Matrix4<f32>,
    goal_scale: Matrix4<f32>,
    goal_yaw_rad: f32,

    lentil: u64,
    lentil_position: cgmath::Matrix4<f32>,
}

impl Demo {
    fn new() -> Self {
        Self {
            api: None,
            window: None,
            mouse_state: MouseState::default(),
            keyboard_state: KeyboardState::default(),
            is_mouse_grabbed: false,
            are_angles_updated: false,

            pitch_rad: 0.0,
            yaw_rad: 0.0,

            goal: 0,
            goal_pos: Matrix4::<f32>::identity(),
            goal_scale: Matrix4::<f32>::from_scale(1.0 / 100.0),
            goal_yaw_rad: 0.0,

            lentil: 0,
            lentil_position: cgmath::Matrix4::from_translation(cgmath::Vector3 {
                x: 5.0,
                y: 0.0,
                z: 0.0,
            }),
        }
    }
    fn get_api(&mut self) -> &mut BloomAPI {
        self.api.as_mut().unwrap()
    }
    pub fn translate(&mut self, dx: f32, dz: f32, dy: f32) {
        let delta = Vec3::new(dx, dy, dz);
        // let quat = Quaternion::from_euler(0.0, 0.0, self.yaw_rad);
        // self.pos += quat.apply(delta);
        // self.api.as_ref().unwrap().update_camera_position(self.pos);
        self.api.as_ref().unwrap().update_camera_position(delta);
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
        self.are_angles_updated = true;
    }
}

impl Bloomable for Demo {
    fn init_window(&mut self, window: Arc<RwLock<Window>>) {
        self.window = Some(window)
    }
    fn init(&mut self, api_in: BloomAPI) -> Result<()> {
        self.api = Some(api_in);
        let scene = self.get_api();

        let red =
            scene.add_material(material::Material::new_basic(Vec3::new(1.0, 0.0, 0.0), 0.))?;
        let grey =
            scene.add_material(material::Material::new_basic(Vec3::new(0.7, 0.7, 0.7), 0.1))?;
        let yellow =
            scene.add_material(material::Material::new_basic(Vec3::new(1.0, 0.8, 0.0), 0.))?;
        let glass = scene.add_material(material::Material::new_clear(Vec3::new(0.9, 1.0, 1.0)))?;
        let mirror = scene.add_material(material::Material::new_basic(
            Vec3::new(1.0, 1.0, 1.0),
            // 0.5,
            0.995,
        ))?;
        let light = scene.add_material(material::Material::new_emissive(
            Vec3::new(0.9, 0.3, 0.1),
            1.0,
        ))?;
        let goal_colour = scene.add_material(material::Material::new_emissive(
            Vec3::new(1.0, 1.0, 1.0),
            1.0,
        ))?;

        let character_material = scene.add_material(material::Material::new(
            Vec3::new(1.0, 1.0, 1.0),
            0.0,
            0.0,
            1.4,
            1.0,
            0.1,
            Vec3::new(0.1, 1.0, 0.1),
        ))?;

        let scale = 5.0;
        let width = 7;
        let height = 7;

        let mut duck = 0;
        let mut goal = 0;
        let goal_position = Matrix4::<f32>::from_translation(cgmath::Vector3::new(
            (width as f32 / 2.0 - 0.5) * scale,
            0.0,
            (height as f32 / 2.0 - 0.5) * scale,
        ));

        let (document, buffers, _) = gltf::import("models/Duck.glb").unwrap();
        for m in document.meshes() {
            for p in m.primitives() {
                let m = Model::new_gltf_primitive(p.clone(), &buffers, character_material);
                let m_id = scene.add_obj(Primitive::Model(m))?;
                duck = scene.add_instance(
                    m_id,
                    Matrix4::<f32>::from_translation(cgmath::Vector3::new(0.0, 0.0, 0.0))
                        * Matrix4::<f32>::from_scale(1.0 / 100.0),
                )?;
                let g = Model::new_gltf_primitive(p, &buffers, goal_colour);
                let g_id = scene.add_obj(Primitive::Model(g))?;
                goal = scene.add_instance(
                    g_id,
                    goal_position * Matrix4::<f32>::from_scale(1.0 / 100.0),
                )?;
            }
        }

        scene.assign_camera_to(
            duck,
            Vec3::new(0.0, 1.5, 0.8),
            Matrix4::<f32>::from_translation(cgmath::Vector3::new(0.0, -1.0, 0.0))
                * Matrix4::from_angle_y(cgmath::Rad(-std::f32::consts::FRAC_PI_2))
                * Matrix4::from_scale(1.0 / 100.0),
        )?;

        let plane_id = scene.add_obj(Primitive::Model(Model::new_plane(grey)?))?;
        let blank_mirror_id = scene.add_obj(Primitive::Model(Model::new_mirror(grey)?))?;
        let mirror_id = scene.add_obj(Primitive::Model(Model::new_mirror(mirror)?))?;

        // Create a maze
        // Each maze unit is 10 of our units
        let mut generator =
            maze_generator::recursive_backtracking::RbGenerator::new(Some(rand::random()));

        let maze = generator.generate(width, height).unwrap();

        let north = maze_generator::prelude::Direction::North;
        let south = maze_generator::prelude::Direction::South;
        let east = maze_generator::prelude::Direction::East;
        let west = maze_generator::prelude::Direction::West;

        // Create the floor
        let _ = scene.add_instance(
            plane_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            }) * cgmath::Matrix4::from_nonuniform_scale(
                width as f32 * scale / 2.0,
                1.0,
                height as f32 * scale / 2.0,
            ),
        );

        log::debug!("Creating maze \n{:?}", maze);

        // let mirror_scale = cgmath::Matrix4::from_nonuniform_scale(scale as f32, scale as f32, 0.1);

        // Loop through every tile of the maze
        for maze_y in 0..height {
            for maze_x in 0..width {
                let coords = maze_generator::prelude::Coordinates::new(maze_x, maze_y);
                let tile = maze.get_field(&coords).unwrap();
                // For each cardinal direction, add a wall if necessary
                // 0, 0 in maze coordinates maps to -(width/2-0.5) * scale), -(height/2-0.5) * scale)
                let x = (maze_x as f32 - (width as f32 / 2.0 - 0.5)) * scale;
                let z = (maze_y as f32 - (height as f32 / 2.0 - 0.5)) * scale;

                let id = if rand::random_bool(0.3) {
                    blank_mirror_id
                } else {
                    mirror_id
                };

                if !tile.has_passage(&north) {
                    let _ = scene.add_instance(
                        id,
                        Matrix4::from_translation(cgmath::Vector3 {
                            x: x,
                            y: 0.0,
                            z: z - scale / 2.0,
                        }) * cgmath::Matrix4::from_nonuniform_scale(scale, scale, 0.1),
                    );
                }
                if !tile.has_passage(&south) {
                    let _ = scene.add_instance(
                        id,
                        Matrix4::from_translation(cgmath::Vector3 {
                            x: x,
                            y: 0.0,
                            z: z + scale / 2.0,
                        }) * cgmath::Matrix4::from_nonuniform_scale(scale, scale, 0.1),
                    );
                }
                if !tile.has_passage(&east) {
                    let _ = scene.add_instance(
                        id,
                        Matrix4::from_translation(cgmath::Vector3 {
                            x: x + scale / 2.0,
                            y: 0.0,
                            z: z,
                        }) * cgmath::Matrix4::from_nonuniform_scale(0.1, scale, scale),
                    );
                }
                if !tile.has_passage(&west) {
                    let _ = scene.add_instance(
                        id,
                        Matrix4::from_translation(cgmath::Vector3 {
                            x: x - scale / 2.0,
                            y: 0.0,
                            z: z,
                        }) * cgmath::Matrix4::from_nonuniform_scale(0.1, scale, scale),
                    );
                }
            }
        }

        // let _ = scene.add_instance(
        //     mirror_id,
        //     Matrix4::from_translation(cgmath::Vector3 {
        //         x: 0.0,
        //         y: 0.0,
        //         z: -7.0,
        //     }) * cgmath::Matrix4::from_nonuniform_scale(5.0, 5.0, 0.1),
        // );
        // let _ = scene.add_instance(
        //     mirror_id,
        //     Matrix4::from_translation(cgmath::Vector3 {
        //         x: -2.5,
        //         y: 0.0,
        //         z: -4.5,
        //     }) * Matrix4::from_angle_y(cgmath::Rad(-std::f32::consts::FRAC_PI_2))
        //         * cgmath::Matrix4::from_nonuniform_scale(5.0, 5.0, 0.1),
        // );
        // let _ = scene.add_instance(
        //     mirror_id,
        //     Matrix4::from_translation(cgmath::Vector3 {
        //         x: 0.0,
        //         y: 0.0,
        //         z: 3.0,
        //     }) * cgmath::Matrix4::from_nonuniform_scale(3.0, 5.0, 0.1),
        // );

        let sphere = Sphere::new(1.0, light).unwrap();
        let sphere_id = scene.add_obj(Primitive::Sphere(sphere))?;
        let _ = scene.add_instance(
            sphere_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: -1.0,
                y: 20.0,
                z: 3.0,
            }),
        );

        let lentil = Lentil::new(1.0, 1.0, glass).unwrap();
        let lentil_id = scene.add_obj(Primitive::Lentil(lentil))?;
        // let _ = scene.add_instance(
        //     lentil_id,
        //     Matrix4::from_translation(cgmath::Vector3 {
        //         x: 2.0,
        //         y: 0.0,
        //         z: 0.0,
        //     }),
        // );
        // let _ = scene.add_instance(
        //     lentil_id,
        //     Matrix4::from_translation(cgmath::Vector3 {
        //         x: -2.0,
        //         y: 0.0,
        //         z: 0.0,
        //     }),
        // );
        // #[rustfmt::skip]
        // let _ = scene.add_instance(
        //     lentil_id,
        //     Matrix4::from_translation(cgmath::Vector3 {
        //             x: 2.0,
        //             y: 0.0,
        //             z: -3.0,
        //         }) * cgmath::Matrix4::new(
        //             0.9182397, 0.3893702,-0.0722967, 0.0,
        //            -0.1784285, 0.5697361, 0.8022245, 0.0,
        //            -0.3535523,-0.7237346, 0.5926290 , 0.0,
        //             0.0      , 0.0      , 0.0      , 1.0,
        //         ),
        // );
        self.lentil = scene.add_instance(
            lentil_id,
            cgmath::Matrix4::from_translation(cgmath::Vector3 {
                x: 5.0,
                y: 0.0,
                z: 0.0,
            }),
        )?;

        self.goal = goal;
        self.goal_pos = goal_position;
        Ok(())
    }
    fn resize(&mut self, _width: u32, _height: u32) {}
    fn raw_input(&mut self, _device_id: winit::event::DeviceId, event: winit::event::DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { mut delta } => {
                delta.0 *= 0.01;
                delta.1 *= 0.01;
                if self.is_mouse_grabbed {
                    let _ = self.look(delta.0 as f32, delta.1 as f32);
                }
            }
            _ => {}
        }
    }

    fn input(&mut self, event: winit::event::WindowEvent) {
        // let api = self.get_api();
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_state.update_position(position);
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
                            self.is_mouse_grabbed = !self.is_mouse_grabbed;
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
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyQ),
                        ..
                    },
                ..
            } => {
                let green = match self
                    .get_api()
                    .add_material(material::Material::new_basic(Vec3::new(0.0, 1.0, 0.0), 0.))
                {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("Failed to add material: {e}");
                        return;
                    }
                };
                log::warn!("Green is ID {green}");
                let cube = match Model::new_cube(green) {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("Failed to create cube: {e}");
                        return;
                    }
                };
                let cube_id = match self.get_api().add_obj(Primitive::Model(cube)) {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("Failed to add model: {e}");
                        return;
                    }
                };
                let mut rng = rand::rng();
                if let Err(e) = self.get_api().add_instance(
                    cube_id,
                    Matrix4::from_translation(cgmath::Vector3 {
                        x: rng.random_range(-10.0..10.0),
                        y: rng.random_range(-10.0..10.0),
                        z: rng.random_range(-10.0..10.0),
                    }),
                ) {
                    log::error!("Failed to add cube: {e}");
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyZ),
                        ..
                    },
                ..
            } => {
                let l = self.lentil;
                self.lentil_position.w.y += 0.05;
                let pos = self.lentil_position;
                if let Err(e) = self.get_api().move_instance_to(l, pos) {
                    log::error!("Failed to move lentil up: {e}");
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyX),
                        ..
                    },
                ..
            } => {
                let l = self.lentil;
                self.lentil_position.w.y -= 0.05;
                let pos = self.lentil_position;
                if let Err(e) = self.get_api().move_instance_to(l, pos) {
                    log::error!("Failed to move lentil down: {e}");
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
        let delta = 0.005 * delta_time.as_millis() as f32;
        let mut movement = (0.0, 0.0, 0.0);
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
        if self.keyboard_state.is_pressed(KeyCode::Space) {
            movement.2 += delta;
        }
        if self.keyboard_state.is_pressed(KeyCode::ControlLeft) {
            movement.2 -= delta;
        }
        self.translate(movement.0, movement.1, movement.2);
        if self.are_angles_updated == true {
            self.api
                .as_ref()
                .unwrap()
                .update_camera_angles(self.pitch_rad, 0.0, self.yaw_rad);
            self.are_angles_updated = false;
        }

        let id = self.goal;
        let transformation =
            self.goal_pos * Matrix4::from_angle_y(cgmath::Rad(self.goal_yaw_rad)) * self.goal_scale;
        let _ = self.get_api().move_instance_to(id, transformation);
        self.goal_yaw_rad += 0.01;
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
    // pub fn get_pos_delta(&self) -> (f32, f32) {
    //     (
    //         (self.current_position.x - self.last_position.x) as f32,
    //         (self.current_position.y - self.last_position.y) as f32,
    //     )
    // }
    // pub fn get_click_delta(&self) -> f32 {
    //     // We just need an approximation to decide if the mouse has moved too much to ignore the click
    //     ((self.current_position.x - self.click_position.x).abs()
    //         + (self.current_position.y - self.click_position.y).abs()) as f32
    // }
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
