use std::collections::HashSet;
use std::f32;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use anyhow::Result;
use bloom::api::{Child, Collider, Instance, Orientation};
use bloom::material::Material;
use bloom::quaternion::Quaternion;
use bloom::{
    self,
    api::{Bloomable, Camera},
    primitives::{Primitive, model::Model, sphere::Sphere},
    vec::Vec3,
};
use cgmath::Matrix4;
use hecs::Entity;
use maze_generator::prelude::Generator;
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
    player: Arc<RwLock<Option<Entity>>>,
    goal: Arc<RwLock<Option<Entity>>>,
    goal_yaw_rad: f32,
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

            player: Arc::new(RwLock::new(None)),
            goal: Arc::new(RwLock::new(None)),
            goal_yaw_rad: 0.0,
        }
    }
    pub fn translate(&mut self, dx: f32, dz: f32, dy: f32, world: &Arc<RwLock<hecs::World>>) {
        let delta: Vec3 = Vec3::new(dx, dy, dz);

        let mut w = world.write().unwrap();
        // Get player orientation
        let player_ori = w
            .query_one_mut::<&mut Orientation>(self.player.read().unwrap().unwrap())
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

        // let red = w.spawn((Material::new_basic(Vec3::new(1.0, 0.0, 0.0), 0.),));
        let grey = w.spawn((Material::new_basic(Vec3::new(0.7, 0.7, 0.7), 0.1),));
        // let yellow = w.spawn((Material::new_basic(Vec3::new(1.0, 0.8, 0.0), 0.),));
        // let glass = w.spawn((Material::new_clear(Vec3::new(0.9, 1.0, 1.0)),));
        let mirror_mat = w.spawn((Material::new_basic(Vec3::new(1.0, 1.0, 1.0), 0.992),));
        let light = w.spawn((Material::new_emissive(Vec3::new(0.9, 0.3, 0.1), 1.0),));
        let goal_colour = w.spawn((Material::new_emissive(Vec3::new(1.0, 1.0, 1.0), 1.0),));

        let character_material = w.spawn((Material::new(
            Vec3::new(1.0, 1.0, 1.0),
            0.0,
            0.0,
            1.4,
            1.0,
            0.1,
            Vec3::new(0.1, 1.0, 0.1),
        ),));

        const MAZE_SCALE: f32 = 5.0;
        const MAZE_WIDTH: i32 = 7;
        const MAZE_HEIGHT: i32 = 7;

        let (document, buffers, _) = gltf::import("models/Duck.glb").unwrap();
        // get the first primitive stored in the file
        let p = document
            .meshes()
            .next()
            .unwrap()
            .primitives()
            .next()
            .unwrap();
        let duck_model = Model::new_gltf_primitive(p.clone(), &buffers, character_material);
        let duck_collider = Collider::new(&duck_model, false);
        let d = w.spawn((Primitive::Model(duck_model),));

        let duck = w.spawn((
            Instance {
                primitive: d,
                base_transform: Matrix4::<f32>::from_translation(cgmath::Vector3::new(
                    0.0, -1.0, 0.0,
                )) * Matrix4::from_angle_y(cgmath::Rad(
                    -0.23 - std::f32::consts::FRAC_PI_2,
                )) * Matrix4::from_scale(1.0 / 100.0),
                initial_transform: Matrix4::<f32>::from_translation(cgmath::Vector3 {
                    x: -(MAZE_WIDTH as f32 / 2.0 - 0.5) * MAZE_SCALE,
                    y: 0.0,
                    z: -(MAZE_HEIGHT as f32 / 2.0 - 0.5) * MAZE_SCALE,
                }),
            },
            duck_collider,
            true,
        ));
        self.player = Arc::new(RwLock::new(Some(duck)));

        let g = w.spawn((Primitive::Model(Model::new_gltf_primitive(
            p.clone(),
            &buffers,
            goal_colour,
        )),));
        self.goal = Arc::new(RwLock::new(Some(w.spawn((Instance {
            primitive: g,
            base_transform: Matrix4::<f32>::from_scale(1.0 / 100.0),
            initial_transform: Matrix4::<f32>::from_translation(cgmath::Vector3 {
                x: (MAZE_WIDTH as f32 / 2.0 - 0.5) * MAZE_SCALE,
                y: 0.0,
                z: (MAZE_HEIGHT as f32 / 2.0 - 0.5) * MAZE_SCALE,
            }),
        },)))));

        // Create a camera
        let camera = w.spawn((
            Camera::default(),
            Child {
                parent: duck,
                offset_pos: Vec3::new(0.0, 0.5, 0.8),
                offset_quat: Quaternion::identity(),
            },
            Orientation::default(),
        ));

        self.camera = Arc::new(RwLock::new(Some(camera)));

        let mirror_model = Model::new_cube(mirror_mat)?;
        let mirror_collider = Collider::new(&mirror_model, true);

        let plane = w.spawn((Primitive::Model(Model::new_plane(grey)?),));
        let blank_mirror = w.spawn((Primitive::Model(Model::new_cube(grey)?),));

        let mirror = w.spawn((Primitive::Model(mirror_model),));

        // Create a maze
        // Each maze unit is 10 of our units
        let mut generator =
            maze_generator::recursive_backtracking::RbGenerator::new(Some(rand::random()));

        let maze = generator.generate(MAZE_WIDTH, MAZE_HEIGHT).unwrap();

        let north = maze_generator::prelude::Direction::North;
        let south = maze_generator::prelude::Direction::South;
        let east = maze_generator::prelude::Direction::East;
        let west = maze_generator::prelude::Direction::West;

        // Create the floor
        let _ = w.spawn((Instance {
            primitive: plane,
            base_transform: Matrix4::from_nonuniform_scale(
                MAZE_WIDTH as f32 * MAZE_SCALE / 2.0,
                1.0,
                MAZE_HEIGHT as f32 * MAZE_SCALE / 2.0,
            ),
            initial_transform: Matrix4::<f32>::from_translation(cgmath::Vector3 {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            }),
        },));

        log::debug!("Creating maze:\n{:?}", maze);

        // Loop through every tile of the maze
        for maze_y in 0..MAZE_HEIGHT {
            for maze_x in 0..MAZE_WIDTH {
                let coords = maze_generator::prelude::Coordinates::new(maze_x, maze_y);
                let tile = maze.get_field(&coords).unwrap();
                // For each cardinal direction, add a wall if necessary
                // 0, 0 in maze coordinates maps to -(width/2-0.5) * scale), -(height/2-0.5) * scale)
                let x = (maze_x as f32 - (MAZE_WIDTH as f32 / 2.0 - 0.5)) * MAZE_SCALE;
                let z = (maze_y as f32 - (MAZE_HEIGHT as f32 / 2.0 - 0.5)) * MAZE_SCALE;

                let id = if rand::random_bool(0.3) {
                    blank_mirror
                } else {
                    mirror
                };

                if !tile.has_passage(&north) {
                    let _ = w.spawn((
                        Instance {
                            primitive: id,
                            base_transform: Matrix4::from_nonuniform_scale(
                                MAZE_SCALE, MAZE_SCALE, 0.1,
                            ),
                            initial_transform: Matrix4::<f32>::from_translation(cgmath::Vector3 {
                                x,
                                y: 0.0,
                                z: z - MAZE_SCALE / 2.0,
                            }),
                        },
                        mirror_collider,
                    ));
                }
                if !tile.has_passage(&south) {
                    let _ = w.spawn((
                        Instance {
                            primitive: id,
                            base_transform: Matrix4::from_nonuniform_scale(
                                MAZE_SCALE, MAZE_SCALE, 0.1,
                            ),
                            initial_transform: Matrix4::<f32>::from_translation(cgmath::Vector3 {
                                x,
                                y: 0.0,
                                z: z + MAZE_SCALE / 2.0,
                            }),
                        },
                        mirror_collider,
                    ));
                }
                if !tile.has_passage(&east) {
                    let _ = w.spawn((
                        Instance {
                            primitive: id,
                            base_transform: Matrix4::from_nonuniform_scale(
                                0.1, MAZE_SCALE, MAZE_SCALE,
                            ),
                            initial_transform: Matrix4::<f32>::from_translation(cgmath::Vector3 {
                                x: x + MAZE_SCALE / 2.0,
                                y: 0.0,
                                z,
                            }),
                        },
                        Orientation::new(
                            Vec3::new(x + MAZE_SCALE / 2.0, 0.0, z),
                            Quaternion::identity(),
                        ),
                        mirror_collider,
                    ));
                }
                if !tile.has_passage(&west) {
                    let _ = w.spawn((
                        Instance {
                            primitive: id,
                            base_transform: Matrix4::from_nonuniform_scale(
                                0.1, MAZE_SCALE, MAZE_SCALE,
                            ),
                            initial_transform: Matrix4::<f32>::from_translation(cgmath::Vector3 {
                                x: x - MAZE_SCALE / 2.0,
                                y: 0.0,
                                z,
                            }),
                        },
                        mirror_collider,
                    ));
                }
            }
        }

        let sphere = w.spawn((Primitive::Sphere(Sphere::new(2.0, light).unwrap()),));
        let _ = w.spawn((
            Instance::new(sphere),
            Orientation::new(
                Vec3::new(
                    MAZE_WIDTH as f32 * MAZE_SCALE / 2.0,
                    MAZE_WIDTH as f32 * MAZE_SCALE,
                    -MAZE_HEIGHT as f32 * MAZE_SCALE / 2.0,
                ),
                Quaternion::identity(),
            ),
        ));

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
        // let api = self.get_api();
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_state.write().unwrap().update_position(position);
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
                self.mouse_state
                    .write()
                    .unwrap()
                    .update_button(button, state);

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
            // WindowEvent::KeyboardInput {
            //     event:
            //         KeyEvent {
            //             state: ElementState::Pressed,
            //             physical_key: PhysicalKey::Code(KeyCode::KeyQ),
            //             ..
            //         },
            //     ..
            // } => {
            //     let green = match self
            //         .get_api()
            //         .add_material(material::Material::new_basic(Vec3::new(0.0, 1.0, 0.0), 0.))
            //     {
            //         Ok(v) => v,
            //         Err(e) => {
            //             log::error!("Failed to add material: {e}");
            //             return;
            //         }
            //     };
            //     let cube = match Model::new_cube(green) {
            //         Ok(v) => v,
            //         Err(e) => {
            //             log::error!("Failed to create cube: {e}");
            //             return;
            //         }
            //     };
            //     let cube_id = match self.get_api().add_obj(Primitive::Model(cube)) {
            //         Ok(v) => v,
            //         Err(e) => {
            //             log::error!("Failed to add model: {e}");
            //             return;
            //         }
            //     };
            //     let mut rng = rand::rng();
            //     if let Err(e) = self.get_api().add_instance(
            //         cube_id,
            //         Matrix4::from_translation(cgmath::Vector3 {
            //             x: rng.random_range(-10.0..10.0),
            //             y: rng.random_range(-10.0..10.0),
            //             z: rng.random_range(-10.0..10.0),
            //         }),
            //     ) {
            //         log::error!("Failed to add cube: {e}");
            //     }
            // }
            // WindowEvent::KeyboardInput {
            //     event:
            //         KeyEvent {
            //             state: ElementState::Pressed,
            //             physical_key: PhysicalKey::Code(KeyCode::KeyZ),
            //             ..
            //         },
            //     ..
            // } => {
            //     let l = self.lentil;
            //     self.lentil_position.w.y += 0.05;
            //     let pos = self.lentil_position;
            //     if let Err(e) = self.get_api().move_instance_to(l, pos) {
            //         log::error!("Failed to move lentil up: {e}");
            //     }
            // }
            // WindowEvent::KeyboardInput {
            //     event:
            //         KeyEvent {
            //             state: ElementState::Pressed,
            //             physical_key: PhysicalKey::Code(KeyCode::KeyX),
            //             ..
            //         },
            //     ..
            // } => {
            //     let l = self.lentil;
            //     self.lentil_position.w.y -= 0.05;
            //     let pos = self.lentil_position;
            //     if let Err(e) = self.get_api().move_instance_to(l, pos) {
            //         log::error!("Failed to move lentil down: {e}");
            //     }
            // }
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
            // let player_ori =
            world
                .write()
                .unwrap()
                .query_one_mut::<&mut Orientation>(self.player.read().unwrap().unwrap())
                .unwrap()
                .quat = Quaternion::from_euler(0.0, 0.0, *self.yaw_rad.lock().unwrap());
            world
                .write()
                .unwrap()
                .query_one_mut::<&mut Orientation>(self.player.read().unwrap().unwrap())
                .unwrap()
                .update();
            world
                .write()
                .unwrap()
                .query_one_mut::<&mut Child>(self.camera.read().unwrap().unwrap())
                .unwrap()
                .offset_quat = Quaternion::from_euler(*self.pitch_rad.lock().unwrap(), 0.0, 0.0);
            *is_updated = false;
        }

        let mut w = world.write().unwrap();
        let ori = w
            .query_one_mut::<&mut Orientation>(self.goal.read().unwrap().unwrap())
            .unwrap();
        ori.quat = Quaternion::from_euler(0.0, 0.0, self.goal_yaw_rad);

        self.goal_yaw_rad += 0.01;
        if self.goal_yaw_rad > std::f32::consts::PI * 2.0 {
            self.goal_yaw_rad -= std::f32::consts::PI * 2.0;
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
