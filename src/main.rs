use std::sync::{Arc, Mutex, Weak};

use bloom;
use bloom::api::BloomAPI;
use bloom::api::Bloomable;
use bloom::material;
use bloom::primitives::lentil::Lentil;
use bloom::primitives::model::Model;
use bloom::primitives::sphere::Sphere;
use bloom::primitives::Primitive;
use bloom::vec::Vec3;
use cgmath::Matrix4;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

fn main() {
    let demo = Demo::new();
    bloom::run(demo);
}

struct Demo {
    api: Weak<Mutex<BloomAPI>>,
    mouse_state: MouseState,
}

impl Demo {
    fn new() -> Self {
        Self {
            api: Weak::default(),
            mouse_state: MouseState::default(),
        }
    }
    fn get_api(&mut self) -> Arc<Mutex<BloomAPI>> {
        self.api.upgrade().unwrap()
    }
}

impl Bloomable for Demo {
    fn init(&mut self, api_in: Weak<Mutex<BloomAPI>>) {
        self.api = api_in;
        let binding = self.get_api();
        let mut api = binding.lock().unwrap();

        let scene = &mut api.scene;

        let red = scene.add_material(material::Material::new_basic(Vec3::new(1.0, 0.0, 0.0), 0.));
        let grey = scene.add_material(material::Material::new_basic(Vec3::new(0.7, 0.7, 0.7), 0.1));
        let yellow =
            scene.add_material(material::Material::new_basic(Vec3::new(1.0, 0.8, 0.0), 0.));
        let glass = scene.add_material(material::Material::new_clear(Vec3::new(0.9, 1.0, 1.0)));
        let mirror =
            scene.add_material(material::Material::new_basic(Vec3::new(1.0, 1.0, 0.9), 1.0));
        let light = scene.add_material(material::Material::new_emissive(
            Vec3::new(0.9, 0.3, 0.1),
            1.0,
        ));

        let (document, buffers, _) = gltf::import("models/Duck.glb").unwrap();
        for m in document.meshes() {
            for p in m.primitives() {
                let m = Model::new_gltf_primitive(p, &buffers, yellow);
                let m_id = scene.add_obj(Primitive::Model(m));
                let _ = scene.add_instance(
                    m_id,
                    Matrix4::<f32>::from_translation(cgmath::Vector3::new(-50.0, -1.0, 0.0))
                        * Matrix4::<f32>::from_scale(1.0 / 100.0),
                );
            }
        }

        let cube = Model::new_cube(red).unwrap();
        let cube_id = scene.add_obj(Primitive::Model(cube));

        let plane_id = scene.add_obj(Primitive::Model(Model::new_plane(grey).unwrap()));
        let mirror_id = scene.add_obj(Primitive::Model(Model::new_mirror(mirror).unwrap()));

        let _ = scene.add_instance(
            cube_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: 2.0,
                y: 3.0,
                z: -1.0,
            }),
        );
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
        let sphere_id = scene.add_obj(Primitive::Sphere(sphere));
        let _ = scene.add_instance(
            sphere_id,
            Matrix4::from_translation(cgmath::Vector3 {
                x: -1.0,
                y: 5.0,
                z: 3.0,
            }),
        );

        let lentil = Lentil::new(1.0, 1.0, glass).unwrap();
        let lentil_id = scene.add_obj(Primitive::Lentil(lentil));
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
    }
    fn resize(&mut self, _width: u32, _height: u32) {}
    fn input(&mut self, event: winit::event::WindowEvent) {
        let binding = self.get_api();
        let mut api = binding.lock().unwrap();
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_state.update_position(position);
                let (mut dx, mut dy) = self.mouse_state.get_pos_delta();
                dx *= -0.01;
                dy *= 0.01;

                if self.mouse_state.left_pressed {
                    api.camera.orbit(dx, dy);
                    api.uniform.reset_samples();
                } else if self.mouse_state.middle_pressed {
                    api.camera.pan(dx, dy);
                    api.uniform.reset_samples();
                } else if self.mouse_state.right_pressed {
                    api.camera.zoom(-dy);
                    api.uniform.reset_samples();
                }
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
                let delta = match delta {
                    MouseScrollDelta::PixelDelta(delta) => 0.001 * delta.y as f32,
                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                };
                api.camera.zoom(delta);
                api.uniform.reset_samples();
            }
            _ => (),
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
