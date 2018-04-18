extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::graphics;
use nmg::entity;
use nmg::render;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

struct Demo {
    last_angle: (f64, f64),
    cube: Option<entity::Handle>,
}

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let cube = entities.add();
        components.transforms.register(cube);
        components.draws.register(cube, 0);
        self.cube = Some(cube);
    }
}

impl nmg::Update for Demo {
    #[allow(unused_variables)]
    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        metadata: nmg::Metadata,
        screen_height: u32,
        screen_width:  u32,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
        input: &input::Manager,
        debug: &mut debug::Handler,
    ) -> render::SharedUBO {
        let shared_ubo = {
            // Compute rotation angle using mouse
            let angle = (
                self.last_angle.0 + input.mouse_delta.0 * 0.005,
                self.last_angle.1 + input.mouse_delta.1 * 0.005,
            );

            self.last_angle = angle;

            // Orbit camera
            let camera_position = alg::Mat::id()
                * alg::Mat::rotation_y(angle.0 as f32)
                * alg::Mat::rotation_x(angle.1 as f32)
                * alg::Mat::translation(0.0, 0.0, -2.0)
                * alg::Vec3::zero();

            let view = alg::Mat::look_at_view(
                camera_position,
                alg::Vec3::zero(), // Target position
                alg::Vec3::up(),
            );

            let projection = {
                alg::Mat::perspective(
                    60.,
                    screen_width as f32 / screen_height as f32,
                    0.01,
                    4.
                )
            };

            render::SharedUBO::new(view, projection)
        };

        components.transforms.set(
            self.cube.unwrap(),
            alg::Vec3::zero(),
            alg::Quat::id(),
            alg::Vec3::one(),
        );

        shared_ubo
    }
}

impl nmg::FixedUpdate for Demo {
    #[allow(unused_variables)]
    fn fixed_update(
        &mut self,
        time: f64,
        fixed_delta: f32,
        metadata: nmg::Metadata,
        screen_height: u32,
        screen_width: u32,
        entities: &mut entity::Manager,
        components: &mut components::Container,
        input: &input::Manager,
        debug: &mut debug::Handler,
    ) { }
}

fn main() {
    let demo = Demo { last_angle: (0.0, 0.0), cube: None };
    let model_data = get_models();
    nmg::go(model_data, demo)
}

// "Load" model(s)
fn get_models() -> Vec<render::ModelData> {
    let cube = render::ModelData::new(
        vec![
            // Top face
            render::Vertex::new(alg::Vec3::new(-0.5, 0.5,  0.5), graphics::Color::white()),
            render::Vertex::new(alg::Vec3::new( 0.5, 0.5,  0.5), graphics::Color::white()),
            render::Vertex::new(alg::Vec3::new( 0.5, 0.5, -0.5), graphics::Color::white()),
            render::Vertex::new(alg::Vec3::new(-0.5, 0.5, -0.5), graphics::Color::white()),

            // Bottom face
            render::Vertex::new(alg::Vec3::new(-0.5, -0.5,  0.5), graphics::Color::red()),
            render::Vertex::new(alg::Vec3::new( 0.5, -0.5,  0.5), graphics::Color::red()),
            render::Vertex::new(alg::Vec3::new( 0.5, -0.5, -0.5), graphics::Color::red()),
            render::Vertex::new(alg::Vec3::new(-0.5, -0.5, -0.5), graphics::Color::red()),
        ], vec![
            // Top face
            0, 1, 2,
            2, 3, 0,

            // Back face
            0, 1, 5,
            5, 4, 0,

            // Front face
            2, 3, 7,
            7, 6, 2,

            // Right face
            1, 2, 6,
            6, 5, 1,

            // Left face
            0, 3, 7,
            7, 4, 0,

            // Bottom face
            4, 5, 6,
            6, 7, 4,
        ],
    );

    vec![cube]
}
