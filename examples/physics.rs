extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::entity;
use nmg::render;
use nmg::components;
use nmg::components::Component;

struct Demo {
    objects: Vec<entity::Handle>,
}

impl nmg::Game for Demo {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        transforms: &mut components::transform::Manager,
        draws:      &mut components::draw::Manager,
    ) {
        let object = entities.add();
        transforms.register(object);
        draws.register(object, 0);

        // Update demo state
        self.objects.push(object);
    }

    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        screen_height: u32,
        screen_width:  u32,
        entities:   &mut entity::Manager,
        transforms: &mut components::transform::Manager,
        draws:      &mut components::draw::Manager,
    ) -> render::SharedUBO {
        let shared_ubo = {
            let view = alg::Mat::look_at_view(
                alg::Vec3::new(-1.0, 0.5, -0.1), // Camera position
                alg::Vec3::new( 0.0, 0.0,  2.0), // Target position
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

        transforms.set(
            self.objects[0],
            alg::Vec3::new(0., 0., 2.),
            alg::Mat::identity(),
            alg::Vec3::one(),
        );

        shared_ubo
    }
}

fn main() {
    let demo = Demo { objects: Vec::new() };
    let model_data = get_models();
    nmg::go(model_data, demo)
}

fn get_models() -> Vec<render::ModelData> {
    let pyramid = render::ModelData::new(
        vec![
            render::Vertex::new( 0.0,  0.5,  0.0, 1., 1., 0.), // Peak
            render::Vertex::new( 0.5, -0.5, -0.5, 1., 0., 0.),
            render::Vertex::new(-0.5, -0.5, -0.5, 1., 0., 1.),
            render::Vertex::new( 0.5, -0.5,  0.5, 1., 1., 0.),
            render::Vertex::new(-0.5, -0.5,  0.5, 1., 1., 1.),
        ], vec![
            0u32, 1u32, 2u32,
            0u32, 3u32, 1u32,
            0u32, 4u32, 3u32,
            0u32, 2u32, 4u32,
            1u32, 2u32, 4u32,
            4u32, 3u32, 1u32,
        ],
    );

    vec![pyramid]
}
