extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::ecs;
use nmg::render;
use nmg::components;

struct Demo {
    objects: Vec<ecs::EntityHandle>,
}

impl nmg::Game for Demo {
    fn start(
        &mut self,
        entities:   &mut ecs::Entities,
        transforms: &mut components::transform::Transforms,
        draws:      &mut components::draw::Draws,
    ) {
        let object_0 = entities.add();
        let object_1 = entities.add();
        let object_2 = entities.add();

        transforms.register(object_0);
        transforms.register(object_1);
        transforms.register(object_2);

        draws.add(object_0, 0);
        draws.add(object_1, 0);
        draws.add(object_2, 0);

        self.objects.push(object_0);
        self.objects.push(object_1);
        self.objects.push(object_2);
    }

    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        screen_height: u32,
        screen_width:  u32,
        entities:   &mut ecs::Entities,
        transforms: &mut components::transform::Transforms,
        draws:      &mut components::draw::Draws,
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

        let angle = time as f32;

        transforms.set(
            self.objects[0],
            alg::Vec3::new(0., 0., 2.),
            alg::Mat::rotation(angle, angle, angle),
            alg::Vec3::one(),
        );

        transforms.set(
            self.objects[1],
            alg::Vec3::new(-0.8, -1.1, 3.),
            alg::Mat::rotation(0., angle, 0.),
            alg::Vec3::new(0.9, 0.9, 1.),
        );

        transforms.set(
            self.objects[2],
            alg::Vec3::new(1.6, 0.8, 4.),
            alg::Mat::rotation(0., 0., angle),
            alg::Vec3::new(0.8, 1.2, 1.),
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
