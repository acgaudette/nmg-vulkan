extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::render;

struct Demo {
    instances: Vec<render::InstanceHandle>,
}

impl nmg::Game for Demo {
    fn start(
        &mut self,
        instances: &mut render::Instances,
    ) {
        self.instances.push(
            instances.add(render::InstanceUBO::default(), 0)
        );

        self.instances.push(
            instances.add(render::InstanceUBO::default(), 0)
        );

        self.instances.push(
            instances.add(render::InstanceUBO::default(), 0)
        );
    }

    fn update(
        &mut self,
        time: f64,
        last_time: f64,
        screen_height: u32,
        screen_width: u32,
        instances: &mut render::Instances,
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

        let instance_ubo_0 = {
            let translation = alg::Mat::translation(0., 0., 2.);
            let rotation = alg::Mat::rotation(angle, angle, angle);
            let scale = alg::Mat::scale(0.8, 1.2, 1.);

            let model = translation * rotation * scale;
            render::InstanceUBO::new(model)
        };

        let instance_ubo_1 = {
            let translation = alg::Mat::translation(-0.5, -1.1, 3.);
            let rotation = alg::Mat::rotation(0., angle, 0.);
            let scale = alg::Mat::scale(0.8, 1.2, 1.);

            let model = translation * rotation * scale;
            render::InstanceUBO::new(model)
        };

        let instance_ubo_2 = {
            let translation = alg::Mat::translation(1.2, 0.8, 4.);
            let rotation = alg::Mat::rotation(angle, 0., 0.);
            let scale = alg::Mat::scale(0.8, 1.2, 1.);

            let model = translation * rotation * scale;
            render::InstanceUBO::new(model)
        };

        instances.update(
            self.instances[0],
            instance_ubo_0,
        );

        instances.update(
            self.instances[1],
            instance_ubo_1,
        );

        instances.update(
            self.instances[2],
            instance_ubo_2,
        );

        shared_ubo
    }
}

fn main() {
    let demo = Demo { instances: Vec::new() };
    let model_data = get_models();
    nmg::go(model_data, demo)
}

fn get_models() -> Vec<render::ModelData> {
    let pyramid = render::ModelData::new(
        vec![
            render::Vertex::new( 0.0,  0.5, 0.5, 1., 0., 0.),
            render::Vertex::new( 0.5, -0.5, 0.0, 1., 0., 0.),
            render::Vertex::new(-0.5, -0.5, 0.0, 1., 0., 0.),

            render::Vertex::new( 0.0,  0.5, 0.5, 0., 1., 0.),
            render::Vertex::new( 0.5, -0.5, 1.0, 0., 1., 0.),
            render::Vertex::new( 0.5, -0.5, 0.0, 0., 1., 0.),

            render::Vertex::new( 0.0,  0.5, 0.5, 0., 0., 1.),
            render::Vertex::new(-0.5, -0.5, 1.0, 0., 0., 1.),
            render::Vertex::new( 0.5, -0.5, 1.0, 0., 0., 1.),

            render::Vertex::new( 0.0,  0.5, 0.5, 1., 1., 0.),
            render::Vertex::new(-0.5, -0.5, 0.0, 1., 1., 0.),
            render::Vertex::new(-0.5, -0.5, 1.0, 1., 1., 0.),

            render::Vertex::new( 0.5, -0.5, 0.0, 1., 0., 0.),
            render::Vertex::new(-0.5, -0.5, 0.0, 1., 0., 0.),
            render::Vertex::new(-0.5, -0.5, 1.0, 0., 0., 1.),

            render::Vertex::new(-0.5, -0.5, 1.0, 0., 0., 1.),
            render::Vertex::new( 0.5, -0.5, 1.0, 0., 1., 0.),
            render::Vertex::new( 0.5, -0.5, 0.0, 1., 0., 0.),
        ], vec![
            0u32, 1u32, 2u32,
            0u32, 4u32, 1u32,
            0u32, 7u32, 4u32,
            0u32, 2u32, 7u32,
            1u32, 2u32, 7u32,
            7u32, 4u32, 1u32,
        ],
    );

    vec![pyramid]
}
