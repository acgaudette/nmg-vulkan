extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::entity;
use nmg::render;
use nmg::components;
use nmg::components::Component;

struct Demo {
    objects: Vec<entity::Handle>,
    mass: f32,
    drag: f32,
}

impl nmg::Game for Demo {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let object = entities.add();
        components.transforms.register(object);
        components.draws.register(object, 0);
        components.rigidbodies.register(object);

        // Initial position
        components.transforms.set(
            object,
            alg::Vec3::new(0., 0., 2.),
            alg::Quat::identity(),
            alg::Vec3::one(),
        );

        // Initial mass, force, torque
        components.rigidbodies.set(
            object,
            self.mass,
            self.drag,
            alg::Vec3::up() * 1000.,
            alg::Vec3::right() * -5000.,
        );


        // Update demo state
        self.objects.push(object);
    }

    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        metadata: nmg::Metadata,
        screen_height: u32,
        screen_width:  u32,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
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

        if metadata.frame > 0 {
            // Reset forces
            components.rigidbodies.set(
                self.objects[0],
                self.mass,
                self.drag,
                alg::Vec3::zero(),
                alg::Vec3::zero(),
            );
        }

        shared_ubo
    }

    fn fixed_update(
        &mut self,
        time: f64,
        fixed_delta: f32,
        metadata: nmg::Metadata,
        screen_height: u32,
        screen_width: u32,
        entities: &mut entity::Manager,
        components: &mut components::Container,
    ) {}
}

fn main() {
    let demo = Demo {
        objects: Vec::new(),
        mass: 10.,
        drag: 3.
    };

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
