extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::entity;
use nmg::render;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

struct Demo {
    objects: Vec<entity::Handle>,
    mass: f32,
    drag: f32,
}

impl nmg::Start for Demo {
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
            alg::Quat::id(),
            alg::Vec3::one(),
        );

        // Initial mass, force, torque
        components.rigidbodies.set(
            object,
            self.mass,
            self.drag,
            alg::Vec3::up() * 1000.,
            alg::Vec3::right() * -2000.,
        );

        /* Set up camera */

        let camera = entities.add();
        components.transforms.register(camera);
        components.cameras.register(camera);

        let camera_position = alg::Vec3::new(-1.0, 0.5, -0.1);
        let target_position = alg::Vec3::new( 0.0, 0.0,  2.0);
        let camera_orientation = alg::Quat::look_at(
            camera_position,
            target_position,
            alg::Vec3::up(),
        );

        components.transforms.set(
            camera,
            camera_position,
            camera_orientation,
            alg::Vec3::zero(),
        );

        // Update demo state
        self.objects.push(object);
    }
}

impl nmg::Update for Demo {
    #[allow(unused_variables)]
    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        metadata: nmg::Metadata,
        screen_width:  u32,
        screen_height: u32,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
        input: &input::Manager,
        debug: &mut debug::Handler,
    ) { }
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
    ) {
        if metadata.fixed_frame > 0 {
            // Reset forces
            components.rigidbodies.set(
                self.objects[0],
                self.mass,
                self.drag,
                alg::Vec3::zero(),
                alg::Vec3::zero(),
            );
        }
    }
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
    let pyramid = render::ModelData::new_with_normals(
        vec![
            render::Vertex::new_raw( 0.0,  0.5,  0.0, 0., 0., 0., 1., 1., 0.),
            render::Vertex::new_raw(-0.5, -0.5, -0.5, 0., 0., 0., 1., 0., 1.),
            render::Vertex::new_raw( 0.5, -0.5, -0.5, 0., 0., 0., 1., 0., 0.),
            render::Vertex::new_raw( 0.5, -0.5,  0.5, 0., 0., 0., 1., 1., 0.),
            render::Vertex::new_raw(-0.5, -0.5,  0.5, 0., 0., 0., 1., 1., 1.),
        ], vec![
            0u32, 2u32, 1u32,
            0u32, 3u32, 2u32,
            0u32, 4u32, 3u32,
            0u32, 1u32, 4u32,
            2u32, 4u32, 1u32,
            3u32, 4u32, 2u32,
        ],
        render::NormalMode::Smooth,
    );

    vec![pyramid]
}
