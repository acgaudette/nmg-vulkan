extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::entity;
use nmg::render;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

struct Demo {
    pyramid: Option<entity::Handle>,
}

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let pyramid = entities.add();
        components.transforms.register(pyramid);
        components.draws.register(pyramid);
        components.draws.bind_model(pyramid, 0);
        self.pyramid = Some(pyramid);

        let light = entities.add();
        components.lights.register(light);
        components.lights.build()
            .directional(alg::Vec3::fwd())
            .intensity(2.0)
            .for_entity(light);

        let camera = entities.add();
        components.transforms.register(camera);
        components.cameras.register(camera);
        components.transforms.set_position(
            camera,
            alg::Vec3::fwd() * -2.0,
        );
    }
}

impl nmg::Update for Demo {
    #[allow(unused_variables)]
    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        metadata: nmg::Metadata,
        screen: nmg::ScreenData,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
        input: &input::Manager,
        debug: &mut debug::Handler,
    ) {
        let pyramid = self.pyramid.unwrap();

        components.transforms.set(
            pyramid,
            alg::Vec3::zero(),
            alg::Quat::axis_angle(alg::Vec3::up(), (time as f32) * 2.),
            alg::Vec3::one(),
        );

        if input.key_held(input::Key::Space) {
            components.draws.unhide(pyramid);
        } else {
            components.draws.hide(pyramid);
        }
    }
}

impl nmg::FixedUpdate for Demo {
    #[allow(unused_variables)]
    fn fixed_update(
        &mut self,
        time: f64,
        fixed_delta: f32,
        metadata: nmg::Metadata,
        screen: nmg::ScreenData,
        entities: &mut entity::Manager,
        components: &mut components::Container,
        input: &input::Manager,
        debug: &mut debug::Handler,
    ) { }
}

fn main() {
    let demo = Demo { pyramid: None };
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
