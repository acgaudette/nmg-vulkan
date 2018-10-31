#[macro_use] extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::render;
use nmg::entity;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

struct Demo {
    pyramid: Option<entity::Handle>,
}

default_traits!(Demo, [nmg::FixedUpdate, components::softbody::Iterate]);

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let pyramid = entities.add();
        components.transforms.register(pyramid);
        components.draws.register(pyramid);
        components.draws.bind_model_index(pyramid, 0);
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
        parameters: &mut render::Parameters,
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

fn main() {
    let demo = Demo { pyramid: None };
    let model_data = get_models();
    nmg::go(model_data, demo)
}

fn get_models() -> Vec<render::ModelData> {
    let pyramid = render::ModelData::new_with_normals(
        "pyramid",
        vec![
            render::Vertex::new_position_color( 0.0,  0.5,  0.0, 1., 1., 0.),
            render::Vertex::new_position_color(-0.5, -0.5, -0.5, 1., 0., 1.),
            render::Vertex::new_position_color( 0.5, -0.5, -0.5, 1., 0., 0.),
            render::Vertex::new_position_color( 0.5, -0.5,  0.5, 1., 1., 0.),
            render::Vertex::new_position_color(-0.5, -0.5,  0.5, 1., 1., 1.),
        ], vec![
            0, 2, 1,
            0, 3, 2,
            0, 4, 3,
            0, 1, 4,
            2, 4, 1,
            3, 4, 2,
        ],
        render::NormalMode::Smooth,
    );

    vec![pyramid]
}
