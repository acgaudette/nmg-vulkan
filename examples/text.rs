#[macro_use] extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::entity;
use nmg::render;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

struct Demo {
    objects: Vec<entity::Handle>,
    light: Option<entity::Handle>,
}

default_traits!(Demo, [nmg::FixedUpdate, components::softbody::Iterate]);

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    ) {
        /* Add text 3d */
        let text_0 = entities.add();
        components.transforms.register(text_0);

        components.texts.register(text_0);
        components.texts.build()
            .text("QUICK BROWN FOX JUMPS OVER THE LAZY COW")
            .scale_factor(1f32)
            .for_entity(text_0);
        self.objects.push(text_0);

        components.transforms.set_position(
            text_0,
            alg::Vec3::new(-1., -1., 2.),
        );

        let text_1 = entities.add();
        components.transforms.register(text_1);

        components.texts.register(text_1);
        components.texts.build()
            .text("quick brown fox jumps over the lazy cow")
            .scale_factor(1f32)
            .for_entity(text_1);
        self.objects.push(text_1);

        components.transforms.set_position(
            text_1,
            alg::Vec3::new(-1., 0., 3.),
        );

        /* Add point light */

        let light = entities.add();
        components.transforms.register(light);

        components.lights.register(light);
        components.lights.build()
            .point_with_radius(8.0)
            .intensity(2.0)
            .for_entity(light);

        self.light = Some(light);

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
            alg::Vec3::one(),
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
        let angle = time as f32;

        // Animate light
        components.transforms.set_position(
            self.light.unwrap(),
            alg::Vec3::new(
                0.0,
                1.0 * angle.sin(),
                1.0 * angle.cos(),
            ) + alg::Vec3::fwd() * 1.0,
        );
    }
}

fn main() {
    let demo = Demo { objects: Vec::new(), light: None };
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
