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
    text_0: Option<entity::Handle>,
    text_1: Option<entity::Handle>,
    label_0: Option<entity::Handle>,
    label_1: Option<entity::Handle>,
    camera: Option<entity::Handle>,
}

default_traits!(Demo, [nmg::FixedUpdate, components::softbody::Iterate]);

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    ) {
        /* 3d texts */

        let text_0 = entities.add();
        components.transforms.register(text_0);
        self.text_0 = Some(text_0);

        components.texts.register(text_0);
        components.texts.build()
            .text("the medium is the\nmassage")
            .scale_factor(1f32)
            .for_entity(text_0);
        self.objects.push(text_0);

        components.transforms.set_position(
            text_0,
            alg::Vec3::new(-1., -1., 2.),
        );

        let text_1 = entities.add();
        components.transforms.register(text_1);
        self.text_1 = Some(text_1);

        components.texts.register(text_1);
        components.texts.build()
            .text("silence is the product")
            .scale_factor(1f32)
            .for_entity(text_1);
        self.objects.push(text_1);

        components.transforms.set_position(
            text_1,
            alg::Vec3::new(-1., 0., 10.),
        );
        components.transforms.parent(
            text_0,
            text_1,
        );

        /* Labels */

        let label_0 = entities.add();
        components.transforms.register(label_0);
        self.label_0 = Some(label_0);

        components.labels.register(label_0);
        components.labels.build()
            .text("nmg_vulkan")
            .pixel_scale_factor(4f32)
            .for_entity(label_0);
        self.objects.push(label_0);

        components.transforms.set_position(
            label_0,
            alg::Vec3::new(0., 0.7, 0.),
        );

        let label_1 = entities.add();
        components.transforms.register(label_1);
        self.label_1 = Some(label_1);

        components.labels.register(label_1);
        components.labels.build()
            .text("FOR YOUR CONSIDERATION")
            .aspect_scale_factor(1f32)
            .for_entity(label_1);
        self.objects.push(label_1);

        components.transforms.set_position(
            label_1,
            alg::Vec3::new(0., 0., 0.),
        );

        /* Set up camera */

        let camera = entities.add();
        components.transforms.register(camera);
        components.cameras.register(camera);

        let camera_position = alg::Vec3::new(-1.0, 0.5, -0.5);
        let target_position = alg::Vec3::new( 0.0, 0.0,  2.0);
        let camera_orientation = alg::Quat::look_at(
            camera_position,
            target_position,
            alg::Vec3::up(),
        );

        self.camera = Some(camera);

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
        let angle = 0.5 * time as f32;

        // Rotate text
        components.transforms.set_orientation(
            self.text_1.unwrap(),
            alg::Quat::axis_angle_raw(alg::Vec3::fwd(), angle),
        );

        // Rotate label
        components.transforms.set_orientation(
            self.label_0.unwrap(),
            alg::Quat::axis_angle_raw(alg::Vec3::fwd(), -angle),
        );

        // Rotate label
        components.transforms.set_orientation(
            self.label_1.unwrap(),
            alg::Quat::axis_angle_raw(alg::Vec3::fwd(), angle),
        );

        // Rotate camera
        components.transforms.set_orientation(
            self.camera.unwrap(),
            alg::Quat::axis_angle_raw(alg::Vec3::up(), angle),
        );
    }
}

fn main() {
    let demo = Demo {
        objects: Vec::new(),
        text_0: None,
        text_1: None,
        label_0: None,
        label_1: None,
        camera: None,
    };

    nmg::go(vec![], demo)
}
