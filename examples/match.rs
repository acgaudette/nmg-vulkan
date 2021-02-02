#[macro_use] extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::render;
use nmg::graphics;
use nmg::entity;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

struct Demo {
    last_angle: alg::Vec2,
    transform: alg::Mat3,
    scale: alg::Vec3,
    orient: alg::Quat,
    offset: f32,
    camera: Option<entity::Handle>,
}

default_traits!(Demo, [nmg::FixedUpdate, components::softbody::Iterate]);

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities: &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let camera = entities.add();
        components.transforms.register(camera);
        components.cameras.register(camera);
        self.camera = Some(camera);
    }
}

impl nmg::Update for Demo {
    #[allow(unused_variables)]
    fn update(
        &mut self,
        time: f64,
        delta: f64,
        metadata: nmg::Metadata,
        screen: nmg::ScreenData,
        parameters: &mut render::Parameters,
        entities: &mut entity::Manager,
        components: &mut components::Container,
        input: &mut input::Manager,
        debug: &mut debug::Handler,
    ) {
        let angle = self.last_angle + input.mouse_delta * 0.012;
        self.last_angle = angle;

        self.transform =
              alg::Mat3::rotation_y(angle.x as f32)
            * alg::Mat3::rotation_x(angle.y as f32);

        let sens = delta as f32;

        if input.key_held(input::Key::H) {
            if input.key_held(input::Key::LShift) { self.scale.x += sens; }
                                             else { self.scale.x -= sens; }
        }

        if input.key_held(input::Key::J) {
            if input.key_held(input::Key::LShift) { self.scale.y += sens; }
                                             else { self.scale.y -= sens; }
        }

        if input.key_held(input::Key::K) {
            if input.key_held(input::Key::LShift) { self.scale.z += sens; }
                                             else { self.scale.z -= sens; }
        }

        self.transform.set_col(0, self.transform.col(0) * self.scale.x);
        self.transform.set_col(1, self.transform.col(1) * self.scale.y);
        self.transform.set_col(2, self.transform.col(2) * self.scale.z);

        self.orient = components::softbody::extract_orient(
            self.orient,
            self.transform,
        );

        self.offset = alg::mix(
            self.offset,
            if input.key_held(input::Key::Space) { 1f32 } else { 0f32},
            delta as f32 * 8.0,
        );

        /* Debug */

        debug.clear_lines();

        debug.add_transform_axes(
                alg::Vec3::zero(),
                self.transform,
                0.5,
        );

        debug.add_local_axes(
                alg::Vec3::one() * self.offset * 0.08,
                self.orient * alg::Vec3::fwd(),
                self.orient * alg::Vec3::up(),
                1.0, // Half size
                1.0,
        );

        /* Update camera */

        components.transforms.set(
            self.camera.unwrap(),
            alg::Vec3::new(0.5, 1.0, -1.0),
            alg::Quat::axis_angle(alg::Vec3::up(), -0.5)
                * alg::Quat::axis_angle(alg::Vec3::right(), 0.5),
            alg::Vec3::one(),
        );
    }
}

fn main() {
    let demo = Demo {
        last_angle: alg::Vec2::zero(),
        transform: alg::Mat3::id(),
        scale: alg::Vec3::one(),
        orient: alg::Quat::id(),
        offset: 0.0,
        camera: None,
    };

    // Demo only renders anything in debug mode
    nmg::go(vec![], demo)
}
