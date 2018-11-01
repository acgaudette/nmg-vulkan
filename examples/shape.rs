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
    shape: Option<entity::Handle>,
    last_target: alg::Vec3,
    camera: Option<entity::Handle>,
}

default_traits!(Demo, [nmg::FixedUpdate, components::softbody::Iterate]);

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities: &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let shape = entities.add();
        components.transforms.register(shape);
        components.softbodies.register(shape);

        components.softbodies.build_instance()
            .make_box_limb(alg::Vec3::one())
            .mass(10.0)
            .rigidity(0.015) // Jiggly
            .for_entity(shape);

        /* Add planes */

        components.softbodies.add_planes(&[
            alg::Plane::new( alg::Vec3::up(),    0.0),
            alg::Plane::new(-alg::Vec3::one(),   2.0),
            alg::Plane::new( alg::Vec3::right(), 3.0),
            alg::Plane::new( alg::Vec3::fwd(),   3.0),
            alg::Plane::new(-alg::Vec3::right(), 3.0),
        ]);

        // Set bounciness and friction
        components.softbodies.set_bounce(0.015);
        components.softbodies.set_friction(0.02);

        // Add camera
        let camera = entities.add();
        components.transforms.register(camera);
        components.cameras.register(camera);

        self.shape = Some(shape);
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
        input: &input::Manager,
        debug: &mut debug::Handler,
    ) {
        /* Debug */

        debug.clear_lines();

        // Ground plane
        debug.add_cross(
            alg::Vec3::zero(),
            4.0,
            graphics::Color::gray(),
        );

        // Draw softbodies
        components.softbodies.draw_all_debug(debug);

        /* Add force on key press */

        let add_force = input.key_pressed(input::Key::Space);

        components.softbodies.set_force(
            self.shape.unwrap(),
            if add_force { alg::Vec3::up() * 800.0 }
            else { alg::Vec3::zero() }
        );

        /* Update camera */

        let camera_position =
              alg::Mat3::rotation_y(90f32.to_radians())
            * alg::Mat4::translation(-3.0, 3.0, -6.0)
            * alg::Vec3::one();

        let target = {
            let new_target = components.transforms.get_position(
                self.shape.unwrap()
            );

            self.last_target.lerp(new_target, delta as f32)
        };

        self.last_target = target;

        components.transforms.set(
            self.camera.unwrap(),
            camera_position,
            alg::Quat::look_at(camera_position, target, alg::Vec3::up()),
            alg::Vec3::one(),
        );
    }
}

fn main() {
    let demo = Demo {
        shape: None,
        last_target: alg::Vec3::zero(),
        camera: None,
    };

    // Demo only renders anything in debug mode
    nmg::go(vec![], demo)
}
