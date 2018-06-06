extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::entity;
use nmg::graphics;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

/* In debug mode, this demo will render in wireframe, with physics markers.
 * In release mode, it will render nothing!
 */

struct Demo {
    first:  Option<entity::Handle>,
    second: Option<entity::Handle>,
    third:  Option<entity::Handle>,
    camera: Option<entity::Handle>,
    last_target: alg::Vec3,
}

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities: &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let first = entities.add();
        components.transforms.register(first);
        components.softbodies.register(first);
        components.softbodies.build_instance()
            .make_limb(alg::Vec3::one())
            .mass(10.0)
            .for_entity(first);

        let second = entities.add();
        components.transforms.register(second);
        components.softbodies.register(second);
        components.softbodies.build_instance()
            .make_limb(alg::Vec3::one())
            .mass(10.0)
            .for_entity(second);

        let third = entities.add();
        components.transforms.register(third);
        components.softbodies.register(third);
        components.softbodies.build_instance()
            .make_limb(alg::Vec3::one())
            .mass(10.0)
            .for_entity(third);

        /* Create joints */

        components.softbodies.build_joint()
            .with_parent(first)
            .offset(alg::Vec3::fwd() * 0.5)
            .xyz(-45.0, 45.0)
            .for_child(second);

        components.softbodies.build_joint()
            .with_parent(first)
            .fwd(-alg::Vec3::fwd())
            .offset(-alg::Vec3::fwd() * 0.5)
            .xyz(-45.0, 45.0)
            .for_child(third);

        /* Add planes */

        components.softbodies.add_planes(&[
            alg::Plane::new(alg::Vec3::up(), 0.0),
            alg::Plane::new(-alg::Vec3::one(), 2.0),
            alg::Plane::new(alg::Vec3::right(), 3.0),
            alg::Plane::new(alg::Vec3::fwd(), 3.0),
            alg::Plane::new(-alg::Vec3::right(), 3.0),
        ]);

        // Set plane bounciness
        components.softbodies.set_bounce(0.04);

        // Add camera
        let camera = entities.add();
        components.transforms.register(camera);
        components.cameras.register(camera);

        self.first = Some(first);
        self.second = Some(second);
        self.third = Some(third);
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

        // Update camera

        let camera_position =
              alg::Mat3::rotation(0.0, 90_f32.to_radians(), 0.0)
            * alg::Mat4::translation(-3.0, 3.0, -6.0)
            * alg::Vec3::one();

        let target = {
            let new_target = (
                  components.transforms.get_position(self.first.unwrap())
                + components.transforms.get_position(self.second.unwrap())
                + components.transforms.get_position(self.third.unwrap())
            ) * 0.33;

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
    let demo = Demo {
        first:  None,
        second: None,
        third:  None,
        camera: None,
        last_target: alg::Vec3::zero(),
    };

    nmg::go(vec![], demo)
}
