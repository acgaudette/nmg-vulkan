extern crate nmg_vulkan as nmg;

use nmg::alg;
use nmg::entity;
use nmg::render;
use nmg::graphics;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

struct Demo {
    shape: Option<entity::Handle>,
    last_target: alg::Vec3,
}

impl nmg::Start for Demo {
    fn start(
        &mut self,
        entities: &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let shape = entities.add();
        components.transforms.register(shape);
        components.softbodies.register(shape);

        components.softbodies.init_limb(
            shape,
            10.0, // Mass
            0.015, // Rigidity (jiggly)
            alg::Vec3::one(), // Scale
        );

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
        components.softbodies.set_friction(0.0);

        self.shape = Some(shape);
    }
}

impl nmg::Update for Demo {
    #[allow(unused_variables)]
    fn update(
        &mut self,
        time: f64,
        delta: f64,
        metadata: nmg::Metadata,
        screen_height: u32,
        screen_width: u32,
        entities: &mut entity::Manager,
        components: &mut components::Container,
        input: &input::Manager,
        debug: &mut debug::Handler,
    ) -> render::SharedUBO {
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

        let shared_ubo = {
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

            let view = alg::Mat4::look_at_view(
                camera_position,
                target,
                alg::Vec3::up(),
            );

            let projection = {
                alg::Mat4::perspective(
                    60.0,
                    screen_width as f32 / screen_height as f32,
                    0.01,
                    8.0,
                )
            };

            render::SharedUBO::new(view, projection)
        };

        shared_ubo
    }
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
    ) { }
}

fn main() {
    let demo = Demo {
        shape:  None,
        last_target: alg::Vec3::zero(),
    };

    // Demo only renders anything in debug mode
    nmg::go(vec![], demo)
}
