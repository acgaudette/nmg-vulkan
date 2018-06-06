extern crate nmg_vulkan as nmg;

use nmg::entity;
use nmg::components;
use nmg::components::Component;
use nmg::input;
use nmg::debug;

struct Demo { }

impl nmg::Start for Demo {
    #[allow(unused_variables)]
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    ) {
        let camera = entities.add();
        components.transforms.register(camera);
        components.cameras.register(camera);
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
    ) { }
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
    nmg::go(vec![], Demo { })
}
