extern crate ini;
#[macro_use]
extern crate lazy_static;
extern crate voodoo as vd;
extern crate voodoo_winit as vdw;

pub mod alg;
pub mod render;
pub mod entity;
pub mod components;
pub mod config;
mod statics;
mod util;

#[cfg(debug_assertions)]
const DEBUG_MODE: bool = true;
#[cfg(not(debug_assertions))]
const DEBUG_MODE: bool = false;

const FIXED_DT: f32 = 1. / 100.;

#[derive(Clone, Copy)]
pub struct Metadata {
    pub frame: u32,
    pub fixed_frame: u32,
    pub fps: u32,
}

impl Metadata {
    fn new() -> Metadata{
        Metadata {
            frame: 0,
            fixed_frame: 0,
            fps: 0,
        }
    }
}

pub trait Game {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    );

    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        metadata: Metadata,
        screen_height: u32,
        screen_width:  u32,
        entities: &mut entity::Manager,
        components: &mut components::Container,
    ) -> render::SharedUBO;

    fn fixed_update(
        &mut self,
        time: f64,
        fixed_delta: f32,
        metadata: Metadata,
        screen_heigh: u32,
        screen_width: u32,
        entities: &mut entity::Manager,
        components: &mut components::Container,
    );
}

pub fn go<T>(model_data: Vec<render::ModelData>, mut game: T)
where
    T: Game,
{
    // Initialize window
    let (events, window) = init_window();

    // Initialize rendering engine
    let mut context = match render::Context::new(&window, model_data) {
        Ok(mut context) => context,
        Err(e) => panic!("Could not create Vulkan context: {}", e)
    };

    let instances = render::Instances::new(context.models.len(), None);

    // Create entities container
    let mut entities = entity::Manager::new(1);

    // Initialize core components
    let mut components = components::Container {
        transforms:  components::transform::Manager::new(1),
        draws:       components::draw::Manager::new(1, instances),
        rigidbodies: components::rigidbody::Manager::new(1),
        softbodies:  components::softbody::Manager::new(1, 1),
    };

    // Start game
    game.start(&mut entities, &mut components);

    // Initiate update loop
    begin_update(
        game,
        &window,
        events,
        &mut context,
        &mut entities,
        &mut components,
    );

    // Synchronize before exit
    context.device.wait_idle();
}

fn init_window() -> (vdw::winit::EventsLoop, vdw::winit::Window) {
    let events = vdw::winit::EventsLoop::new();

    let window = vdw::winit::WindowBuilder::new()
        .with_title(statics::TITLE)
        .build(&events);

    if let Err(e) = window {
        panic!("{}", e);
    }

    (events, window.unwrap())
}

fn begin_update<T>(
    mut game:   T,
    window:     &vdw::winit::Window,
    mut events: vdw::winit::EventsLoop,
    context:    &mut render::Context,
    entities:   &mut entity::Manager,
    components: &mut components::Container,
) where
    T: Game,
{
    let mut running = true;

    let start = std::time::Instant::now();
    let mut last_time = 0f64;
    let mut accumulator = 0f32; // Fixed-framerate accumulator
    let mut last_updated = std::time::Instant::now();
    let mut last_frame = 0u32;

    let mut metadata = Metadata::new();

    let settings = &config::UNIVERSAL_CONFIG;
    let target_fps = settings.section(Some("settings"))
        .unwrap().get("fps").unwrap();

    println!("Target fps: {}", target_fps);

    loop {
        // Handle window events
        events.poll_events(|event| {
            match event {
                // Rebuild the swapchain if the window changes size
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::Resized(
                        width, height
                    ),
                    ..
                } => {
                    if width == 0 || height == 0 { return; }

                    if let Err(e) = context.refresh_swapchain(width, height) {
                        panic!("{}", e);
                    }
                },

                // Stop the application if the window was closed
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::Closed,
                    ..
                } => {
                    running = false;
                },

                _ => ()
            }
        });

        if !running { break; }

        /* Time calculations */

        let now = std::time::Instant::now();
        let duration = now.duration_since(start);

        let time = duration.as_secs() as f64
            + (duration.subsec_nanos() as f64 / 1000000000.);

        let delta = time - last_time;
        last_time = time;

        // Update scene data
        let shared_ubo = game.update(
            time,
            delta,
            metadata,
            context.swapchain.extent().height(),
            context.swapchain.extent().width(),
            entities,
            components,
        );

        /* Fixed update loop */

        accumulator += delta as f32;

        while accumulator >= FIXED_DT {
            game.fixed_update(
                time,
                FIXED_DT,
                metadata,
                context.swapchain.extent().height(),
                context.swapchain.extent().width(),
                entities,
                components,
            );

            // Update core components
            components.rigidbodies.simulate(&mut components.transforms);
            components.softbodies.simulate(&mut components.transforms);

            accumulator -= FIXED_DT;
            metadata.fixed_frame += 1;
        }

        // Update core component
        components.draws.transfer(
            &mut components.transforms,
            &mut components.softbodies,
        );

        // Update renderer
        if let Err(e) = context.update(
            &components.draws.instances,
            shared_ubo,
        ) {
            // Irrecoverable error
            panic!("{}", e);
        }

        // Render frame
        if let Err(e) = context.draw() {
            // Handle render errors
            if let vd::ErrorKind::ApiCall(result, _) = e.kind {
                // Rebuild the swapchain if it becomes out of date
                // or suboptimal
                if result == vd::CallResult::ErrorOutOfDateKhr
                    || result == vd::CallResult::SuboptimalKhr
                {
                    // Use existing window size
                    if let Some(size) = window.get_inner_size_pixels() {
                        match context.refresh_swapchain(size.0, size.1) {
                            Ok(()) => continue,
                            Err(e) => eprintln!("{}", e) // Fall through
                        }
                    } else {
                        // Fall through
                        eprintln!("Failed to acquire window size")
                    }
                }
            }

            // Irrecoverable error
            panic!("{}", e);
        }

        // Increment frame counter
        metadata.frame += 1;

        if now.duration_since(last_updated).as_secs() > 0 {
            // Frames per second
            metadata.fps = metadata.frame - last_frame;
            last_frame = metadata.frame;

            if DEBUG_MODE {
                println!("Frames per second: {}", metadata.fps);
            }

            last_updated = now;
        }
    }
}
