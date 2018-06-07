#![feature(core_intrinsics)]

extern crate voodoo as vd;
extern crate voodoo_winit as vdw;
extern crate ini;

#[macro_use] extern crate lazy_static;

#[cfg(debug_assertions)]
#[macro_use]
macro_rules! fn_name {
    () => {{
        // Store the function name as a static string
        fn name_of<T>(_: T) -> &'static str {
            extern crate core;
            unsafe { core::intrinsics::type_name::<T>() }
        }

        fn f() {} // Declare bogus function to query function chain
        let name = name_of(f);
        &name[6..name.len() - 4] // Remove f() from the result
    }}
}

pub mod alg;
pub mod render;
pub mod graphics;
pub mod entity;
pub mod components;
pub mod config;
pub mod input;
pub mod obj_loader;
pub mod debug;
mod statics;
mod util;

use std::thread;

const FIXED_DT: f32 = 1. / 100.;
const LIMIT_NS: u32 = 100_000;

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

#[derive(Clone, Copy)]
pub struct ScreenData {
    width: u32,
    height: u32,
}

pub trait Start {
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    );
}

pub trait Update {
    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        metadata: Metadata,
        screen: ScreenData,
        entities: &mut entity::Manager,
        components: &mut components::Container,
        input: &input::Manager,
        debug: &mut debug::Handler,
    );
}

pub trait FixedUpdate {
    fn fixed_update(
        &mut self,
        time: f64,
        fixed_delta: f32,
        metadata: Metadata,
        screen: ScreenData,
        entities: &mut entity::Manager,
        components: &mut components::Container,
        input: &input::Manager,
        debug: &mut debug::Handler,
    );
}

pub fn go<T>(model_data: Vec<render::ModelData>, mut game: T)
where
    T: Start + Update + FixedUpdate
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
        transforms: components::transform::Manager::new(1),
        cameras:    components::camera::Manager::new(1),
        lights:     components::light::Manager::new(8),
        draws:      components::draw::Manager::new(1, instances),
        softbodies: components::softbody::Manager::new(1, 1, 1),
    };

    // Create input manager
    let mut input = input::Manager::new();

    // Initialize debug struct
    let mut debug = debug::Handler::new();

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
        &mut input,
        &mut debug,
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
    input:      &mut input::Manager,
    debug:      &mut debug::Handler,
) where
    T: Start + Update + FixedUpdate
{
    let mut running = true;

    let start = std::time::Instant::now();
    let mut last_time = 0f64;
    let mut accumulator = 0f64; // Fixed-framerate accumulator
    let mut last_updated_counter = start;
    let mut last_updated_renderer = start;
    let mut last_frame = 0u32;

    let mut metadata = Metadata::new();

    /* Frame limiter */

    let target_fps = config::load_section_setting::<u32>(
        &config::ENGINE_CONFIG,
        "settings",
        "fps",
    );

    println!("Target frames per second: {}", target_fps);

    // Maximum time allowed to render a frame (ns)
    let frame_limit = if target_fps != 0 {
        1_000_000_000 / target_fps
    } else { 0 };

    /* Time scaling factor for the fixed timestep;
     * useful for debugging physics
     */

    let fixed_step_factor = config::load_section_setting::<f32>(
        &config::ENGINE_CONFIG,
        "settings",
        "fixed_step_factor"
    );

    let fixed_step = (
        FIXED_DT * fixed_step_factor
    ) as f64;

    loop {
        // Update last frame of input
        input.increment_key_states();

        // Reset dirty input
        input.mouse_delta = alg::Vec2::zero();

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

                // Grab mouse cursor if window is focused; otherwise release
                // TODO: potentially do something with set_cursor_state error
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::Focused(
                        focused
                    ),
                    ..
                } => {
                    if focused {
                        if let Err(e) = window.set_cursor_state(
                            vdw::winit::CursorState::Grab
                        ) { eprintln!("{}", e); }
                    } else {
                        if let Err(e) = window.set_cursor_state(
                            vdw::winit::CursorState::Normal
                        ) { eprintln!("{}", e); }
                    }
                },

                // Keyboard input
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::KeyboardInput {
                        input: vdw::winit::KeyboardInput {
                            state,
                            virtual_keycode: Some(virtual_keycode),
                            ..
                        },
                        ..
                    },
                    ..
                }=> {
                    if let Some(keycode) = vdw_key_to_key(virtual_keycode) {
                        input.set_key_pressed(
                            keycode as usize,
                            match state {
                                vdw::winit::ElementState::Pressed => true,
                                vdw::winit::ElementState::Released => false,
                            },
                        );
                    }
                },

                // Mouse input
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::CursorMoved {
                        position,
                        ..
                    },
                    ..
                } => {
                    input.cursor_coords = alg::Vec2::new(
                        position.0 as f32,
                        position.1 as f32,
                    );
                },

                vdw::winit::Event::DeviceEvent {
                    event: vdw::winit::DeviceEvent::MouseMotion {
                        delta
                    },
                    ..
                } => {
                    input.mouse_delta = alg::Vec2::new(
                        delta.0 as f32,
                        delta.1 as f32,
                    );
                },

                _ => ()
            }
        });

        if !running { break; }

        /* Time calculations */

        let now = std::time::Instant::now();
        let duration = now.duration_since(start);

        let time = duration.as_secs() as f64
            + (duration.subsec_nanos() as f64 / 1_000_000_000.);

        let delta = time - last_time;
        last_time = time;

        // Screen data
        let screen = {
            let extent = context.swapchain.extent();

            ScreenData {
                width: extent.width(),
                height: extent.height(),
            }
        };

        // Update game via callback
        game.update(
            time,
            delta,
            metadata,
            screen,
            entities,
            components,
            input,
            debug,
        );

        /* Fixed update loop */

        accumulator += delta;

        while accumulator >= fixed_step {
            game.fixed_update(
                time,
                FIXED_DT,
                metadata,
                screen,
                entities,
                components,
                input,
                debug,
            );

            // Update physics component
            components.softbodies.simulate(&mut components.transforms);

            accumulator -= fixed_step;
            metadata.fixed_frame += 1;
        }

        // Update render-related components
        components.lights.update(&components.transforms);
        components.draws.transfer(
            &components.transforms,
            &components.softbodies,
            &components.lights,
        );

        // Get shared UBO from camera component
        let shared_ubo = components.cameras.compute(
            &components.transforms,
            screen,
        );

        // Update renderer
        if let Err(e) = context.update(
            &components.draws.instances,
            shared_ubo,
        ) {
            // Irrecoverable error
            panic!("{}", e);
        }

        #[cfg(debug_assertions)] {
            if let Err(e) = context.update_debug(&debug.lines) {
                // Irrecoverable error
                panic!("{}", e);
            }
        }

        /* Limit frames per second */

        loop {
            let ns_since_update = std::time::Instant::now()
                .duration_since(last_updated_renderer)
                .subsec_nanos();

            if ns_since_update >= frame_limit { break; }

            let sleep_time = frame_limit - ns_since_update;
            if sleep_time > LIMIT_NS {
                thread::sleep(
                    std::time::Duration::new(0, sleep_time - LIMIT_NS)
                );
            }

            thread::sleep(std::time::Duration::new(0, 0));
        }

        // Reset now
        let now = std::time::Instant::now();
        last_updated_renderer = now;

        // Render frame
        if let Err(e) = context.draw(&components.draws.instances) {
            // Handle render errors
            if let vd::ErrorKind::ApiCall(result, _) = e.kind {
                // Rebuild the swapchain if it becomes out of date
                // or suboptimal
                if result == vd::CallResult::ErrorOutOfDateKhr
                    || result == vd::CallResult::SuboptimalKhr
                {
                    // Use existing window size
                    if let Some(size) = window.get_inner_size() {
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

        if now.duration_since(last_updated_counter).as_secs() > 0 {
            // Frames per second
            metadata.fps = metadata.frame - last_frame;
            last_frame = metadata.frame;

            #[cfg(debug_assertions)] {
                println!("Frames per second: {}", metadata.fps);
            }

            last_updated_counter = now;
        }
    }
}

fn vdw_key_to_key(keycode: vdw::winit::VirtualKeyCode) -> Option<input::Key> {
    use vdw::winit::VirtualKeyCode;
    use input::Key;

    match keycode {
        VirtualKeyCode::W =>        Some(Key::W),
        VirtualKeyCode::A =>        Some(Key::A),
        VirtualKeyCode::S =>        Some(Key::S),
        VirtualKeyCode::D =>        Some(Key::D),
        VirtualKeyCode::Up =>       Some(Key::Up),
        VirtualKeyCode::Down =>     Some(Key::Down),
        VirtualKeyCode::Left =>     Some(Key::Left),
        VirtualKeyCode::Right =>    Some(Key::Right),
        VirtualKeyCode::Space =>    Some(Key::Space),
        VirtualKeyCode::Return =>   Some(Key::Enter),
        VirtualKeyCode::LControl => Some(Key::LCtrl),
        VirtualKeyCode::LShift =>   Some(Key::LShift),
        _ => None,
    }
}
