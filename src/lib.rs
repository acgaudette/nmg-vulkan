#![feature(core_intrinsics)]
#![feature(or_patterns)]

#[cfg(feature = "memory-test")] extern crate jemallocator;
#[cfg(feature = "memory-test")] #[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

extern crate voodoo as vd;
extern crate voodoo_winit as vdw;
extern crate gilrs;
extern crate fnv;
extern crate ini;

#[macro_use] extern crate lazy_static;

#[cfg(debug_assertions)]
#[macro_use]
macro_rules! fn_name {
    () => {{
        // Store the function name as a static string
        fn name_of<T>(_: T) -> &'static str {
            extern crate core;
            core::intrinsics::type_name::<T>()
        }

        fn f() {} // Declare bogus function to query function chain
        let name = name_of(f);
        &name[6..name.len() - 4] // Remove f() from the result
    }}
}

#[macro_export]
macro_rules! default_traits {
    ($t: ty, [$($trait: ty),* $(,)*]) => {
        $(impl $trait for $t { })*
    }
}

#[macro_use]
pub mod alg;
pub mod render;
pub mod graphics;
pub mod entity;
pub mod components;
pub mod config;
pub mod input;
pub mod obj_loader;
pub mod debug;
pub mod font;
mod statics;
mod util;

use std::thread;
use components::Component;

const FIXED_DT: f32 = 1. / 120.;
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
    pub width: u32,
    pub height: u32,
}

pub trait Start {
    #[allow(unused_variables)]
    fn start(
        &mut self,
        entities:   &mut entity::Manager,
        components: &mut components::Container,
    ) { }
}

pub trait Update {
    #[allow(unused_variables)]
    fn update(
        &mut self,
        time:  f64,
        delta: f64,
        metadata: Metadata,
        screen: ScreenData,
        parameters: &mut render::Parameters,
        entities: &mut entity::Manager,
        components: &mut components::Container,
        input: &mut input::Manager,
        debug: &mut debug::Handler,
    ) { }
}

pub trait FixedUpdate {
    #[allow(unused_variables)]
    fn fixed_update(
        &mut self,
        time: f64,
        fixed_delta: f32,
        metadata: Metadata,
        screen: ScreenData,
        parameters: &mut render::Parameters,
        entities: &mut entity::Manager,
        components: &mut components::Container,
        input: &mut input::Manager,
        debug: &mut debug::Handler,
    ) { }
}

pub fn go<T>(model_data: Vec<render::ModelData>, mut game: T)
where
    T: Start + Update + FixedUpdate
        + components::softbody::Iterate
{
    // Initialize window
    let (events, window) = init_window();

    // Initialize rendering engine
    let mut context = match render::Context::new(&window, model_data) {
        Ok(context) => context,
        Err(e) => panic!("Could not create Vulkan context: {}", e),
    };

    let mut parameters = render::Parameters::new();
    let instances = render::Instances::new(
        context.models.len(),
        &context.model_names,
        None,
    );

    /* Initialize input */

    let mut gamepads = match gilrs::GilrsBuilder::new().build() {
        Ok(pads) => pads,
        Err(e) => panic!("Could not create gamepad context: {}", e),
    };

    let mut input = input::Manager::new();

    // Create entities container
    let mut entities = entity::Manager::new(1);

    // Initialize core components
    let mut components = components::Container {
        transforms: components::transform::Manager::new(1),
        cameras:    components::camera::Manager::new(1),
        lights:     components::light::Manager::new(8),
        draws:      components::draw::Manager::new(1, instances),
        softbodies: components::softbody::Manager::new(1, 1, 1),
        texts:      components::text::Manager::new(8),
        labels:     components::label::Manager::new(8),
    };

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
        &mut parameters,
        &mut gamepads,
        &mut input,
        &mut entities,
        &mut components,
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
    parameters: &mut render::Parameters,
    mut gamepads: &mut gilrs::Gilrs,
    input:      &mut input::Manager,
    entities:   &mut entity::Manager,
    components: &mut components::Container,
    debug:      &mut debug::Handler,
) where
    T: Start + Update + FixedUpdate
        + components::softbody::Iterate
{
    let mut running = true;

    let start = std::time::Instant::now();
    let mut last_time = 0f64;
    let mut accumulator = 0f64; // Fixed-framerate accumulator
    let mut last_updated_counter = start;
    let mut last_updated_renderer = start;
    let mut last_frame = 0u32;

    let mut metadata = Metadata::new();

    let show_fps = config::load_section_setting::<bool>(
        &config::ENGINE_CONFIG,
        "settings",
        "show_fps",
    );

    let debug_fps = if show_fps {
        let handle = entities.add();
        components.transforms.register(handle);
        components.transforms.set_position(
            handle,
            alg::Vec3::new(-1.0, -1.0, 0.),
        );

        components.labels.register(handle);
        components.labels.build()
            .text("Frames per second: 0")
            .for_entity(handle);

        Some(handle)
    } else { None };

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

    /* Gamepad rumble */

    let mut rumble_gamepads = gamepads.gamepads()
        .filter_map(
            |(id, pad)|
                if pad.is_ff_supported() { Some(id) } else { None }
        ).collect::<Vec<_>>();

    let (rumble_lo, rumble_hi) = {
        let scheduling = gilrs::ff::Replay {
                 after: gilrs::ff::Ticks::from_ms(0),
              play_for: gilrs::ff::Ticks::from_ms(512),
            with_delay: gilrs::ff::Ticks::from_ms(0),
        };

        let mut builder_lo = gilrs::ff::EffectBuilder::new();

        builder_lo
            .gain(0.0)
            .repeat(gilrs::ff::Repeat::Infinitely)
            .distance_model(gilrs::ff::DistanceModel::None)
            .gamepads(&rumble_gamepads);

        let mut builder_hi = builder_lo.clone();

        (
            builder_lo
            .add_effect(
                gilrs::ff::BaseEffect {
                    kind: gilrs::ff::BaseEffectType::Weak { magnitude: u16::MAX },
                    scheduling, .. Default::default()
                }).finish(&mut gamepads)
                  .expect("Bad gamepad rumble array"),

            builder_hi
            .add_effect(
                gilrs::ff::BaseEffect {
                    kind: gilrs::ff::BaseEffectType::Strong { magnitude: u16::MAX },
                    scheduling, .. Default::default()
                }).finish(&mut gamepads)
                  .expect("Bad gamepad rumble array"),
        )
    };

    rumble_lo.play()
        .expect("Bad gamepad rumble array");
    rumble_hi.play()
        .expect("Bad gamepad rumble array");

    loop {
        // Update last frame of input
        input.increment_key_states();

        /* Gamepad input */

        while let Some(event) = gamepads.next_event() {
            match event {
                gilrs::Event {
                    id,
                    event: gilrs::EventType::Connected
                         | gilrs::EventType::Disconnected,
                    ..
                } => {
                    let gamepad = gamepads.gamepad(id);
                    let connected = if gamepad.is_connected() {
                        rumble_gamepads.push(id);
                        "connected"
                    } else {
                        let found = rumble_gamepads.iter().position(|&u| u == id)
                            .expect("Corrupted gamepad rumble array");
                        rumble_gamepads.remove(found);
                        "disconnected"
                    };

                    rumble_lo.set_gamepads(&rumble_gamepads, gamepads)
                        .expect("Bad gamepad rumble array");
                    rumble_hi.set_gamepads(&rumble_gamepads, gamepads)
                        .expect("Bad gamepad rumble array");

                    println!(
                        "Gamepad (id={}) {}: \"{}\" ({})",
                        id,
                        connected,
                        gamepad.name(),
                        match gamepad.power_info() {
                            gilrs::PowerInfo::Unknown => "unknown power source",
                            gilrs::PowerInfo::Wired => "wired",
                            gilrs::PowerInfo::Discharging(_) => "discharging",
                            gilrs::PowerInfo::Charging(_) => "charging",
                            gilrs::PowerInfo::Charged => "fully charged",
                        }
                    );
                },

                gilrs::Event {
                    id: _,
                    event: gilrs::EventType::AxisChanged(axis, s, _),
                    ..
                } => {
                    match axis {
                        gilrs::Axis::LeftStickX  => input.joy_l.x = s,
                        gilrs::Axis::LeftStickY  => input.joy_l.y = s,
                        gilrs::Axis::RightStickX => input.joy_r.x = s,
                        gilrs::Axis::RightStickY => input.joy_r.y = s,

                        _ => (),
                    };
                },

                gilrs::Event {
                    id: _,
                    event: gilrs::EventType::ButtonChanged(button, s, _),
                    ..
                } => {
                    match button {
                        gilrs::Button::LeftTrigger2  => input.trig_l = s,
                        gilrs::Button::RightTrigger2 => input.trig_r = s,

                        _ => (),
                    };
                },

                gilrs::Event {
                    id: _,
                    event: gilrs::EventType::ButtonPressed(button,  _)
                         | gilrs::EventType::ButtonReleased(button, _),
                    ..
                } => {
                    let key = match button {
                        gilrs::Button::North => Some(input::Key::North),
                        gilrs::Button::East => Some(input::Key::East),
                        gilrs::Button::South => Some(input::Key::South),
                        gilrs::Button::West => Some(input::Key::West),
                        _ => None,
                    };

                    if let Some(code) = key {
                        input.set_key_pressed(
                            code as usize,
                            match event.event {
                                gilrs::EventType::ButtonPressed(_, _) => true,
                                gilrs::EventType::ButtonReleased(_, _) => false,
                                _ => panic!(),
                            },
                        );
                    }
                },

                _ => (),
            };
        }

        let rumble_mix = input.mix_rumble();

        if rumble_mix.0 > 0.0 {
            match rumble_lo.set_gain(rumble_mix.0) {
                Ok(_) => (),
                Err(e) => panic!("Error setting rumble gain: {}", e)
            }
        }

        if rumble_mix.1 > 0.0 {
            match rumble_hi.set_gain(rumble_mix.1) {
                Ok(_) => (),
                Err(e) => panic!("Error setting rumble gain: {}", e)
            }
        }

        // Reset dirty input
        input.mouse_delta = alg::Vec2::zero();
        input.rumbles_lo.clear();
        input.rumbles_hi.clear();

        // Handle window events
        events.poll_events(|event| {
            match event {
                // Rebuild the swapchain if the window changes size
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::Resized(size),
                    ..
                } => {
                    let (width, height) = size.into();
                    if width == 0 || height == 0 { return; }

                    if let Err(e) = context.refresh_swapchain(width, height) {
                        panic!("{}", e);
                    }
                },

                // Stop the application if the window was closed
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::CloseRequested,
                    ..
                } => {
                    running = false;
                },

                // Grab mouse cursor if window is focused; release otherwise
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::Focused(focused),
                    ..
                } => {
                    if let Err(e) = window.grab_cursor(focused) {
                        eprintln!("{}", e);
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
                        position.x as f32,
                        position.y as f32,
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
            parameters,
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
                parameters,
                entities,
                components,
                input,
                debug,
            );

            // Update physics component
            components.softbodies.simulate(
                &mut game,
                &mut components.transforms,
            );

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

        components.texts.update(&components.transforms);
        components.labels.update(&components.transforms, screen);

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
        if let Err(e) = context.draw(
            &parameters,
            &components.draws.instances,
            &mut components.texts,
            &mut components.labels,
        ) {
            // Handle render errors
            if let vd::ErrorKind::ApiCall(result, _) = e.kind {
                // Rebuild the swapchain if it becomes out of date
                // or suboptimal
                if result == vd::CallResult::ErrorOutOfDateKhr
                    || result == vd::CallResult::SuboptimalKhr
                {
                    // Use existing window size
                    if let Some(size) = window.get_inner_size() {
                        match context.refresh_swapchain(
                            size.width as u32,
                            size.height as u32,
                        ) {
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

            if debug_fps.is_some() {
                components.labels.build()
                    .text(&format!("Frames per second: {}", metadata.fps))
                    .for_entity(debug_fps.unwrap());
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
        VirtualKeyCode::H =>        Some(Key::H),
        VirtualKeyCode::J =>        Some(Key::J),
        VirtualKeyCode::K =>        Some(Key::K),
        VirtualKeyCode::L =>        Some(Key::L),
        VirtualKeyCode::Z =>        Some(Key::Z),
        VirtualKeyCode::X =>        Some(Key::X),
        VirtualKeyCode::C =>        Some(Key::C),
        VirtualKeyCode::V =>        Some(Key::V),
        VirtualKeyCode::Q =>        Some(Key::Q),
        VirtualKeyCode::E =>        Some(Key::E),
        VirtualKeyCode::M =>        Some(Key::M),
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
