extern crate voodoo as vd;
extern crate voodoo_winit as vdw;

pub mod alg;
pub mod render;
mod ecs;
mod statics;
mod util;
mod components;

pub trait Game {
    fn start(
        &mut self,
        instances: &mut render::Instances,
    );

    fn update(
        &mut self,
        time: f64,
        delta: f64,
        screen_height: u32,
        screen_width: u32,
        instances: &mut render::Instances,
    ) -> render::SharedUBO;
}

pub fn go<T>(model_data: Vec<render::ModelData>, mut game: T)
where
    T: Game,
{
    let (events, window) = init_window();
    let context = render::Context::new(&window, model_data);

    match context {
        Ok(mut context) => {
            game.start(&mut context.instances);
            update(game, &window, events, &mut context);
        }

        Err(e) => eprintln!("Could not create Vulkan context: {}", e)
    }
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

fn update<T>(
    mut game: T,
    window: &vdw::winit::Window,
    mut events: vdw::winit::EventsLoop,
    context: &mut render::Context,
) where
    T: Game,
{
    let mut running = true;
    let start = std::time::Instant::now();
    let mut last_time = 0f64;

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
            context.swapchain.extent().height(),
            context.swapchain.extent().width(),
            &mut context.instances,
        );

        // Update renderer
        if let Err(e) = context.update(shared_ubo) {
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
    }

    // Synchronize before exit
    context.device.wait_idle();
}
