extern crate voodoo as vd;
extern crate voodoo_winit as vdw;

mod statics;
mod render;
mod ops;

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

fn main() {
    let (events, window) = init_window();
    let context = render::Context::new(&window);

    match context {
        Ok(mut context) => update(&window, events, &mut context),
        Err(e) => eprintln!("Could not create Vulkan context: {}", e)
    }
}

fn update(
    window: &vdw::winit::Window,
    mut events: vdw::winit::EventsLoop,
    context: &mut render::Context,
) {
    let mut running = true;

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

        // Render frame
        if let Err(e) = render::draw(
            &context.device,
            &context.swapchain,
            &context.image_available,
            &context.render_complete,
            &context.command_buffers,
            context.graphics_family,
            context.present_family
        ) {
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
