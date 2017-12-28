extern crate voodoo as vd;
extern crate voodoo_winit as vdw;

mod statics;
mod render;

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

    match render::Context::new(&window) {
        Ok(mut context) => update(events, &mut context),
        Err(e) => eprintln!("Could not create Vulkan context: {}", e)
    }
}

fn update(
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
            // Rebuild the swapchain if it becomes out of date
            if let vd::ErrorKind::ApiCall(result, _) = e.kind {
                if result == vd::CallResult::ErrorOutOfDateKhr {
                    match context.refresh_swapchain(1280, 720) {
                        Ok(()) => continue,
                        Err(e) => eprintln!("{}", e)
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
