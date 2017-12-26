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
        Ok(context) => update(events, &context),
        Err(e) => eprintln!("Could not create Vulkan context: {}", e)
    }
}

fn update(
    mut events: vdw::winit::EventsLoop,
    context: &render::Context,
) {
    let mut running = true;

    loop {
        // Handle window events
        events.poll_events(|event| {
            match event {
                vdw::winit::Event::WindowEvent {
                    event: vdw::winit::WindowEvent::Closed,
                    ..
                } => {
                    running = false;
                },
                _ => ()
            }
        });

        if !running {
            break;
        }

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
            panic!("{}", e);
        }
    }

    // Synchronize
    context.device.wait_idle();
}
