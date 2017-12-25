extern crate voodoo as vd;
extern crate voodoo_winit;

use voodoo_winit::winit as vdw;
use std::ffi::CString;

fn init_vulkan() -> vd::Result<vd::Instance> {
    let app_name = CString::new("NMG")?;

    let app_info = vd::ApplicationInfo::builder()
        .application_name(&app_name)
        .application_version((0, 1, 0))
        .api_version((1, 0, 0))
        .build();

    let create_info = vd::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .build();

    let loader = vd::Loader::new()?;

    vd::Instance::builder()
        .application_info(&app_info)
        .enabled_extensions(&loader.enumerate_instance_extension_properties()?)
        .build(loader)
}

fn init_window() -> (vdw::Window, vdw::EventsLoop) {
    let events = vdw::EventsLoop::new();

    let window = vdw::WindowBuilder::new()
        .with_title("NMG")
        .build(&events)
        .unwrap();

    (window, events)
}

fn update(instance: vd::Instance) {
    loop { }
}

fn main() {
    let instance = init_vulkan().unwrap();
    let (window, events) = init_window();

    update(instance);
}
