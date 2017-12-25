extern crate voodoo as vd;
extern crate voodoo_winit;

use std::ffi::CString;
use voodoo_winit::winit as vdw;

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

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
    let extensions = loader.enumerate_instance_extension_properties()?;

    /* Validation layers */

    let mut layers: &[&str] = &[];

    if ENABLE_VALIDATION_LAYERS {
        let layer_names: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];
        if loader.verify_layer_support(layer_names).unwrap() {
            layers = layer_names;
            println!("Validation layers successfully loaded");
        } else {
            eprintln!("Validation layers could not be loaded");
        }
    }

    let instance = vd::Instance::builder()
        .application_info(&app_info)
        .enabled_extensions(&extensions)
        .enabled_layer_names(layers)
        .print_debug_report(ENABLE_VALIDATION_LAYERS)
        .build(loader)?;

    let physical_devices = instance.loader().enumerate_physical_devices(&instance)?;

    if physical_devices.is_empty() {
        return Err("no GPUs with Vulkan support".into())
    }

    Ok(instance)
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
