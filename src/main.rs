extern crate voodoo as vd;
extern crate voodoo_winit;

use std::ffi::CString;
use voodoo_winit as vdw;

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

fn init_vulkan() -> vd::Result<vd::Instance> {
    /* Application */

    let app_name = CString::new("NMG")?;
    let app_info = vd::ApplicationInfo::builder()
        .application_name(&app_name)
        .application_version((0, 1, 0))
        .api_version((1, 0, 0))
        .build();

    let create_info = vd::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .build();

    /* Extensions */

    let loader = vd::Loader::new()?;
    let extensions = loader.enumerate_instance_extension_properties()?;

    /* Validation layers */

    const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];

    let mut layers: &[&str] = &[];

    if ENABLE_VALIDATION_LAYERS {
        if loader.verify_layer_support(VALIDATION_LAYERS).unwrap() {
            layers = VALIDATION_LAYERS;
            println!("Validation layers successfully loaded");
        } else {
            eprintln!("Validation layers could not be loaded");
        }
    }

    /* Instance */

    let instance = vd::Instance::builder()
        .application_info(&app_info)
        .enabled_extensions(&extensions)
        .enabled_layer_names(layers)
        .print_debug_report(ENABLE_VALIDATION_LAYERS)
        .build(loader)?;

    /* Window */

    let events = vdw::winit::EventsLoop::new();
    let window = vdw::winit::WindowBuilder::new()
        .with_title("NMG")
        .build(&events)
        .unwrap();

    /* Surface */

    let surface = vdw::create_surface(instance.clone(), &window)?;

    /* Physical device */

    const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

    let physical_devices = instance.physical_devices()?;

    if physical_devices.is_empty() {
        return Err("no GPUs with Vulkan support".into())
    }

    let mut physical_device = None;
    let mut graphics_family = 0;
    let mut present_family = 0;

    for device in physical_devices {
        if device.verify_extension_support(DEVICE_EXTENSIONS)? {
            let details = vd::SwapchainSupportDetails::new(&surface, &device)?;

            if details.formats.is_empty() || details.present_modes.is_empty() {
                continue;
            }
        } else {
            continue;
        }

        if let Ok((i, j)) = get_indices(&device, &surface) {
            physical_device = Some(device);
            graphics_family = i;
            present_family = j;
        }
    }

    if let None = physical_device {
        return Err("no suitable GPUs found".into())
    }

    let physical_device = physical_device.unwrap();

    /* Logical device */

    let graphics_q_create_info = vd::DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_family)
        .queue_priorities(&[1.0])
        .build();

    let present_q_create_info = vd::DeviceQueueCreateInfo::builder()
        .queue_family_index(present_family)
        .queue_priorities(&[1.0])
        .build();

    let mut infos = vec![graphics_q_create_info];

    if graphics_family != present_family {
        infos.push(present_q_create_info);
    }

    let features = vd::PhysicalDeviceFeatures::builder()
        .build();

    let device = vd::Device::builder()
        .queue_create_infos(&infos)
        .enabled_features(&features)
        .enabled_extension_names(DEVICE_EXTENSIONS)
        .build(physical_device)?;

    let graphics_q = device.get_device_queue(graphics_family, 0);
    let present_q = device.get_device_queue(present_family, 0);

    Ok(instance)
}

fn get_indices(
    physical_device: &vd::PhysicalDevice, surface: &vd::SurfaceKhr
) -> vd::Result<(u32, u32)> {
    let q_families = physical_device.queue_family_properties()?;

    let mut graphics_family = None;
    let mut present_family = None;

    let mut i = 0u32;

    for family in q_families {
        if family.queue_count() > 0 {
            if family.queue_flags().contains(vd::QueueFlags::GRAPHICS) {
                graphics_family = Some(i);
            }

            if physical_device.surface_support_khr(i, surface)? {
                present_family = Some(i);
            }
        }

        if let (Some(g), Some(p)) = (graphics_family, present_family) {
            return Ok((g, p))
        }

        i += 1;
    }

    Err("queue families for physical device not found".into())
}

fn update(instance: vd::Instance) {
    loop { }
}

fn main() {
    let instance = init_vulkan().unwrap();

    update(instance);
}
