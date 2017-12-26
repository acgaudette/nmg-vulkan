extern crate voodoo as vd;
extern crate voodoo_winit as vdw;

const TITLE: &'static str = env!("CARGO_PKG_NAME");

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];
const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

const SHADER_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/");

struct VulkanContext {
    device:          vd::Device,
    swapchain:       vd::SwapchainKhr,
    command_buffers: Vec<vd::CommandBuffer>,
    graphics_family: u32,
    present_family:  u32,

    image_available: vd::Semaphore,
    render_complete: vd::Semaphore,

    _framebuffers: Vec<vd::Framebuffer>,
    _render_pass:  vd::RenderPass,
    _views:        Vec<vd::ImageView>,
    _pipeline:     vd::GraphicsPipeline
}

impl VulkanContext {
    pub fn new(window: &vdw::winit::Window) -> vd::Result<VulkanContext> {
        let (
            device,
            swapchain,
            command_buffers,
            graphics_family,
            present_family,
            _framebuffers,
            _render_pass,
            _views,
            _pipeline
        ) = init_vulkan(window)?;

        let (image_available, render_complete) = init_drawing(device.clone())?;

        Ok(
            VulkanContext {
                device,
                swapchain,
                command_buffers,
                graphics_family,
                present_family,
                image_available,
                render_complete,
                _framebuffers,
                _render_pass,
                _views,
                _pipeline
            }
        )
    }
}

fn init_vulkan(window: &vdw::winit::Window) -> vd::Result<(
    vd::Device,
    vd::SwapchainKhr,
    Vec<vd::CommandBuffer>,
    u32,
    u32,
    Vec<vd::Framebuffer>,
    vd::RenderPass,
    Vec<vd::ImageView>,
    vd::GraphicsPipeline
)> {
    /* Application */

    let app_name = std::ffi::CString::new(TITLE)?;
    let app_info = vd::ApplicationInfo::builder()
        .application_name(&app_name)
        .application_version((0, 1, 0))
        .api_version((1, 0, 0))
        .build();

    /* Extensions */

    let loader = vd::Loader::new()?;
    let extensions = loader.enumerate_instance_extension_properties()?;

    /* Validation layers */

    let mut layers: &[&str] = &[];

    if ENABLE_VALIDATION_LAYERS {
        if loader.verify_layer_support(VALIDATION_LAYERS)? {
            layers = VALIDATION_LAYERS;
            println!("Validation layers successfully loaded");
        } else {
            // Continue without validation layers; handle error here
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

    /* Physical device */

    let physical_devices = instance.physical_devices()?;

    if physical_devices.is_empty() {
        return Err("no GPUs with Vulkan support".into())
    }

    // Create surface from window
    let surface = vdw::create_surface(instance.clone(), window)?;

    let mut physical_device = None;
    let mut formats = None;
    let mut present_modes = None;
    let mut graphics_family = 0;
    let mut present_family = 0;

    // Find a Vulkan-ready GPU
    for device in physical_devices {
        // Check for swapchain support
        if let Ok((f, p)) = get_swapchain_details(&device, &surface) {
            formats = Some(f);
            present_modes = Some(p);
        }

        // Check for graphics and presentation queue support
        if let Ok((i, j)) = get_q_indices(&device, &surface) {
            physical_device = Some(device);
            graphics_family = i;
            present_family = j;
        }
    }

    if let None = physical_device {
        return Err("no suitable GPUs found".into())
    }

    let physical_device = physical_device.unwrap();
    let formats = formats.unwrap();
    let present_modes = present_modes.unwrap();

    /* Surface */

    let surface_format = {
        // Begin with the initial values
        let mut format = formats[0].format();
        let mut color_space = formats[0].color_space();

        // Ideal scenario (card doesn't care)
        if formats.len() == 1 && formats[0].format() == vd::Format::Undefined {
            format = vd::Format::B8G8R8A8Unorm;
            color_space = vd::ColorSpaceKhr::SrgbNonlinearKhr;
        }

        // Search for what we want directly
        for option in formats {
            if option.format() == vd::Format::B8G8R8A8Unorm
                && option.color_space() == vd::ColorSpaceKhr::SrgbNonlinearKhr
            {
                format = vd::Format::B8G8R8A8Unorm;
                color_space = vd::ColorSpaceKhr::SrgbNonlinearKhr;
            }
        }

        vd::SurfaceFormatKhr::builder()
            .format(format)
            .color_space(color_space)
            .build()
    };

    let present_mode = {
        // Fall back on FIFO (guaranteed to be supported)
        let mut mode = vd::PresentModeKhr::FifoKhr;

        for option in present_modes {
            // Prefer triple buffering
            if option == vd::PresentModeKhr::MailboxKhr {
                mode = vd::PresentModeKhr::MailboxKhr;
                break;
            // Otherwise, prefer immediate
            } else if option == vd::PresentModeKhr::ImmediateKhr {
                mode = vd::PresentModeKhr::ImmediateKhr;
            }
        }

        mode
    };

    let capabilities = physical_device.surface_capabilities_khr(&surface)?;

    let swap_extent = {
        let mut extent = vd::Extent2d::default();

        // Common case--use the resolution of the current window
        if capabilities.current_extent().width() != u32::max_value() {
            extent = capabilities.current_extent().clone();
        } else {
            // Handle special case window managers and clamp
            extent.set_width(
                std::cmp::max(
                    capabilities.min_image_extent().width(),
                    std::cmp::min(
                        capabilities.max_image_extent().width(),
                        1280 // Default
                    )
                )
            );
            extent.set_height(
                std::cmp::max(
                    capabilities.min_image_extent().height(),
                    std::cmp::min(
                        capabilities.max_image_extent().height(),
                        720 // Default
                    )
                )
            );
        }

        extent
    };

    /* Logical device */

    let graphics_q_create_info = vd::DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_family)
        .queue_priorities(&[1.0])
        .build();

    let present_q_create_info = vd::DeviceQueueCreateInfo::builder()
        .queue_family_index(present_family)
        .queue_priorities(&[1.0])
        .build();

    // Combine queues if they share the same index
    let mut infos = vec![graphics_q_create_info];
    let mut indices = vec![graphics_family];
    let mut sharing_mode = vd::SharingMode::Exclusive;

    if graphics_family != present_family {
        infos.push(present_q_create_info);
        indices.push(present_family);
        sharing_mode = vd::SharingMode::Concurrent;
    }

    let features = vd::PhysicalDeviceFeatures::builder()
        .build();

    let device = vd::Device::builder()
        .queue_create_infos(&infos)
        .enabled_features(&features)
        .enabled_extension_names(DEVICE_EXTENSIONS)
        .build(physical_device)?;

    /* Swapchain */

    // Frame queue size
    let image_count = {
        let mut count = capabilities.min_image_count() + 1;

        // Check for exceeding the limit
        if capabilities.max_image_count() > 0
            && count > capabilities.max_image_count()
        {
            count = capabilities.max_image_count();
        }

        count
    };

    let swapchain = vd::SwapchainKhr::builder()
        .surface(&surface)
        .min_image_count(image_count)
        .image_format(surface_format.format())
        .image_color_space(surface_format.color_space())
        .image_extent(swap_extent)
        .image_array_layers(1)
        .image_usage(vd::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(sharing_mode)
        .queue_family_indices(&indices)
        .pre_transform(capabilities.current_transform()) // No change
        .composite_alpha(vd::CompositeAlphaFlagsKhr::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .build(device.clone())?;

    /* Image views */

    let chain = swapchain.clone();

    if chain.images().is_empty() {
        return Err("empty swapchain".into());
    }

    let mut views = Vec::with_capacity(chain.images().len());

    for i in 0..chain.images().len() {
        let view = vd::ImageView::builder()
            .image(&chain.images()[i])
            .view_type(vd::ImageViewType::Type2d)
            .format(chain.image_format())
            .components(vd::ComponentMapping::default())
            .subresource_range(
                vd::ImageSubresourceRange::builder()
                    .aspect_mask(vd::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build()
            ).build(device.clone(), Some(chain.clone()))?;

        views.push(view);
    }

    if views.is_empty() {
        return Err("empty views".into());
    }

    /* Shaders */

    let vert_buffer = vd::util::read_spir_v_file(
        [SHADER_PATH, "vert.spv"].concat()
    )?;

    let frag_buffer = vd::util::read_spir_v_file(
        [SHADER_PATH, "frag.spv"].concat()
    )?;

    let vert_mod = vd::ShaderModule::new(device.clone(), &vert_buffer)?;
    let frag_mod = vd::ShaderModule::new(device.clone(), &frag_buffer)?;

    let main = std::ffi::CString::new("main")?;

    let vert_stage = vd::PipelineShaderStageCreateInfo::builder()
        .stage(vd::ShaderStageFlags::VERTEX)
        .module(&vert_mod)
        .name(&main)
        .build();

    let frag_stage = vd::PipelineShaderStageCreateInfo::builder()
        .stage(vd::ShaderStageFlags::FRAGMENT)
        .module(&frag_mod)
        .name(&main)
        .build();

    let stages = &[vert_stage, frag_stage];

    /* Fixed-functions */

    let vert_info = vd::PipelineVertexInputStateCreateInfo::builder()
        .build();

    let assembly = vd::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vd::PrimitiveTopology::TriangleList)
        .primitive_restart_enable(false)
        .build();

    let viewports = &[
        vd::Viewport::builder()
            .x(0f32)
            .y(0f32)
            .width(swapchain.extent().width() as f32)
            .height(swapchain.extent().height() as f32)
            .min_depth(0f32)
            .max_depth(1f32)
            .build()
    ];

    let scissors = &[
        vd::Rect2d::builder()
            .offset(
                vd::Offset2d::builder()
                    .x(0)
                    .y(0)
                    .build()
            ).extent(swapchain.extent().clone())
            .build()
    ];

    let viewport_state = vd::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors)
        .build();

    let rasterizer = vd::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vd::PolygonMode::Fill)
        .cull_mode(vd::CullModeFlags::NONE)
        .front_face(vd::FrontFace::CounterClockwise)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0f32)
        .depth_bias_clamp(0f32)
        .depth_bias_slope_factor(0f32)
        .line_width(1f32)
        .build();

    let multisampling = vd::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vd::SampleCountFlags::COUNT_1)
        .sample_shading_enable(false)
        .min_sample_shading(1f32)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .build();

    // Alpha blending
    let attachments = &[
        vd::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vd::BlendFactor::SrcAlpha)
            .dst_color_blend_factor(vd::BlendFactor::OneMinusSrcAlpha)
            .color_blend_op(vd::BlendOp::Add)
            .src_alpha_blend_factor(vd::BlendFactor::One)
            .src_alpha_blend_factor(vd::BlendFactor::Zero)
            .alpha_blend_op(vd::BlendOp::Add)
            .color_write_mask(
                vd::ColorComponentFlags::R
                | vd::ColorComponentFlags::G
                | vd::ColorComponentFlags::B
                | vd::ColorComponentFlags::A
            ).build()
    ];

    let blending = vd::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vd::LogicOp::Copy)
        .attachments(attachments)
        .blend_constants([0f32; 4])
        .build();

    let layout = vd::PipelineLayout::builder()
        .build(device.clone())?;

    /* Render passes */

    // Clear framebuffer
    let color_attachment = vd::AttachmentDescription::builder()
        .format(swapchain.image_format())
        .samples(vd::SampleCountFlags::COUNT_1)
        .load_op(vd::AttachmentLoadOp::Clear)
        .store_op(vd::AttachmentStoreOp::Store)
        .stencil_load_op(vd::AttachmentLoadOp::DontCare)
        .stencil_store_op(vd::AttachmentStoreOp::DontCare)
        .initial_layout(vd::ImageLayout::Undefined)
        .final_layout(vd::ImageLayout::PresentSrcKhr)
        .build();

    let color_attachment_ref = vd::AttachmentReference::builder()
        .attachment(0)
        .layout(vd::ImageLayout::ColorAttachmentOptimal)
        .build();

    let color_attachments = &[color_attachment_ref];

    let subpass = vd::SubpassDescription::builder()
        .pipeline_bind_point(vd::PipelineBindPoint::Graphics)
        .color_attachments(color_attachments)
        .build();

    let dependency = vd::SubpassDependency::builder()
        .src_subpass(vd::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vd::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vd::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vd::AccessFlags::COLOR_ATTACHMENT_READ
            | vd::AccessFlags::COLOR_ATTACHMENT_WRITE
        ).build();

    let pass = vd::RenderPass::builder()
        .attachments(&[color_attachment])
        .subpasses(&[subpass])
        .dependencies(&[dependency])
        .build(device.clone())?;

    /* Pipeline */

    let pipeline = vd::GraphicsPipeline::builder()
        .stages(stages)
        .vertex_input_state(&vert_info)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&blending)
        .layout(&layout)
        .render_pass(&pass)
        .subpass(0)
        .base_pipeline_index(-1)
        .build(device.clone())?;

    /* Framebuffers */

    let mut framebuffers = Vec::with_capacity(views.len());

    for i in 0..views.len() {
        let attachments = &[&views[i]];

        let framebuffer = vd::Framebuffer::builder()
            .render_pass(&pass)
            .attachments(attachments)
            .width(swapchain.extent().width())
            .height(swapchain.extent().height())
            .layers(1)
            .build(device.clone())?;

        framebuffers.push(framebuffer)
    }

    if framebuffers.is_empty() {
        return Err("empty framebuffers vector".into());
    }

    /* Command buffers */

    let pool = vd::CommandPool::builder()
        .queue_family_index(graphics_family)
        .build(device.clone())?;

    let command_buffers = pool.allocate_command_buffers(
        vd::CommandBufferLevel::Primary,
        framebuffers.len() as u32,
    )?;

    for i in 0..command_buffers.len() {
        command_buffers[i].begin(vd::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;

        // Clear color
        let clear = &[
            vd::ClearValue {
                color: vd::ClearColorValue {
                    float32: [0f32, 0f32, 0f32, 1f32]
                }
            }
        ];

        let pass_info = vd::RenderPassBeginInfo::builder()
            .render_pass(&pass)
            .framebuffer(&framebuffers[i])
            .render_area(
                vd::Rect2d::builder()
                    .offset(
                        vd::Offset2d::builder()
                            .x(0)
                            .y(0)
                            .build()
                    ).extent(swapchain.extent().clone())
                    .build()
            ).clear_values(clear)
            .build();

        /* Execute render pass */

        command_buffers[i].begin_render_pass(
            &pass_info,
            vd::SubpassContents::Inline
        );

        command_buffers[i].bind_pipeline(
            vd::PipelineBindPoint::Graphics,
            &pipeline.handle()
        );

        command_buffers[i].draw(
            3, 1, 0, 0
        );

        command_buffers[i].end_render_pass();
        command_buffers[i].end()?;
    }

    Ok((
        device,
        swapchain,
        command_buffers.into_vec(),
        graphics_family,
        present_family,
        framebuffers,
        pass,
        views,
        pipeline
    ))
}

fn get_q_indices(
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

fn get_swapchain_details(
    physical_device: &vd::PhysicalDevice, surface: &vd::SurfaceKhr
) -> vd::Result<(Vec<vd::SurfaceFormatKhr>, Vec<vd::PresentModeKhr>)> {
    if !physical_device.verify_extension_support(DEVICE_EXTENSIONS)? {
        return Err("required GPU extensions not supported".into())
    }

    let formats = physical_device.surface_formats_khr(surface)?;
    let present_modes = physical_device.surface_present_modes_khr(surface)?;

    if formats.is_empty() {
        return Err("no valid surface format".into())
    }

    if present_modes.is_empty() {
        return Err("no valid present mode".into())
    }

    Ok((formats.into_vec(), present_modes.into_vec()))
}

fn init_drawing(device: vd::Device) -> vd::Result<(vd::Semaphore, vd::Semaphore)> {
    let image_available = vd::Semaphore::new(
        device.clone(),
        vd::SemaphoreCreateFlags::empty()
    )?;

    let render_complete = vd::Semaphore::new(
        device,
        vd::SemaphoreCreateFlags::empty()
    )?;

    Ok((image_available, render_complete))
}

fn render(
    device: &vd::Device,
    swapchain: &vd::SwapchainKhr,
    image_available: &vd::Semaphore,
    render_complete: &vd::Semaphore,
    command_buffers: &Vec<vd::CommandBuffer>,
    graphics_family: u32,
    present_family: u32
) -> vd::Result<()> {
    let index = swapchain.acquire_next_image_khr(
        u64::max_value(),
        Some(image_available),
        None
    )?;

    let waits = &[image_available.handle()];
    let signals = &[render_complete.handle()];
    let command_buffers = &[command_buffers[index as usize].handle()];

    let info = vd::SubmitInfo::builder()
        .wait_semaphores(waits)
        .wait_dst_stage_mask(&vd::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .command_buffers(command_buffers)
        .signal_semaphores(signals)
        .build();

    let graphics_q = device.get_device_queue(graphics_family, 0);

    match graphics_q {
        Some(gq) => {
            unsafe {
                device.queue_submit(gq, &[info], None)?;
            }

            let swapchains = &[swapchain.handle()];
            let indices = &[index];

            let present_q = device.get_device_queue(present_family, 0);

            match present_q {
                Some(pq) => {
                    let info = vd::PresentInfoKhr::builder()
                        .wait_semaphores(waits)
                        .swapchains(swapchains)
                        .image_indices(indices)
                        .build();

                    unsafe {
                        device.queue_present_khr(pq, &info)?;
                    }

                    // Synchronize with GPU in debug mode
                    // (prevents memory leaks from the validation layers)
                    if ENABLE_VALIDATION_LAYERS {
                        device.wait_idle();
                    }
                },
                None => return Err("no present queue".into())
            }
        },
        None => return Err("no graphics queue".into())
    }

    Ok(())
}

fn init_window() -> (vdw::winit::EventsLoop, vdw::winit::Window) {
    let events = vdw::winit::EventsLoop::new();

    let window = vdw::winit::WindowBuilder::new()
        .with_title(TITLE)
        .build(&events);

    if let Err(e) = window {
        panic!("{}", e);
    }

    (events, window.unwrap())
}

fn main() {
    let (events, window) = init_window();

    match VulkanContext::new(&window) {
        Ok(context) => update(events, &context),
        Err(e) => eprintln!("Could not create Vulkan context: {}", e)
    }
}

fn update(
    mut events: vdw::winit::EventsLoop,
    context: &VulkanContext,
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
        if let Err(e) = render(
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
