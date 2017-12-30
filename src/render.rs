extern crate voodoo as vd;
extern crate voodoo_winit as vdw;

use std;
use statics;
use ops;

macro_rules! offset_of {
    ($struct:ty, $field:tt) => (
        unsafe {
            let value: $struct = $crate::std::mem::uninitialized();
            let base = &value as *const _ as u32;
            let indent = &value.$field as *const _ as u32;
            $crate::std::mem::forget(value);

            indent - base
        }
    );
}

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];
const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

const SHADER_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/");

pub struct Context<'a> {
    /* Drawing fields */

    pub device:          vd::Device,
    pub swapchain:       vd::SwapchainKhr,
    pub command_buffers: Vec<vd::CommandBuffer>,
    pub graphics_family: u32,
    pub present_family:  u32,
    pub image_available: vd::Semaphore,
    pub render_complete: vd::Semaphore,

    /* Swapchain recreation data */

    physical_device: vd::PhysicalDevice,
    surface:         vd::SurfaceKhr,
    surface_format:  vd::SurfaceFormatKhr,
    sharing_mode:    vd::SharingMode,
    indices:         Vec<u32>,
    present_mode:    vd::PresentModeKhr,

    /* Fixed information */

    stages:         [vd::PipelineShaderStageCreateInfo<'a>; 2],
    assembly:       vd::PipelineInputAssemblyStateCreateInfo<'a>,
    rasterizer:     vd::PipelineRasterizationStateCreateInfo<'a>,
    multisampling:  vd::PipelineMultisampleStateCreateInfo<'a>,
    layout:         vd::PipelineLayout,
    drawing_pool:   vd::CommandPool,
    transient_pool: vd::CommandPool,

    /* Unsafe data */

    vertex_buffer: vd::BufferHandle,
    vertex_memory: vd::DeviceMemoryHandle,
    index_buffer : vd::BufferHandle,
    index_memory:  vd::DeviceMemoryHandle,

    /* Persistent data */

    _vert_mod:     vd::ShaderModule,
    _frag_mod:     vd::ShaderModule,
    _framebuffers: Vec<vd::Framebuffer>,
    _render_pass:  vd::RenderPass,
    _views:        Vec<vd::ImageView>,
    _pipeline:     vd::GraphicsPipeline,
}

impl<'a> Context<'a> {
    pub fn new(window: &vdw::winit::Window) -> vd::Result<Context> {
        let (
            surface,
            physical_device,
            graphics_family,
            present_family,
            surface_format,
            present_mode,
            indices,
            sharing_mode,
            device,
            drawing_pool,
            transient_pool,
            image_available,
            render_complete,
        ) = init_vulkan(window)?;

        let (
            _vert_mod,
            _frag_mod,
            stages,
            assembly,
            rasterizer,
            multisampling,
            layout,
        ) = init_fixed(device.clone())?;

        let (swapchain, _views) = init_swapchain(
            &physical_device,
            &surface,
            1280, 720, // Default
            &surface_format,
            sharing_mode,
            &indices,
            present_mode,
            None,
            &device,
        )?;

        let _render_pass = init_render_pass(&swapchain, &device)?;

        let _pipeline = init_pipeline(
            &swapchain,
            &stages,
            &assembly,
            &rasterizer,
            &multisampling,
            &layout,
            &_render_pass,
            &device,
        )?;

        let (
            _framebuffers,
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            command_buffers
        ) = init_drawing(
            &swapchain,
            &_views,
            &_render_pass,
            &device,
            &physical_device,
            &transient_pool,
            graphics_family,
            &drawing_pool,
            &_pipeline,
        )?;

        Ok(
            Context {
                device,
                swapchain,
                command_buffers,
                graphics_family,
                present_family,
                image_available,
                render_complete,
                physical_device,
                surface,
                surface_format,
                sharing_mode,
                indices,
                present_mode,
                stages,
                assembly,
                rasterizer,
                multisampling,
                layout,
                drawing_pool,
                transient_pool,
                vertex_buffer,
                vertex_memory,
                index_buffer,
                index_memory,
                _vert_mod,
                _frag_mod,
                _framebuffers,
                _render_pass,
                _views,
                _pipeline,
            }
        )
    }

    pub fn refresh_swapchain(
        &mut self, width: u32, height: u32
    ) -> vd::Result<()> {
        let (swapchain, _views) = init_swapchain(
            &self.physical_device,
            &self.surface,
            width, height,
            &self.surface_format,
            self.sharing_mode,
            &self.indices,
            self.present_mode,
            Some(&self.swapchain), // Pass in old swapchain
            &self.device,
        )?;

        let _render_pass = init_render_pass(&swapchain, &self.device)?;

        let _pipeline = init_pipeline(
            &swapchain,
            &self.stages,
            &self.assembly,
            &self.rasterizer,
            &self.multisampling,
            &self.layout,
            &_render_pass,
            &self.device,
        )?;

        let (
            _framebuffers,
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            command_buffers,
        ) = init_drawing(
            &swapchain,
            &_views,
            &_render_pass,
            &self.device,
            &self.physical_device,
            &self.transient_pool,
            self.graphics_family,
            &self.drawing_pool,
            &_pipeline,
        )?;

        // Synchronize
        self.device.wait_idle();

        /* Coup */

        self.swapchain = swapchain;
        self.command_buffers = command_buffers;

        unsafe {
            self.free_host();
        }

        self.vertex_buffer = vertex_buffer;
        self.vertex_memory = vertex_memory;

        self.index_buffer = index_buffer;
        self.index_memory = index_memory;

        self._framebuffers = _framebuffers;
        self._render_pass = _render_pass;
        self._views = _views;
        self._pipeline = _pipeline;

        Ok(())
    }

    unsafe fn free_host(&mut self) {
        // Free host vertex and index buffer allocations
        self.device.destroy_buffer(self.vertex_buffer, None);
        self.device.free_memory(self.vertex_memory, None);
        self.device.destroy_buffer(self.index_buffer, None);
        self.device.free_memory(self.index_memory, None);
    }
}

impl<'a> Drop for Context<'a> {
    fn drop(&mut self) {
        unsafe {
            self.free_host();
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Vertex {
    position: ops::Vec2,
    color:    ops::Vec3,
}

impl Vertex {
    fn new(x: f32, y: f32, r: f32, g: f32, b: f32) -> Vertex {
        Vertex {
            position: ops::Vec2::new(x, y),
            color:    ops::Vec3::new(r, g, b),
        }
    }

    fn binding_description() -> vd::VertexInputBindingDescription {
        vd::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vd::VertexInputRate::Vertex)
            .build()
    }

    fn attribute_descriptions() -> [vd::VertexInputAttributeDescription; 2] {
        [
            vd::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vd::Format::R32G32Sfloat)
                .offset(offset_of!(Vertex, position))
                .build(),
            vd::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vd::Format::R32G32B32Sfloat)
                .offset(offset_of!(Vertex, color))
                .build(),
        ]
    }
}

fn init_vulkan(window: &vdw::winit::Window) -> vd::Result<(
    vd::SurfaceKhr,
    vd::PhysicalDevice,
    u32,
    u32,
    vd::SurfaceFormatKhr,
    vd::PresentModeKhr,
    Vec<u32>,
    vd::SharingMode,
    vd::Device,
    vd::CommandPool,
    vd::CommandPool,
    vd::Semaphore,
    vd::Semaphore,
)> {
    /* Application */

    let app_name = std::ffi::CString::new(statics::TITLE)?;
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
        .build(physical_device.clone())?;

    /* Command buffers */

    let drawing_pool = vd::CommandPool::builder()
        .queue_family_index(graphics_family)
        .flags(vd::CommandPoolCreateFlags::empty())
        .build(device.clone())?;

    // For buffers with short lifetimes
    let transient_pool = vd::CommandPool::builder()
        .queue_family_index(graphics_family)
        .flags(vd::CommandPoolCreateFlags::TRANSIENT)
        .build(device.clone())?;

    /* Synchronization */

    let image_available = vd::Semaphore::new(
        device.clone(),
        vd::SemaphoreCreateFlags::empty()
    )?;

    let render_complete = vd::Semaphore::new(
        device.clone(),
        vd::SemaphoreCreateFlags::empty()
    )?;

    Ok((
        surface,
        physical_device,
        graphics_family,
        present_family,
        surface_format,
        present_mode,
        indices,
        sharing_mode,
        device,
        drawing_pool,
        transient_pool,
        image_available,
        render_complete,
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

fn init_fixed<'a>(
    device: vd::Device,
) -> vd::Result<(
    vd::ShaderModule,
    vd::ShaderModule,
    [vd::PipelineShaderStageCreateInfo<'a>; 2],
    vd::PipelineInputAssemblyStateCreateInfo<'a>,
    vd::PipelineRasterizationStateCreateInfo<'a>,
    vd::PipelineMultisampleStateCreateInfo<'a>,
    vd::PipelineLayout,
)> {
    /* Shaders */

    let vert_buffer = vd::util::read_spir_v_file(
        [SHADER_PATH, "vert.spv"].concat()
    )?;

    let frag_buffer = vd::util::read_spir_v_file(
        [SHADER_PATH, "frag.spv"].concat()
    )?;

    let vert_mod = vd::ShaderModule::new(device.clone(), &vert_buffer)?;
    let frag_mod = vd::ShaderModule::new(device.clone(), &frag_buffer)?;

    let main = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();

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

    let stages = [vert_stage, frag_stage];

    /* Fixed-functions */

    let assembly = vd::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vd::PrimitiveTopology::TriangleList)
        .primitive_restart_enable(false)
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

    let layout = vd::PipelineLayout::builder()
        .build(device)?;

    Ok((
        vert_mod,
        frag_mod,
        stages,
        assembly,
        rasterizer,
        multisampling,
        layout,
    ))
}

fn init_swapchain(
    physical_device: &vd::PhysicalDevice,
    surface:         &vd::SurfaceKhr,
    window_width:    u32,
    window_height:   u32,
    surface_format:  &vd::SurfaceFormatKhr,
    sharing_mode:    vd::SharingMode,
    indices:         &[u32],
    present_mode:    vd::PresentModeKhr,
    old_swapchain:   Option<&vd::SwapchainKhr>,
    device:          &vd::Device,
) -> vd::Result<(
    vd::SwapchainKhr,
    Vec<vd::ImageView>,
)> {
    /* Surface */

    let capabilities = physical_device.surface_capabilities_khr(&surface)?;

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
                        window_width,
                    )
                )
            );
            extent.set_height(
                std::cmp::max(
                    capabilities.min_image_extent().height(),
                    std::cmp::min(
                        capabilities.max_image_extent().height(),
                        window_height,
                    )
                )
            );
        }

        extent
    };

    /* Swapchain */

    let swapchain = {
        let mut builder = vd::SwapchainKhr::builder(); builder
            .surface(&surface)
            .min_image_count(image_count)
            .image_format(surface_format.format())
            .image_color_space(surface_format.color_space())
            .image_extent(swap_extent)
            .image_array_layers(1)
            .image_usage(vd::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(indices)
            .pre_transform(capabilities.current_transform()) // No change
            .composite_alpha(vd::CompositeAlphaFlagsKhr::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        if let Some(old) = old_swapchain {
            builder.old_swapchain(old.handle());
        }

        builder.build(device.clone())?
    };

    /* Image views */

    if swapchain.images().is_empty() {
        return Err("empty swapchain".into());
    }

    let views = {
        let mut views = Vec::with_capacity(swapchain.images().len());

        for i in 0..swapchain.images().len() {
            let chain = swapchain.clone();

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
                ).build(device.clone(), Some(chain))?;

            views.push(view);
        }

        views
    };

    if views.is_empty() {
        return Err("empty views".into());
    }

    Ok((swapchain, views))
}

fn init_render_pass(
    swapchain: &vd::SwapchainKhr,
    device:    &vd::Device
) -> vd::Result<(vd::RenderPass)> {
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

    let attachment_refs = [
        vd::AttachmentReference::builder()
            .attachment(0)
            .layout(vd::ImageLayout::ColorAttachmentOptimal)
            .build()
    ];

    let subpass = vd::SubpassDescription::builder()
        .pipeline_bind_point(vd::PipelineBindPoint::Graphics)
        .color_attachments(&attachment_refs)
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

    Ok(
        vd::RenderPass::builder()
            .attachments(&[color_attachment])
            .subpasses(&[subpass])
            .dependencies(&[dependency])
            .build(device.clone())?
    )
}

fn init_pipeline(
    swapchain:     &vd::SwapchainKhr,
    stages:        &[vd::PipelineShaderStageCreateInfo; 2],
    assembly:      &vd::PipelineInputAssemblyStateCreateInfo,
    rasterizer:    &vd::PipelineRasterizationStateCreateInfo,
    multisampling: &vd::PipelineMultisampleStateCreateInfo,
    layout:        &vd::PipelineLayout,
    render_pass:   &vd::RenderPass,
    device:        &vd::Device,
) -> vd::Result<(vd::GraphicsPipeline)> {
    /*
     * Fixed functions (these will be allocated on the heap later,
     * inside the graphics pipeline)
     */

    let binding_description = [Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_descriptions();

    let vert_info = vd::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_description)
        .vertex_attribute_descriptions(&attribute_descriptions)
        .build();

    let attachments = [
        // Alpha blending
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
        .attachments(&attachments)
        .blend_constants([0f32; 4])
        .build();

    /* Fixed functions (dependent on swapchain) */

    let viewports = [
        vd::Viewport::builder()
            .x(0f32)
            .y(0f32)
            .width(swapchain.extent().width() as f32)
            .height(swapchain.extent().height() as f32)
            .min_depth(0f32)
            .max_depth(1f32)
            .build()
    ];

    let scissors = [
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
        .viewports(&viewports)
        .scissors(&scissors)
        .build();

    /* Pipeline */

    Ok(
        vd::GraphicsPipeline::builder()
        .stages(stages)
        .vertex_input_state(&vert_info)
        .input_assembly_state(assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(rasterizer)
        .multisample_state(multisampling)
        .color_blend_state(&blending)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0)
        .base_pipeline_index(-1)
        .build(device.clone())?
    )
}

fn init_drawing(
    swapchain:       &vd::SwapchainKhr,
    views:           &[vd::ImageView],
    render_pass:     &vd::RenderPass,
    device:          &vd::Device,
    physical_device: &vd::PhysicalDevice,
    transient_pool:  &vd::CommandPool,
    graphics_family: u32,
    drawing_pool:    &vd::CommandPool,
    pipeline:        &vd::GraphicsPipeline,
) -> vd::Result<(
    Vec<vd::Framebuffer>,
    vd::BufferHandle,
    vd::DeviceMemoryHandle,
    vd::BufferHandle,
    vd::DeviceMemoryHandle,
    Vec<vd::CommandBuffer>,
)> {
    /* Framebuffers */

    let mut framebuffers = Vec::with_capacity(views.len());

    for i in 0..views.len() {
        let attachments = [&views[i]];

        let framebuffer = vd::Framebuffer::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(swapchain.extent().width())
            .height(swapchain.extent().height())
            .layers(1)
            .build(device.clone())?;

        framebuffers.push(framebuffer)
    }

    if framebuffers.is_empty() {
        return Err("empty framebuffers vector".into());
    }

    /* Vertex data */

    let vertices = [
        Vertex::new(-0.25f32, -0.5f32, 1f32, 0f32, 0f32),
        Vertex::new(-0.50f32,  0.5f32, 0f32, 1f32, 0f32),
        Vertex::new( 0.25f32,  0.5f32, 0f32, 0f32, 1f32),
        Vertex::new( 0.50f32, -0.5f32, 1f32, 1f32, 0f32),
    ];

    let indices = [
        0u16, 1u16, 2u16,
        2u16, 3u16, 0u16,
    ];

    /* Vertex buffer */

    let properties = physical_device.memory_properties();

    let (vertex_buffer, vertex_memory) = create_buffers(
        &vertices,
        &properties,
        device,
        vd::BufferUsageFlags::VERTEX_BUFFER,
        transient_pool,
        graphics_family,
    )?;

    /* Index buffer */

    let (index_buffer, index_memory) = create_buffers(
        &indices,
        &properties,
        device,
        vd::BufferUsageFlags::INDEX_BUFFER,
        transient_pool,
        graphics_family,
    )?;

    /* Command buffers */

    let command_buffers = drawing_pool.allocate_command_buffers(
        vd::CommandBufferLevel::Primary,
        framebuffers.len() as u32,
    )?;

    for i in 0..command_buffers.len() {
        command_buffers[i].begin(
            vd::CommandBufferUsageFlags::SIMULTANEOUS_USE,
        )?;

        // Clear color
        let clear = [
            vd::ClearValue {
                color: vd::ClearColorValue {
                    float32: [0f32, 0f32, 0f32, 1f32]
                }
            }
        ];

        let pass_info = vd::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
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
            ).clear_values(&clear)
            .build();

        /* Execute render pass */

        command_buffers[i].begin_render_pass(
            &pass_info,
            vd::SubpassContents::Inline,
        );

        command_buffers[i].bind_pipeline(
            vd::PipelineBindPoint::Graphics,
            &pipeline.handle(),
        );

        unsafe {
            let handle = command_buffers[i].handle();

            device.cmd_bind_vertex_buffers(
                handle,
                0,
                &[vertex_buffer],
                &[0],
            );

            device.cmd_bind_index_buffer(
                handle,
                index_buffer,
                0,
                vd::IndexType::Uint16,
            );
        }

        command_buffers[i].draw_indexed(
            indices.len() as u32,
            1,
            0,
            0,
            0,
        );

        command_buffers[i].end_render_pass();
        command_buffers[i].end()?;
    }

    Ok((
        framebuffers,
        vertex_buffer,
        vertex_memory,
        index_buffer,
        index_memory,
        command_buffers.into_vec(),
    ))
}

fn create_buffers<T: std::marker::Copy>(
    data:            &[T],
    properties:      &vd::PhysicalDeviceMemoryProperties,
    device:          &vd::Device,
    usage:           vd::BufferUsageFlags,
    transient_pool:  &vd::CommandPool,
    graphics_family: u32,
) -> vd::Result<(
    vd::BufferHandle,
    vd::DeviceMemoryHandle,
)> {
    let size = std::mem::size_of_val(data) as u64;

    // Local buffer (source)
    let (host_buffer, host_memory) = create_buffer(
        size,
        vd::BufferUsageFlags::TRANSFER_SRC,
        device,
        vd::MemoryPropertyFlags::HOST_VISIBLE
        | vd::MemoryPropertyFlags::HOST_COHERENT,
        properties,
    )?;

    // Memory-mapped IO
    unsafe {
        let ptr = device.map_memory(
            host_memory,
            0,
            size,
            vd::MemoryMapFlags::empty(),
        )?;

        let destination = std::slice::from_raw_parts_mut(
            ptr,
            data.len()
        );

        // Copy data
        destination.copy_from_slice(&data);

        device.unmap_memory(host_memory);
    }

    // GPU buffer (destination)
    let (device_buffer, device_memory) = create_buffer(
        size,
        usage | vd::BufferUsageFlags::TRANSFER_DST,
        device,
        vd::MemoryPropertyFlags::DEVICE_LOCAL,
        properties,
    )?;

    let transfer_buffer = transient_pool.allocate_command_buffer(
        vd::CommandBufferLevel::Primary,
    )?;

    transfer_buffer.begin(
        vd::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
    )?;

    // Copy buffer to GPU
    unsafe {
        let copy = vd::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(size)
            .build();

        device.cmd_copy_buffer(
            transfer_buffer.handle(),
            host_buffer,
            device_buffer,
            &[copy],
        );
    }

    transfer_buffer.end()?;

    let handles = [transfer_buffer.handle()];

    let submit_info = vd::SubmitInfo::builder()
        .command_buffers(&handles)
        .build();

    match device.get_device_queue(graphics_family, 0) {
        Some(gq) => unsafe {
            device.queue_submit(gq, &[submit_info], None)?;
        },

        None => return Err("no graphics queue".into())
    }

    // Block until transfer completion
    device.wait_idle();

    // Clean up device buffer
    unsafe {
        device.destroy_buffer(device_buffer, None);
        device.free_memory(device_memory, None);
    }

    Ok((host_buffer, host_memory))
}

fn create_buffer(
    size:       u64,
    usage:      vd::BufferUsageFlags,
    device:     &vd::Device,
    flags:      vd::MemoryPropertyFlags,
    properties: &vd::PhysicalDeviceMemoryProperties,
) -> vd::Result<(vd::BufferHandle, vd::DeviceMemoryHandle)> {
    let info = vd::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vd::SharingMode::Exclusive)
        .flags(vd::BufferCreateFlags::empty())
        .build();

    let buffer = unsafe {
        device.create_buffer(&info, None)?
    };

    let requirements = unsafe {
        device.get_buffer_memory_requirements(buffer)
    };

    let info = vd::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size())
        .memory_type_index(
            get_memory_type(
                requirements.memory_type_bits(),
                flags,
                properties.memory_types(),
            )?
        ).build();

    // Allocate GPU memory
    let handle = unsafe {
        device.allocate_memory(&info, None)?
    };

    unsafe {
        device.bind_buffer_memory(buffer, handle, 0)?;
    }

    Ok((buffer, handle))
}

fn get_memory_type(
    filter: u32,
    flags:  vd::MemoryPropertyFlags,
    types:  &[vd::MemoryType],
) -> vd::Result<u32> {
    for i in 0..types.len() {
        if filter & (1 << i) > 0
            && flags.intersects(types[i].property_flags())
        {
            return Ok(i as u32)
        }
    }

    Err("no valid memory type available on GPU".into())
}

pub fn draw(
    device:          &vd::Device,
    swapchain:       &vd::SwapchainKhr,
    image_available: &vd::Semaphore,
    render_complete: &vd::Semaphore,
    command_buffers: &Vec<vd::CommandBuffer>,
    graphics_family: u32,
    present_family:  u32,
) -> vd::Result<()> {
    let index = swapchain.acquire_next_image_khr(
        u64::max_value(), // Disable timeout
        Some(image_available),
        None,
    )?;

    let command_buffers = [command_buffers[index as usize].handle()];

    // Synchronization primitives
    let available_signals = [image_available.handle()];
    let complete_signals = [render_complete.handle()];

    // Wait for available images to render to
    let info = vd::SubmitInfo::builder()
        .wait_semaphores(&available_signals)
        .wait_dst_stage_mask(&vd::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .command_buffers(&command_buffers)
        .signal_semaphores(&complete_signals)
        .build();

    match device.get_device_queue(graphics_family, 0) {
        Some(gq) => {
            unsafe {
                // Render
                device.queue_submit(gq, &[info], None)?;
            }

            let swapchains = [swapchain.handle()];
            let indices = [index];

            let present_q = device.get_device_queue(present_family, 0);

            match present_q {
                Some(pq) => {
                    // Wait for complete frames to present
                    let info = vd::PresentInfoKhr::builder()
                        .wait_semaphores(&complete_signals)
                        .swapchains(&swapchains)
                        .image_indices(&indices)
                        .build();

                    unsafe {
                        // Present
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
