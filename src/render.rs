extern crate voodoo as vd;
extern crate voodoo_winit as vdw;

use std;
use statics;
use alg;
use util;

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

const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];
const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];
const SHADER_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/");

const MAX_INSTANCES: u64 = 1024;
#[cfg(debug_assertions)]
const MAX_DEBUG_LINES: u64 = 512;

// Good GPUs have a minimum alignment of 256, which gives us about 12
// vertices to work with (adjusting for matrix size and padding).
const DYNAMIC_UBO_WIDTH: usize = 256;

pub const MAX_SOFTBODY_VERT: usize = (
    DYNAMIC_UBO_WIDTH - std::mem::size_of::<alg::Mat>()
) / std::mem::size_of::<PaddedVec3>();

#[allow(dead_code)]
pub struct Context<'a> {
    pub device:    vd::Device,
    pub swapchain: vd::SwapchainKhr,
    pub models:    Vec<Model>, // Lookup table

    /* Swapchain recreation data */

    surface:        vd::SurfaceKhr,
    surface_format: vd::SurfaceFormatKhr,
    sharing_mode:   vd::SharingMode,
    q_indices:      Vec<u32>,
    present_mode:   vd::PresentModeKhr,

    /* Fixed information */

    graphics_family: u32,
    present_family:  u32,
    drawing_pool:    vd::CommandPool,
    transient_pool:  vd::CommandPool,
    image_available: vd::Semaphore,
    render_complete: vd::Semaphore,
    command_fences:  Vec<vd::Fence>,
    shader_stages:   [vd::PipelineShaderStageCreateInfo<'a>; 2],
    depth_format:    vd::Format,
    assembly:        vd::PipelineInputAssemblyStateCreateInfo<'a>,
    rasterizer:      vd::PipelineRasterizationStateCreateInfo<'a>,
    multisampling:   vd::PipelineMultisampleStateCreateInfo<'a>,
    ubo_layout:      vd::DescriptorSetLayout,
    pipeline_layout: vd::PipelineLayout,
    render_pass:     vd::RenderPass,
    pipeline:        vd::GraphicsPipeline,
    framebuffers:    Vec<vd::Framebuffer>,
    ubo_alignment:   u64,
    descriptor_sets: Vec<vd::DescriptorSet>,
    command_buffers: Vec<vd::CommandBuffer>,

    /* Unsafe data */

    vertex_buffer:  vd::BufferHandle,
    vertex_memory:  vd::DeviceMemoryHandle,
    index_buffer:   vd::BufferHandle,
    index_memory:   vd::DeviceMemoryHandle,
    depth_memory:   vd::DeviceMemoryHandle,
    ubo_buffer:     vd::BufferHandle,
    ubo_memory:     vd::DeviceMemoryHandle,
    dyn_ubo_buffer: vd::BufferHandle,
    dyn_ubo_memory: vd::DeviceMemoryHandle,

    /* Debug data */

    debug_data: Option<DebugData>,
    debug_line_count: u32,

    /* Persistent data */

    _vert_mod:        vd::ShaderModule,
    _frag_mod:        vd::ShaderModule,
    _depth_image:     vd::Image,
    _views:           Vec<vd::ImageView>,
    _descriptor_pool: vd::DescriptorPool,
}

impl<'a> Context<'a> {
    pub fn new(
        window:     &vdw::winit::Window,
        model_data: Vec<ModelData>,
    ) -> vd::Result<Context> {
        let (
            surface,
            graphics_family,
            present_family,
            surface_format,
            present_mode,
            q_indices,
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
            shader_stages,
        ) = load_shaders(device.clone())?;

        let (
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            models,
        ) = load_models(
            model_data,
            &device,
            &transient_pool,
            graphics_family,
        )?;

        let (
            depth_format,
            assembly,
            rasterizer,
            multisampling,
            ubo_layout,
            pipeline_layout,
        ) = init_fixed(device.clone())?;

        let (swapchain, command_fences, _views) = init_swapchain(
            &device,
            &surface,
            1280, 720, // Default
            &surface_format,
            sharing_mode,
            &q_indices,
            present_mode,
            None,
        )?;

        let render_pass = init_render_pass(
            &swapchain,
            depth_format,
            &device,
        )?;

        let pipeline = init_pipeline(
            &swapchain,
            &shader_stages,
            &assembly,
            &rasterizer,
            &multisampling,
            &pipeline_layout,
            &render_pass,
            &device,
        )?;

        /* Optional debug data */

        let debug_data = init_debug(
            &swapchain,
            &render_pass,
            &pipeline_layout,
            &device,
        )?;

        let debug_line_count = 0;

        let (
            _depth_image,
            depth_memory,
            framebuffers,
            ubo_buffer,
            ubo_memory,
            dyn_ubo_buffer,
            dyn_ubo_memory,
            ubo_alignment,
            descriptor_sets,
            _descriptor_pool,
        ) = init_drawing(
            &swapchain,
            depth_format,
            &_views,
            &render_pass,
            &device,
            &transient_pool,
            graphics_family,
            ubo_layout.handle(),
        )?;

        let command_buffers = init_commands(&drawing_pool, &framebuffers)?;

        // Return newly-built context structure
        Ok(
            Context {
                device,
                swapchain,
                models,
                surface,
                surface_format,
                sharing_mode,
                q_indices,
                present_mode,
                graphics_family,
                present_family,
                drawing_pool,
                transient_pool,
                image_available,
                render_complete,
                command_fences,
                shader_stages,
                depth_format,
                assembly,
                rasterizer,
                multisampling,
                ubo_layout,
                pipeline_layout,
                render_pass,
                pipeline,
                framebuffers,
                ubo_alignment,
                descriptor_sets,
                command_buffers,
                vertex_buffer,
                vertex_memory,
                index_buffer,
                index_memory,
                depth_memory,
                ubo_buffer,
                ubo_memory,
                dyn_ubo_buffer,
                dyn_ubo_memory,
                debug_data,
                debug_line_count,
                _vert_mod,
                _frag_mod,
                _depth_image,
                _views,
                _descriptor_pool,
            }
        )
    }

    pub fn refresh_swapchain(
        &mut self, width: u32, height: u32
    ) -> vd::Result<()> {
        let (swapchain, command_fences, _views) = init_swapchain(
            &self.device,
            &self.surface,
            width, height,
            &self.surface_format,
            self.sharing_mode,
            &self.q_indices,
            self.present_mode,
            Some(&self.swapchain), // Pass in old swapchain
        )?;

        let render_pass = init_render_pass(
            &swapchain,
            self.depth_format,
            &self.device,
        )?;

        let pipeline = init_pipeline(
            &swapchain,
            &self.shader_stages,
            &self.assembly,
            &self.rasterizer,
            &self.multisampling,
            &self.pipeline_layout,
            &render_pass,
            &self.device,
        )?;

        #[allow(unused_variables)]
        let debug_data = init_debug(
            &swapchain,
            &render_pass,
            &self.pipeline_layout,
            &self.device,
        )?;

        let (
            _depth_image,
            depth_memory,
            framebuffers,
            ubo_buffer,
            ubo_memory,
            dyn_ubo_buffer,
            dyn_ubo_memory,
            ubo_alignment,
            descriptor_sets,
            _descriptor_pool,
        ) = init_drawing(
            &swapchain,
            self.depth_format,
            &_views,
            &render_pass,
            &self.device,
            &self.transient_pool,
            self.graphics_family,
            self.ubo_layout.handle(),
        )?;

        let command_buffers = init_commands(
            &self.drawing_pool,
            &framebuffers,
        )?;

        // Synchronize
        self.device.wait_idle();

        /* Coup */

        self.swapchain = swapchain;
        self.command_fences = command_fences;
        self.render_pass = render_pass;
        self.pipeline = pipeline;
        self.framebuffers = framebuffers;
        self.ubo_alignment = ubo_alignment;
        self.descriptor_sets = descriptor_sets;
        self.command_buffers = command_buffers;

        unsafe {
            self.free_device_refresh();
        }

        self.depth_memory = depth_memory;
        self.ubo_buffer = ubo_buffer;
        self.ubo_memory = ubo_memory;
        self.dyn_ubo_buffer = dyn_ubo_buffer;
        self.dyn_ubo_memory = dyn_ubo_memory;

        self._depth_image = _depth_image;
        self._views = _views;
        self._descriptor_pool = _descriptor_pool;

        #[cfg(debug_assertions)] {
            self.debug_data = debug_data;
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    pub fn update_debug(&mut self, lines: &[DebugLine]) -> vd::Result<()> {
        /* Copy debug data to GPU */

        unsafe {
            copy_buffer(
                &self.device,
                self.debug_data.as_ref().unwrap().memory,
                (lines.len() * std::mem::size_of::<DebugLine>()) as u64,
                &lines,
            )?;
        }

        // Update debug line count
        self.debug_line_count = lines.len() as u32;

        Ok(())
    }

    pub fn update(
        &mut self,
        instances: &Instances,
        shared_ubo: SharedUBO,
    ) -> vd::Result<()> {
        /* Copy UBOs to GPU */

        unsafe {
            copy_buffer(
                &self.device,
                self.ubo_memory,
                std::mem::size_of::<SharedUBO>() as u64,
                &[shared_ubo],
            )?;
        }

        // Not optimal: requires copies and a heap allocation
        let mut dynamic_buffer = util::AlignedBuffer::<InstanceUBO>::new(
            self.ubo_alignment as usize,
            instances.count(),
        );

        for model in &instances.data {
            for ubo in model {
                dynamic_buffer.push(ubo.clone());
            }
        }

        unsafe {
            copy_buffer(
                &self.device,
                self.dyn_ubo_memory,
                dynamic_buffer.size as u64,
                &dynamic_buffer.finalize(),
            )?;
        }

        Ok(())
    }

    pub fn draw(&self, instances: &Instances) -> vd::Result<()> {
        // Note: will most likely return an image index that is still in use
        let index = self.swapchain.acquire_next_image_khr(
            u64::max_value(), // Disable timeout
            Some(&self.image_available),
            None,
        )?;

        // Get command buffer to use this frame
        let cmd_buffer = &self.command_buffers[index as usize];

        let clears = [
            // Clear color
            vd::ClearValue {
                color: vd::ClearColorValue {
                    float32: [0f32, 0f32, 0f32, 1f32]
                }
            },

            vd::ClearValue {
                depthStencil: vd::vks::VkClearDepthStencilValue {
                    depth: 1., // Initialized to max depth
                    stencil: 0,
                }
            },
        ];

        // Get handle to this command buffer's fence
        let fence = self.command_fences[index as usize].handle();

        unsafe {
            // Wait for command buffer to become available
            self.device.wait_for_fences(
                &[fence],
                false,
                u64::max_value(),
            )?;

            // Unsignal fence
            self.device.reset_fences(&[fence])?;
        }

        // Reset command buffer (now that it's no longer in use)
        cmd_buffer.reset(
            vd::CommandBufferResetFlags::empty(),
        )?;

        /* Build command buffer */

        cmd_buffer.begin(
            vd::CommandBufferUsageFlags::SIMULTANEOUS_USE,
        )?;

        let handle = cmd_buffer.handle();

        debug_assert!(index < self.framebuffers.len() as u32);

        let pass_info = vd::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass.handle())
            .framebuffer(&self.framebuffers[index as usize])
            .render_area(
                vd::Rect2d::builder()
                    .offset(
                        vd::Offset2d::builder()
                            .x(0)
                            .y(0)
                            .build()
                    ).extent(self.swapchain.extent().clone())
                    .build()
            ).clear_values(&clears)
            .build();

        /* Execute render pass */

        cmd_buffer.begin_render_pass(
            &pass_info,
            vd::SubpassContents::Inline,
        );

        cmd_buffer.bind_pipeline(
            vd::PipelineBindPoint::Graphics,
            &self.pipeline.handle(),
        );

        unsafe {
            self.device.cmd_bind_vertex_buffers(
                handle,
                0,
                &[self.vertex_buffer],
                &[0],
            );

            self.device.cmd_bind_index_buffer(
                handle,
                self.index_buffer,
                0,
                vd::IndexType::Uint32,
            );
        }

        debug_assert!(self.models.len() == instances.data.len());

        let mut instance = 0;
        for j in 0..self.models.len() {
            // Render each instance
            for _ in 0..instances.data[j].len() {
                // Bind uniform data
                cmd_buffer.bind_descriptor_sets(
                    vd::PipelineBindPoint::Graphics,
                    &self.pipeline_layout,
                    0,
                    &[&self.descriptor_sets[0]], // Single descriptor set
                    // Offset dynamic uniform buffer
                    &[self.ubo_alignment as u32 * instance as u32],
                );

                // Draw call
                cmd_buffer.draw_indexed(
                    self.models[j].index_count,
                    1,
                    self.models[j].index_offset,
                    0, // No vertex offset
                    0,
                );

                instance += 1;
            }
        }

        #[cfg(debug_assertions)] {
            /* Draw debug data */

            cmd_buffer.bind_pipeline(
                vd::PipelineBindPoint::Graphics,
                &self.debug_data.as_ref().unwrap().pipeline.handle(),
            );

            cmd_buffer.bind_descriptor_sets(
                vd::PipelineBindPoint::Graphics,
                &self.pipeline_layout,
                0,
                &[&self.descriptor_sets[0]], // Single descriptor set
                &[0], // Ignore the dynamic uniform buffer
            );

            unsafe {
                self.device.cmd_bind_vertex_buffers(
                    handle,
                    0,
                    &[self.debug_data.as_ref().unwrap().buffer],
                    &[0],
                );
            }

            for i in 0..self.debug_line_count {
                cmd_buffer.draw(
                    2,
                    1,
                    i * 2,
                    0,
                );
            }
        }

        cmd_buffer.end_render_pass();
        cmd_buffer.end()?;

        /* Submit render and presentation queues */

        // Synchronization primitives
        let available_signals = [self.image_available.handle()];
        let complete_signals = [self.render_complete.handle()];

        let cmd_buffer_handles = [cmd_buffer.handle()];

        // Wait for available images to render to
        let info = vd::SubmitInfo::builder()
            .wait_semaphores(&available_signals)
            .wait_dst_stage_mask(
                &vd::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            ).command_buffers(&cmd_buffer_handles)
            .signal_semaphores(&complete_signals)
            .build();

        match self.device.get_device_queue(self.graphics_family, 0) {
            Some(gq) => {
                unsafe {
                    // Render; pass in fence to signal after done
                    self.device.queue_submit(gq, &[info], Some(fence))?;
                }

                let swapchains = [self.swapchain.handle()];
                let indices = [index];

                let present_q = self.device.get_device_queue(
                    self.present_family,
                    0,
                );

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
                            self.device.queue_present_khr(pq, &info)?;
                        }

                        // Synchronize with GPU in debug mode
                        // (prevents memory leaks from the validation layers)
                        #[cfg(debug_assertions)] {
                            self.device.wait_idle();
                        }
                    },

                    None => return Err("no present queue".into())
                }
            },

            None => return Err("no graphics queue".into())
        }

        Ok(())
    }

    // Free memory allocated on the GPU at init
    unsafe fn free_device_init(&mut self) {
        // Vertex buffer
        self.device.destroy_buffer(self.vertex_buffer, None);
        self.device.free_memory(self.vertex_memory, None);

        // Index buffer
        self.device.destroy_buffer(self.index_buffer, None);
        self.device.free_memory(self.index_memory, None);
    }

    // Free memory allocated on the GPU at refresh
    unsafe fn free_device_refresh(&mut self) {
        // Depth image
        self.device.free_memory(self.depth_memory, None);

        // Uniform buffers
        self.device.destroy_buffer(self.ubo_buffer, None);
        self.device.free_memory(self.ubo_memory, None);
        self.device.destroy_buffer(self.dyn_ubo_buffer, None);
        self.device.free_memory(self.dyn_ubo_memory, None);

        #[cfg(debug_assertions)] {
            /* Debug buffer */

            self.device.destroy_buffer(
                self.debug_data.as_ref().unwrap().buffer,
                None,
            );

            self.device.free_memory(
                self.debug_data.as_ref().unwrap().memory,
                None,
            );
        }
    }
}

impl<'a> Drop for Context<'a> {
    fn drop(&mut self) {
        unsafe {
            self.free_device_refresh();
            self.free_device_init();
        }
    }
}

#[allow(dead_code)]
struct DebugData {
    buffer:   vd::BufferHandle,
    memory:   vd::DeviceMemoryHandle,
    pipeline: vd::GraphicsPipeline,
    _vert:    vd::ShaderModule,
    _frag:    vd::ShaderModule,
}

#[derive(Clone)]
pub struct ModelData {
    pub vertices: Vec<Vertex>,
    pub indices:  Vec<u32>,
}

impl ModelData {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> ModelData {
        ModelData {
            vertices,
            indices,
        }
    }
}

pub struct Model {
    index_count:  u32,
    index_offset: u32,
    vertex_count: usize,
}

impl Model {
    fn new(
        index_count:  u32,
        index_offset: u32,
        vertex_count: usize,
    ) -> Model {
        Model {
            index_count,
            index_offset,
            vertex_count,
        }
    }
}

pub struct Instances {
    data: Vec<Vec<InstanceUBO>>,
}

impl Instances {
    pub fn new(model_count: usize, hints: Option<&[usize]>) -> Instances {
        let mut data = Vec::with_capacity(model_count);

        match hints {
            Some(hints) => {
                assert!(model_count == hints.len());

                for i in 0..model_count {
                    data.push(Vec::with_capacity(hints[i]));
                }
            }

            None => {
                for _ in 0..model_count {
                    data.push(Vec::new());
                }
            }
        };

        Instances { data }
    }

    // Returns handle to new instance
    pub fn add(
        &mut self,
        instance_data: InstanceUBO,
        model_index: usize,
    ) -> InstanceHandle {
        self.data[model_index].push(instance_data);

        InstanceHandle::new(
            model_index as u32,
            (self.data[model_index].len() - 1) as u32,
        )
    }

    // Modify data for an existing instance
    pub fn update(
        &mut self,
        handle: InstanceHandle,
        data: InstanceUBO,
    ) {
        let (m, i) = (
            handle.model_index() as usize,
            handle.instance_index() as usize,
        );

        self.data[m][i] = data;
    }

    // Count instances (O(model_count))
    pub fn count(&self) -> usize {
        let mut count = 0;

        for model in &self.data {
            count += model.len();
        }

        count
    }
}

#[derive(Clone, Copy)]
pub struct InstanceHandle {
    _value: u32,
}

impl InstanceHandle {
    fn new(model_index: u32, instance_index: u32) -> InstanceHandle {
        debug_assert!(model_index >> 16 == 0);
        debug_assert!(instance_index >> 16 == 0);

        InstanceHandle {
            _value: instance_index | (model_index << 16)
        }
    }

    fn model_index(self) -> u16 {
        (self._value >> 16) as u16
    }

    fn instance_index(self) -> u16 {
        self._value as u16
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Vertex {
    pub position: alg::Vec3,
    pub color:    alg::Vec3,
}

impl Vertex {
    pub fn new(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32) -> Vertex {
        Vertex {
            position: alg::Vec3::new(x, y, z),
            color:    alg::Vec3::new(r, g, b),
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
                .format(vd::Format::R32G32B32Sfloat)
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

#[cfg(debug_assertions)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DebugLine {
    start: Vertex,
    end: Vertex,
}

#[cfg(debug_assertions)]
impl DebugLine {
    pub fn new(line: alg::Line, r: f32, g: f32, b: f32) -> DebugLine {
        let start = Vertex::new(
            line.start.x,
            line.start.y,
            line.start.z,
            r,
            g,
            b,
        );

        let end = Vertex::new(
            line.end.x,
            line.end.y,
            line.end.z,
            r,
            g,
            b,
        );

        DebugLine {
            start,
            end,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct PaddedVec3 {
    value: alg::Vec3,
    pad: f32,
}

impl PaddedVec3 {
    pub fn new(value: alg::Vec3) -> PaddedVec3 {
        PaddedVec3 {
            value: value,
            pad: 0.,
        }
    }
}

impl Default for PaddedVec3 {
    fn default() -> PaddedVec3 {
        PaddedVec3::new(alg::Vec3::zero())
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SharedUBO {
    view:       alg::Mat,
    projection: alg::Mat,
}

impl SharedUBO {
    pub fn new(view: alg::Mat, projection: alg::Mat) -> SharedUBO {
        SharedUBO {
            view,
            projection,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct InstanceUBO {
    model: alg::Mat,
    offsets: [PaddedVec3; MAX_SOFTBODY_VERT],
}

impl InstanceUBO {
    pub fn new(
        model: alg::Mat,
        offsets: [PaddedVec3; MAX_SOFTBODY_VERT],
    ) -> InstanceUBO {
        InstanceUBO {
            model,
            offsets,
        }
    }
}

impl Default for InstanceUBO {
    fn default() -> InstanceUBO {
        InstanceUBO {
            model: alg::Mat::identity(),
            offsets: [PaddedVec3::default(); MAX_SOFTBODY_VERT],
        }
    }
}

fn init_vulkan(window: &vdw::winit::Window) -> vd::Result<(
    vd::SurfaceKhr,
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
        .print_debug_report(cfg!(debug_assertions))
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

            // Check for graphics and presentation queue support
            if let Ok((i, j)) = get_q_indices(&device, &surface) {
                physical_device = Some(device);
                graphics_family = i;
                present_family = j;

                break;
            }
        }
    }

    if physical_device.is_none() {
        return Err("no suitable GPUs found".into())
    }

    println!(
        "Device graphics/present families: {}/{}",
        graphics_family,
        present_family,
    );

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

    println!("Swapchain present mode: {:?}", present_mode);

    /* Logical device */

    let graphics_q_create_info = vd::DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_family)
        .queue_priorities(&[1.])
        .build();

    let present_q_create_info = vd::DeviceQueueCreateInfo::builder()
        .queue_family_index(present_family)
        .queue_priorities(&[1.])
        .build();

    // Combine queues if they share the same index
    let mut infos = vec![graphics_q_create_info];
    let mut q_indices = vec![graphics_family];
    let mut sharing_mode = vd::SharingMode::Exclusive;

    if graphics_family != present_family {
        infos.push(present_q_create_info);
        q_indices.push(present_family);
        sharing_mode = vd::SharingMode::Concurrent;
    }

    let features = vd::PhysicalDeviceFeatures::builder()
        .build();

    let device = vd::Device::builder()
        .queue_create_infos(&infos)
        .enabled_features(&features)
        .enabled_extension_names(DEVICE_EXTENSIONS)
        .build(physical_device.clone())?;

    /* Command buffer pools */

    let drawing_pool = vd::CommandPool::builder()
        .queue_family_index(graphics_family)
        .flags(vd::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
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
        graphics_family,
        present_family,
        surface_format,
        present_mode,
        q_indices,
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

fn load_shaders<'a>(device: vd::Device) -> vd::Result<(
    vd::ShaderModule,
    vd::ShaderModule,
    [vd::PipelineShaderStageCreateInfo<'a>; 2],
)> {
    let vert_buffer = vd::util::read_spir_v_file(
        [SHADER_PATH, "vert.spv"].concat()
    )?;

    let frag_buffer = vd::util::read_spir_v_file(
        [SHADER_PATH, "frag.spv"].concat()
    )?;

    let vert_mod = vd::ShaderModule::new(device.clone(), &vert_buffer)?;
    let frag_mod = vd::ShaderModule::new(device, &frag_buffer)?;

    let main = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();

    let vert_stage = vd::PipelineShaderStageCreateInfo::builder()
        .stage(vd::ShaderStageFlags::VERTEX)
        .module(&vert_mod)
        .name(main)
        .build();

    let frag_stage = vd::PipelineShaderStageCreateInfo::builder()
        .stage(vd::ShaderStageFlags::FRAGMENT)
        .module(&frag_mod)
        .name(main)
        .build();

    Ok((
        vert_mod,
        frag_mod,
        [vert_stage, frag_stage],
    ))
}

fn load_models(
    model_data:      Vec<ModelData>,
    device:          &vd::Device,
    transient_pool:  &vd::CommandPool,
    graphics_family: u32,
) -> vd::Result<(
    vd::BufferHandle,
    vd::DeviceMemoryHandle,
    vd::BufferHandle,
    vd::DeviceMemoryHandle,
    Vec<Model>,
)> {
    /* Concatenate model data */

    let (vertices_len, indices_len) = {
        let mut i = 0usize;
        let mut j = 0usize;

        for data in &model_data {
            let len = data.vertices.len();
            i += len;
            j += len / 3;
        }

        (i, j)
    };

    let (vertices, indices, models) = {
        let mut vertices = Vec::with_capacity(vertices_len);
        let mut indices = Vec::with_capacity(indices_len);
        let mut models = Vec::with_capacity(model_data.len());

        let mut offset = 0;

        for mut data in model_data {
            // Destructive
            vertices.append(&mut data.vertices);

            let index_count = data.indices.len() as u32;

            for index in &data.indices {
                indices.push(index + offset);
            }

            models.push(Model::new(index_count, offset, data.vertices.len()));

            offset += index_count;
        }

        (vertices, indices, models)
    };

    /* Vertex buffer */

    let properties = device.physical_device().memory_properties();

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

    Ok((
        vertex_buffer,
        vertex_memory,
        index_buffer,
        index_memory,
        models,
    ))
}

#[cfg(not(debug_assertions))]
#[allow(unused_variables)]
fn init_debug(
    swapchain: &vd::SwapchainKhr,
    render_pass: &vd::RenderPass,
    pipeline_layout: &vd::PipelineLayout,
    device: &vd::Device,
) -> vd::Result<Option<DebugData>> { Ok(None) }

#[cfg(debug_assertions)]
fn init_debug(
    swapchain: &vd::SwapchainKhr,
    render_pass: &vd::RenderPass,
    pipeline_layout: &vd::PipelineLayout,
    device: &vd::Device,
) -> vd::Result<Option<DebugData>> {
    let properties = device.physical_device().memory_properties();

    // Allocate empty debug vertex buffer
    let (buffer, memory) = create_buffer(
        MAX_DEBUG_LINES * 2 * std::mem::size_of::<Vertex>() as u64,
        vd::BufferUsageFlags::VERTEX_BUFFER,
        device,
        vd::MemoryPropertyFlags::HOST_VISIBLE,
        &properties,
    )?;

    /* Load debug shaders */

    let vert_buffer = vd::util::read_spir_v_file(
        [SHADER_PATH, "debug_vert.spv"].concat()
    )?;

    let frag_buffer = vd::util::read_spir_v_file(
        [SHADER_PATH, "debug_frag.spv"].concat()
    )?;

    let vert_mod = vd::ShaderModule::new(device.clone(), &vert_buffer)?;
    let frag_mod = vd::ShaderModule::new(device.clone(), &frag_buffer)?;

    let main = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();

    let vert_stage = vd::PipelineShaderStageCreateInfo::builder()
        .stage(vd::ShaderStageFlags::VERTEX)
        .module(&vert_mod)
        .name(main)
        .build();

    let frag_stage = vd::PipelineShaderStageCreateInfo::builder()
        .stage(vd::ShaderStageFlags::FRAGMENT)
        .module(&frag_mod)
        .name(main)
        .build();

    /* Create debug pipeline */

    let assembly = vd::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vd::PrimitiveTopology::LineList) // Render lines
        .primitive_restart_enable(false)
        .build();

    let rasterizer = vd::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vd::PolygonMode::Line) // Render lines
        .cull_mode(vd::CullModeFlags::NONE)
        .depth_bias_enable(false)
        .line_width(1f32)
        .build();

    let multisampling = vd::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vd::SampleCountFlags::COUNT_1)
        .sample_shading_enable(false)
        .min_sample_shading(1f32)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .build();

    let binding_description = [Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_descriptions();

    let vert_info = vd::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_description)
        .vertex_attribute_descriptions(&attribute_descriptions)
        .build();

    // Don't blend
    let attachments = [
        vd::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(
                vd::ColorComponentFlags::R
                | vd::ColorComponentFlags::G
                | vd::ColorComponentFlags::B
            ).build()
    ];

    let blending = vd::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&attachments)
        .blend_constants([0f32; 4])
        .build();

    // Always draw on top
    let stencil = vd::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vd::CompareOp::Never)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .build();

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

    let pipeline = vd::GraphicsPipeline::builder()
        .stages(&[vert_stage, frag_stage])
        .vertex_input_state(&vert_info)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&blending)
        .depth_stencil_state(&stencil)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .base_pipeline_index(-1)
        .build(device.clone())?;

    let data = DebugData {
        buffer: buffer,
        memory: memory,
        pipeline: pipeline,
        _vert: vert_mod,
        _frag: frag_mod,
    };

    Ok(Some(data))
}

fn init_fixed<'a>(device: vd::Device) -> vd::Result<(
    vd::Format,
    vd::PipelineInputAssemblyStateCreateInfo<'a>,
    vd::PipelineRasterizationStateCreateInfo<'a>,
    vd::PipelineMultisampleStateCreateInfo<'a>,
    vd::DescriptorSetLayout,
    vd::PipelineLayout,
)> {
    /* Depth buffer */

    // Query for depth image format
    let depth_format = {
        let formats = [
            vd::Format::D32Sfloat,
            vd::Format::D32SfloatS8Uint,
            vd::Format::D24UnormS8Uint,
        ];

        let mut format = None;

        for &option in &formats {
            let properties = device.physical_device().format_properties(option);

            // Optimal tiling
            if properties.optimal_tiling_features().contains(
                vd::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            ) {
                format = Some(option);
                break;
            }
        }

        if format.is_none() {
            return Err("GPU does not support required depth format".into());
        }

        format.unwrap()
    };

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

    /* Descriptor set layout */

    let ubo_layout = {
        let shared_binding = vd::DescriptorSetLayoutBinding::builder()
            .binding(0) // First binding
            .descriptor_type(vd::DescriptorType::UniformBuffer)
            .descriptor_count(1) // Single descriptor (UBO)
            .stage_flags(vd::ShaderStageFlags::VERTEX)
            .build();

        let dynamic_binding = vd::DescriptorSetLayoutBinding::builder()
            .binding(1) // Second binding
            .descriptor_type(vd::DescriptorType::UniformBufferDynamic)
            .descriptor_count(1) // Single descriptor (UBO)
            .stage_flags(vd::ShaderStageFlags::VERTEX)
            .build();

        vd::DescriptorSetLayout::builder()
            .bindings(&[shared_binding, dynamic_binding])
            .build(device.clone())?
    };

    let pipeline_layout = vd::PipelineLayout::builder()
        .set_layouts(&[ubo_layout.handle()])
        .build(device)?;

    // Dependent on DYNAMIC_UBO_WIDTH
    println!("Max softbody vertices: {}", MAX_SOFTBODY_VERT);

    Ok((
        depth_format,
        assembly,
        rasterizer,
        multisampling,
        ubo_layout,
        pipeline_layout,
    ))
}

fn init_swapchain(
    device:          &vd::Device,
    surface:         &vd::SurfaceKhr,
    window_width:    u32,
    window_height:   u32,
    surface_format:  &vd::SurfaceFormatKhr,
    sharing_mode:    vd::SharingMode,
    indices:         &[u32],
    present_mode:    vd::PresentModeKhr,
    old_swapchain:   Option<&vd::SwapchainKhr>,
) -> vd::Result<(
    vd::SwapchainKhr,
    Vec<vd::Fence>,
    Vec<vd::ImageView>,
)> {
    /* Surface */

    let capabilities = device.physical_device().surface_capabilities_khr(
        surface
    )?;

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

    println!("Swapchain image count: {}", image_count);

    // Create command buffer fences, given image count
    let mut command_fences = Vec::with_capacity(image_count as usize);

    for _ in 0..image_count {
        command_fences.push(
            vd::Fence::new(device.clone(), vd::FenceCreateFlags::SIGNALED)?,
        );
    }

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
            .surface(surface)
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

    Ok((swapchain, command_fences, views))
}

fn init_render_pass(
    swapchain:    &vd::SwapchainKhr,
    depth_format: vd::Format,
    device:       &vd::Device
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

    let depth_attachment = vd::AttachmentDescription::builder()
        .format(depth_format)
        .samples(vd::SampleCountFlags::COUNT_1)
        .load_op(vd::AttachmentLoadOp::Clear)
        .store_op(vd::AttachmentStoreOp::DontCare)
        .stencil_load_op(vd::AttachmentLoadOp::DontCare)
        .stencil_store_op(vd::AttachmentStoreOp::DontCare)
        .initial_layout(vd::ImageLayout::Undefined)
        .final_layout(vd::ImageLayout::DepthStencilAttachmentOptimal)
        .build();

    let color_refs = [
        vd::AttachmentReference::builder()
            .attachment(0)
            .layout(vd::ImageLayout::ColorAttachmentOptimal)
            .build(),
    ];

    let depth_ref = vd::AttachmentReference::builder()
        .attachment(1)
        .layout(vd::ImageLayout::DepthStencilAttachmentOptimal)
        .build();

    let subpass = vd::SubpassDescription::builder()
        .pipeline_bind_point(vd::PipelineBindPoint::Graphics)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref)
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
            .attachments(&[color_attachment, depth_attachment])
            .subpasses(&[subpass])
            .dependencies(&[dependency])
            .build(device.clone())?
    )
}

fn init_pipeline(
    swapchain:       &vd::SwapchainKhr,
    stages:          &[vd::PipelineShaderStageCreateInfo; 2],
    assembly:        &vd::PipelineInputAssemblyStateCreateInfo,
    rasterizer:      &vd::PipelineRasterizationStateCreateInfo,
    multisampling:   &vd::PipelineMultisampleStateCreateInfo,
    pipeline_layout: &vd::PipelineLayout,
    render_pass:     &vd::RenderPass,
    device:          &vd::Device,
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

    let stencil = vd::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vd::CompareOp::Less) // Closer fragments, lower depth
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
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
        .depth_stencil_state(&stencil)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .base_pipeline_index(-1)
        .build(device.clone())?
    )
}

fn init_drawing(
    swapchain:       &vd::SwapchainKhr,
    depth_format:    vd::Format,
    views:           &[vd::ImageView],
    render_pass:     &vd::RenderPass,
    device:          &vd::Device,
    transient_pool:  &vd::CommandPool,
    graphics_family: u32,
    ubo_layout:      vd::DescriptorSetLayoutHandle,
) -> vd::Result<(
    vd::Image,
    vd::DeviceMemoryHandle,
    Vec<vd::Framebuffer>,
    vd::BufferHandle,
    vd::DeviceMemoryHandle,
    vd::BufferHandle,
    vd::DeviceMemoryHandle,
    u64,
    Vec<vd::DescriptorSet>,
    vd::DescriptorPool,
)> {
    /* Depth buffer */

    let extent = vd::Extent3d::builder()
        .width(swapchain.extent().width())
        .height(swapchain.extent().height())
        .depth(1)
        .build();

    let depth_image = vd::Image::builder()
        .image_type(vd::ImageType::Type2d)
        .format(depth_format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vd::SampleCountFlags::COUNT_1)
        .tiling(vd::ImageTiling::Optimal)
        .usage(vd::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vd::SharingMode::Exclusive)
        .initial_layout(vd::ImageLayout::Undefined)
        .build(device.clone())?;

    let requirements = unsafe {
        device.get_image_memory_requirements(depth_image.handle())
    };

    let properties = device.physical_device().memory_properties();

    // Allocate space for depth image on GPU
    let depth_info = vd::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size())
        .memory_type_index(
            get_memory_type(
                requirements.memory_type_bits(),
                vd::MemoryPropertyFlags::DEVICE_LOCAL,
                properties.memory_types(),
            )?
        ).build();

    let depth_memory_handle = unsafe {
        device.allocate_memory(&depth_info, None)?
    };

    // Bind depth image to GPU
    unsafe {
        device.bind_image_memory(
            depth_image.handle(),
            depth_memory_handle,
            0,
        )?;
    }

    let depth_view = vd::ImageView::builder()
        .image(depth_image.handle())
        .view_type(vd::ImageViewType::Type2d)
        .format(depth_format)
        .components(vd::ComponentMapping::default())
        .subresource_range(
            vd::ImageSubresourceRange::builder()
                .aspect_mask(vd::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build()
        ).build(device.clone(), None)?;

    /* Transition depth image layout */

    let transfer_buffer = get_transfer_buffer(transient_pool)?;

    let mut flags = vd::ImageAspectFlags::DEPTH;

    if depth_format == vd::Format::D32SfloatS8Uint
        || depth_format == vd::Format::D24UnormS8Uint
    {
        flags |= vd::ImageAspectFlags::STENCIL;
    }

    let subresource_range = vd::ImageSubresourceRange::builder()
        .aspect_mask(flags)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let barrier = vd::ImageMemoryBarrier::builder()
        .src_access_mask(vd::AccessFlags::empty())
        .dst_access_mask(
            vd::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vd::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        ).old_layout(vd::ImageLayout::Undefined)
        .new_layout(vd::ImageLayout::DepthStencilAttachmentOptimal)
        .src_queue_family_index(vd::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vd::QUEUE_FAMILY_IGNORED)
        .image(&depth_image)
        .subresource_range(subresource_range)
        .build();

    transfer_buffer.pipeline_barrier(
        vd::PipelineStageFlags::TOP_OF_PIPE,
        vd::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        vd::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );

    end_transfer_buffer(&transfer_buffer, device, graphics_family)?;

    /* Framebuffers */

    let mut framebuffers = Vec::with_capacity(views.len());

    for view in views {
        let attachments = [view, &depth_view];

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

    /* Uniform buffers */

    let pool_sizes = {
        let size = vd::DescriptorPoolSize::builder()
            .type_of(vd::DescriptorType::UniformBuffer)
            .descriptor_count(1) // Shared by all models
            .build();

        let dynamic_size = vd::DescriptorPoolSize::builder()
            .type_of(vd::DescriptorType::UniformBufferDynamic)
            .descriptor_count(1) // Shared by all models
            .build();

        [size, dynamic_size]
    };

    let descriptor_pool = vd::DescriptorPool::builder()
        .pool_sizes(&pool_sizes)
        .flags(vd::DescriptorPoolCreateFlags::empty())
        .max_sets(1)
        .build(device.clone())?;

    // Each set will contain two descriptors
    let sets = descriptor_pool.allocate_descriptor_sets(&[ubo_layout])?;

    debug_assert!(sets.len() == 1);

    let minimum_alignment = device
        .physical_device()
        .properties()
        .limits()
        .min_uniform_buffer_offset_alignment();

    if minimum_alignment == 0 {
        return Err(
            "Invalid minimum uniform buffer offset alignment".into()
        );
    }

    // Compute maximum possible alignment
    let ubo_alignment = |preferred| {
        (preferred + minimum_alignment - 1) & !(minimum_alignment - 1)
    };

    /* Shared */

    let shared_alignment = ubo_alignment(
        std::mem::size_of::<SharedUBO>() as u64
    );

    // Allocate a buffer for the shared UBO
    let (ubo_buffer, ubo_memory) = create_buffer(
        shared_alignment, // Contains single UBO
        vd::BufferUsageFlags::UNIFORM_BUFFER,
        device,
        vd::MemoryPropertyFlags::HOST_VISIBLE
        | vd::MemoryPropertyFlags::HOST_COHERENT,
        &properties,
    )?;

    let shared_info = vd::DescriptorBufferInfo::builder()
        .buffer(ubo_buffer)
        .offset(0)
        .range(shared_alignment)
        .build();

    /* Dynamic */

    debug_assert!(std::mem::size_of::<InstanceUBO>() == DYNAMIC_UBO_WIDTH);

    // Can't guarantee the minimum alignment will equal DYNAMIC_UBO_WIDTH,
    // even though it probably will.
    let dynamic_alignment = ubo_alignment(DYNAMIC_UBO_WIDTH as u64);

    let dynamic_size = MAX_INSTANCES * dynamic_alignment;

    // Allocate a single buffer for the remaining UBOs
    let (dyn_ubo_buffer, dyn_ubo_memory) = create_buffer(
        dynamic_size,
        vd::BufferUsageFlags::UNIFORM_BUFFER,
        device,
        vd::MemoryPropertyFlags::HOST_VISIBLE,
        &properties,
    )?;

    let dynamic_info = vd::DescriptorBufferInfo::builder()
        .buffer(dyn_ubo_buffer)
        .offset(0)
        .range(dynamic_alignment)
        .build();

    // Write shared and dynamic UBOs
    let writes = [
        vd::WriteDescriptorSet::builder()
            .dst_set(sets[0])
            .dst_binding(0) // First binding
            .dst_array_element(0)
            .descriptor_count(1)
            .descriptor_type(vd::DescriptorType::UniformBuffer)
            .buffer_info(&shared_info)
            .build(),
        vd::WriteDescriptorSet::builder()
            .dst_set(sets[0])
            .dst_binding(1) // Second binding
            .dst_array_element(0)
            .descriptor_count(1)
            .descriptor_type(vd::DescriptorType::UniformBufferDynamic)
            .buffer_info(&dynamic_info)
            .build(),
    ];

    // No copies (causes segfault?)
    descriptor_pool.update_descriptor_sets(&writes, &[]);

    Ok((
        depth_image,
        depth_memory_handle,
        framebuffers,
        ubo_buffer,
        ubo_memory,
        dyn_ubo_buffer,
        dyn_ubo_memory,
        dynamic_alignment,
        sets.into_vec(),
        descriptor_pool,
    ))
}

fn init_commands(
    drawing_pool: &vd::CommandPool,
    framebuffers: &[vd::Framebuffer],
) -> vd::Result<Vec<vd::CommandBuffer>> {
    // Allocate command buffers from pool
    let command_buffers = drawing_pool.allocate_command_buffers(
        vd::CommandBufferLevel::Primary,
        framebuffers.len() as u32,
    )?;

    Ok(command_buffers.into_vec())
}

fn get_transfer_buffer(
    transient_pool: &vd::CommandPool,
) -> vd::Result<vd::CommandBuffer> {
    let buffer = transient_pool.allocate_command_buffer(
        vd::CommandBufferLevel::Primary,
    )?;

    buffer.begin(
        vd::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
    )?;

    Ok(buffer)
}

fn end_transfer_buffer(
    buffer:          &vd::CommandBuffer,
    device:          &vd::Device,
    graphics_family: u32,
) -> vd::Result<()> {
    buffer.end()?;

    let handles = [buffer.handle()];

    let info = vd::SubmitInfo::builder()
        .command_buffers(&handles)
        .build();

    match device.get_device_queue(graphics_family, 0) {
        Some(gq) => unsafe {
            device.queue_submit(gq, &[info], None)?;
        },

        None => return Err("no graphics queue".into())
    }

    // Block until transfer completion
    device.wait_idle();

    Ok(())
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
    // Length of slice * length of data type
    let size = std::mem::size_of_val(data) as u64;

    // Local buffer
    let (host_buffer, host_memory) = create_buffer(
        size,
        vd::BufferUsageFlags::TRANSFER_SRC,
        device,
        vd::MemoryPropertyFlags::HOST_VISIBLE
        | vd::MemoryPropertyFlags::HOST_COHERENT,
        properties,
    )?;

    // Transfer data to host (source)
    unsafe {
        copy_buffer(device, host_memory, size, data)?;
    }

    // GPU buffer (destination)
    let (device_buffer, device_memory) = create_buffer(
        size,
        usage | vd::BufferUsageFlags::TRANSFER_DST,
        device,
        vd::MemoryPropertyFlags::DEVICE_LOCAL,
        properties,
    )?;

    let transfer_buffer = get_transfer_buffer(transient_pool)?;

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

    end_transfer_buffer(&transfer_buffer, device, graphics_family)?;

    // Clean up host buffer
    unsafe {
        device.destroy_buffer(host_buffer, None);
        device.free_memory(host_memory, None);
    }

    Ok((device_buffer, device_memory))
}

// Allocate (empty) buffer on the GPU
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

// Memory-mapped IO
unsafe fn copy_buffer<T: std::marker::Copy>(
    device: &vd::Device,
    memory: vd::DeviceMemoryHandle,
    size:   u64,
    data:   &[T],
) -> vd::Result<()> {
    let ptr = device.map_memory(
        memory,
        0,
        size,
        vd::MemoryMapFlags::empty(),
    )?;

    let destination = std::slice::from_raw_parts_mut(
        ptr,
        data.len()
    );

    // Copy data
    destination.copy_from_slice(data);

    device.unmap_memory(memory);

    Ok(())
}
