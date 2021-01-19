extern crate fnv;

use render;
use entity;
use components;

use components::transform;
use components::bitmap;

/// Builder pattern for text
pub struct TextBuilder<'a> {
    manager: &'a mut Manager,
    text: bitmap::Text,
}

impl<'a> TextBuilder<'a> {
    pub fn new(manager: &'a mut Manager) -> TextBuilder<'a> {
        TextBuilder {
            manager,
            text: bitmap::Text::empty_2d_instance(),
        }
    }

    pub fn text(&mut self, text: &str) -> &mut TextBuilder<'a> {
        self.text.set_str(text);
        self
    }

    pub fn alignment(&mut self, alignment: bitmap::TextAlign) -> &mut TextBuilder<'a> {
        self.text.align = alignment;
        self
    }

    pub fn anchor_x(&mut self, anchor_ratio: f32) -> &mut TextBuilder<'a> {
        self.text.anchor_x = anchor_ratio;
        self
    }

    pub fn anchor_y(&mut self, anchor_ratio: f32) -> &mut TextBuilder<'a> {
        self.text.anchor_y = anchor_ratio;
        self
    }

    /**
     * Scaling with reference to each pixel of texture
     */
    pub fn pixel_scale_factor(&mut self, scale_factor: f32) -> &mut TextBuilder<'a> {
        self.text.scale = bitmap::TextScale::Pixel;
        self.text.scale_factor = scale_factor;
        self
    }

    /**
     * Scaling with respect to aspect ratio
     */
    pub fn aspect_scale_factor(&mut self, scale_factor: f32) -> &mut TextBuilder<'a> {
        self.text.scale = bitmap::TextScale::Aspect;
        self.text.scale_factor = scale_factor;
        self
    }

    /// Finalize
    pub fn for_entity(&mut self, entity: entity::Handle) {
        #[cfg(debug_assertions)] {
            if self.text.text.is_empty() {
                eprintln!("Warning: Text string is empty");
            }
        }

        self.manager.set(entity, self.text.clone());
    }
}

pub struct Manager {
    instances: fnv::FnvHashMap<entity::Handle, bitmap::Text>,
    pub instance_data: Vec<render::FontUBO>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        self.instances.insert(
            entity,
            bitmap::Text::empty_2d_instance(),
        );
    }

    fn registered(&self, entity: entity::Handle) -> bool {
        self.instances.contains_key(&entity)
    }

    fn count(&self) -> usize {
        self.instances.len()
    }

    #[cfg(debug_assertions)] fn debug_name(&self) -> &str { "Label" }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            instances: fnv::FnvHashMap::with_capacity_and_hasher(
                hint,
                Default::default(),
            ),
            instance_data: Vec::with_capacity(hint),
        }
    }

    pub fn build(&mut self) -> TextBuilder {
        TextBuilder::new(self)
    }

    fn set(&mut self, entity: entity::Handle, text: bitmap::Text) {
        debug_validate_entity!(self, entity);
        *self.instances.get_mut(&entity).unwrap() = text;
    }

    /// Changes string that the label will render
    pub fn set_str(&mut self, entity: entity::Handle, str: &str) {
        debug_validate_entity!(self, entity);
        let instance = self.instances.get_mut(&entity).unwrap();
        instance.set_str(str);
    }

    pub(crate) fn update(
        &mut self,
        transforms: &transform::Manager,
        screen: ::ScreenData,
    ) {
        let inv_aspect = screen.height as f32 / screen.width as f32;
        self.instance_data.clear();

        for (entity, _) in &mut self.instances {
            // Initialize projection * model matrix to model matrix
            let mut proj_model = transforms.get_mat(*entity);
            // TODO: Translate by anchors
            // let framebuffer_height = screen.height;
            // println!("{}", *framebuffer_height);

            // Scale X axis by inverse aspect ratio
            proj_model.x0 = inv_aspect * proj_model.x0;
            proj_model.x1 = inv_aspect * proj_model.x1;
            proj_model.x2 = inv_aspect * proj_model.x2;
            proj_model.x3 = inv_aspect * proj_model.x3;

            let font_ubo = render::FontUBO {
                model: proj_model,
            };

            self.instance_data.push(font_ubo);
        }
    }

    pub(crate) fn prepare_bitmap_text(
        &mut self,
        vertex_ptr: *mut *mut *mut render::FontVertex_2d,
        idx_ptr: *mut *mut u32,
        framebuffer_height: u32,
        text_instances: &mut Vec<render::TextInstance>,
    ) {
        // Calls function that shares functionality with other types of text
        for (_, text_instance) in self.instances.iter() {
            let mut idx_offset = 0u32;
            let char_scale = bitmap::get_2d_char_scale(
                text_instance,
                framebuffer_height,
            );
            let quads_props = bitmap::prepare_text::<render::FontVertex_2d>(
                text_instance,
                idx_ptr,
                &mut &mut idx_offset,
                text_instances,
                char_scale,
            );
            bitmap::write_vertices_2d(
                quads_props,
                vertex_ptr,
            );
        }
    }
}
