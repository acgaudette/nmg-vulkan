extern crate fnv;

use render;
use entity;
use components;
use font;

use components::transform;
use components::bitmap;

/// Builder pattern for text
pub struct TextBuilder<'a> {
    manager: &'a mut Manager,
    text: render::Text,
}

impl<'a> TextBuilder<'a> {
    pub fn new(manager: &'a mut Manager) -> TextBuilder<'a> {
        TextBuilder {
            manager,
            text: render::Text::empty_2d_instance(),
        }
    }

    pub fn text(&mut self, text: &str) -> &mut TextBuilder<'a> {
        self.text.text = text.into();
        self
    }

    pub fn alignment(&mut self, alignment: render::TextAlign) -> &mut TextBuilder<'a> {
        self.text.align = alignment;
        self
    }

    /**
     * Scaling with reference to each pixel of texture
     */
    pub fn pixel_scale_factor(&mut self, scale_factor: f32) -> &mut TextBuilder<'a> {
        self.text.scale = render::TextScale::Pixel;
        self.text.scale_factor = scale_factor;
        self
    }

    /**
     * Scaling with respect to aspect ratio
     */
    pub fn aspect_scale_factor(&mut self, scale_factor: f32) -> &mut TextBuilder<'a> {
        self.text.scale = render::TextScale::Aspect;
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
    instances: fnv::FnvHashMap<entity::Handle, render::Text>,
    pub instance_data: Vec<render::FontUBO>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        self.instances.insert(
            entity,
            render::Text::empty_2d_instance(),
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

    fn set(&mut self, entity: entity::Handle, text: render::Text) {
        debug_validate_entity!(self, entity);
        *self.instances.get_mut(&entity).unwrap() = text;
    }

    /// Changes string that the label will render
    pub fn set_str(&mut self, entity: entity::Handle, str: &str) {
        debug_validate_entity!(self, entity);
        let instance = self.instances.get_mut(&entity).unwrap();
        instance.text = str.to_string();
    }

    pub(crate) fn update(&mut self, transforms: &transform::Manager) {
        self.instance_data.clear();
        for (entity, _) in &mut self.instances {
            let font_ubo = render::FontUBO {
                model: transforms.get_mat(*entity)
            };
            self.instance_data.push(font_ubo);
        }
    }

    pub(crate) fn prepare_bitmap_text(
        &mut self,
        font_data: &font::Data,
        vertex_ptr: *mut *mut *mut render::FontVertex_2d,
        idx_ptr: *mut *mut u32,
        framebuffer_width:  u32,
        framebuffer_height: u32,
        text_instances: &mut Vec<render::TextInstance>,
    ) {
        // Calls function that shares functionality with other types of text
        for (_, text_instance) in self.instances.iter() {
            let mut idx_offset = 0u32;
            bitmap::prepare_text(
                text_instance,
                font_data,
                vertex_ptr,
                idx_ptr,
                &mut &mut idx_offset,
                framebuffer_width,
                framebuffer_height,
                text_instances,
            );
        }
    }
}
