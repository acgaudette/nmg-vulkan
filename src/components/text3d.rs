extern crate fnv;

use render;
use entity;
use components;

use components::transform;
use components::text;

use font::*;
/// Builder pattern for text
pub struct TextBuilder<'a> {
    manager: &'a mut Manager,
    text: render::Text,
}

impl<'a> TextBuilder<'a> {
    pub fn new(manager: &'a mut Manager) -> TextBuilder<'a> {
        TextBuilder {
            manager,
            text: render::Text::empty_3d_instance(),
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

    pub fn scale_factor(&mut self, scale_factor: f32) -> &mut TextBuilder<'a> {
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
}

impl components::Component for Manager {

    fn register(&mut self, entity: entity::Handle) {
        self.instances.insert(
            entity,
            render::Text::empty_3d_instance(),
        );
    }

    fn registered(&self, entity: entity::Handle) -> bool {
        self.instances.contains_key(&entity)
    }

    fn count(&self) -> usize {
        self.instances.len()
    }

    #[cfg(debug_assertions)] fn debug_name(&self) -> &str { "Text" }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            instances: fnv::FnvHashMap::with_capacity_and_hasher(
                hint,
                Default::default(),
            ),
        }
    }

    pub fn build(&mut self) -> TextBuilder {
        TextBuilder::new(self)
    }

    fn set(&mut self, entity: entity::Handle, text: render::Text) {
        debug_validate_entity!(self, entity);
        *self.instances.get_mut(&entity).unwrap() = text;
    }

    // Update point light positions from transform component
    pub(crate) fn update(&mut self, transforms: &transform::Manager) {
        for (entity, text3d) in &mut self.instances {
            text3d.position = transforms.get_position(*entity);
        }
    }

    //TODO: Implement add_text functionality from render
    // Add parameters here instead of just self
    pub fn prepare_bitmap_text(
        &mut self,
        font_data: &Font,
        ptr: *mut *mut render::FontData,
        framebuffer_width:  u32,
        framebuffer_height: u32,
        num_letters: *mut u64,
    ) {
        text::prepare_bitmap_text(
            &mut self.instances,
            font_data,
            ptr,
            framebuffer_width,
            framebuffer_height,
            num_letters,
        );
    }
}
