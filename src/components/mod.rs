macro_rules! debug_validate_entity {
    ($component: path, $entity: expr) => {
        #[cfg(debug_assertions)] {
            use components::Component;

            if !$component.registered($entity) {
                let call = fn_name!();

                panic!(
                    "{} component not found for entity {} in {}(...)",
                    $component.debug_name(),
                    $entity,
                    &call[12..call.len()]
                );
            }
        }
    }
}

pub mod transform;
pub mod camera;
pub mod light;
pub mod draw;
pub mod softbody;
pub mod bitmap;
pub mod text;
pub mod label;

use entity;

pub trait Component {
    fn register(&mut self, entity: entity::Handle);
    fn registered(&self, entity: entity::Handle) -> bool;
    fn count(&self) -> usize;

    #[cfg(debug_assertions)]
    fn debug_name(&self) -> &str;
}

pub struct Container {
    pub transforms: transform::Manager,
    pub cameras:    camera::Manager,
    pub lights:     light::Manager,
    pub draws:      draw::Manager,
    pub softbodies: softbody::Manager,
    pub texts:      text::Manager,
    pub labels:     label::Manager,
}
