macro_rules! debug_validate_entity {
    ($component: path, $entity: path) => {
        #[cfg(debug_assertions)] {
            if !$component.registered($entity) {
                let call = fn_name!();

                panic!(
                    "Component not found for entity {} in {}(...)",
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

use entity;

pub trait Component {
    fn register(&mut self, entity: entity::Handle);
    fn registered(&self, entity: entity::Handle) -> bool;
    fn count(&self) -> usize;
}

pub struct Container {
    pub transforms: transform::Manager,
    pub cameras:    camera::Manager,
    pub lights:     light::Manager,
    pub draws:      draw::Manager,
    pub softbodies: softbody::Manager,
}
