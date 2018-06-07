pub mod transform;
pub mod camera;
pub mod light;
pub mod draw;
pub mod softbody;

use entity;

pub trait Component {
    fn register(&mut self, entity: entity::Handle);
    fn count(&self) -> usize;
}

pub struct Container {
    pub transforms: transform::Manager,
    pub cameras:    camera::Manager,
    pub lights:     light::Manager,
    pub draws:      draw::Manager,
    pub softbodies: softbody::Manager,
}
