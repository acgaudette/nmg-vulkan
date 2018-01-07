pub mod transform;
pub mod draw;
pub mod rigidbody;
pub mod softbody;

use entity;

pub trait Component {
    fn register(&mut self, entity: entity::Handle);
    fn count(&self) -> usize;
}

pub struct Container {
    pub transforms:  transform::Manager,
    pub draws:       draw::Manager,
    pub rigidbodies: rigidbody::Manager,
}
