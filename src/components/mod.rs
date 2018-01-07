pub mod transform;
pub mod draw;
pub mod rigidbody;

use entity;

pub trait Component {
    fn register(&mut self, entity: entity::Handle);
    fn count(&self) -> usize;
}
