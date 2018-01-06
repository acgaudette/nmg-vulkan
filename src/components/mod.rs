pub mod transform;
pub mod draw;

use entity;

pub trait Component {
    fn register(&mut self, entity: entity::EntityHandle);
    fn count(&self) -> usize;
}
