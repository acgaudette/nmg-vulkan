pub mod transform;
pub mod draw;

use ecs;

pub trait Component {
    fn register(&mut self, entity: ecs::EntityHandle);
    fn count(&self) -> usize;
}
