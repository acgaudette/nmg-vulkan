use std;

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct EntityHandle {
    _value: u32,
}

impl EntityHandle {
    fn new(index: u32) -> EntityHandle {
        EntityHandle {
            _value: index,
        }
    }
}
