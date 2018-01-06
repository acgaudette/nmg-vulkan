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

pub struct Entities {
    data: std::collections::HashSet<EntityHandle>,
    index: u32,
}

impl Entities {
    pub fn new(hint: usize) -> Entities {
        Entities {
            data:  std::collections::HashSet::with_capacity(hint),
            index: 0,
        }
    }
}
