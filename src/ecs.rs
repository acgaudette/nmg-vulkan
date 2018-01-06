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

    pub fn add(&mut self) -> EntityHandle {
        loop {
            // If an entity already exists, skip
            if self.check(EntityHandle::new(self.index)) {
                self.index += 1;
                continue;
            }

            break;
        }

        // Add new entity
        let handle = EntityHandle::new(self.index);
        self.data.insert(handle);

        // Offset for next time
        self.index += 1;

        handle
    }

    pub fn check(&self, handle: EntityHandle) -> bool {
        self.data.contains(&handle)
    }

    pub fn remove(&mut self, handle: EntityHandle) {
        self.data.remove(&handle);
    }
}
