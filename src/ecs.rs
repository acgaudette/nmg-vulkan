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

    pub fn get_index(self) -> u32 {
        self._value
    }
}

impl std::fmt::Display for EntityHandle {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "{}", self._value)
    }
}

pub struct Entities {
    data:  std::collections::HashSet<EntityHandle>,
    index: u32,
    count: u32,
}

impl Entities {
    pub fn new(hint: usize) -> Entities {
        Entities {
            data:  std::collections::HashSet::with_capacity(hint),
            index: 0,
            count: 0,
        }
    }

    pub fn add(&mut self) -> EntityHandle {
        if self.count == u32::max_value() {
            panic!("Out of space for new entities!");
        }

        loop {
            // If an entity already exists, skip
            if self.check(EntityHandle::new(self.index)) {
                self.index = self.index.wrapping_add(1);
                continue;
            }

            break;
        }

        // Add new entity
        let handle = EntityHandle::new(self.index);
        self.data.insert(handle);
        self.count += 1;

        // Offset for next time
        self.index = self.index.wrapping_add(1);

        handle
    }

    pub fn check(&self, handle: EntityHandle) -> bool {
        self.data.contains(&handle)
    }

    // Idempotent--but access a handle after remove() at your own risk!
    pub fn remove(&mut self, handle: EntityHandle) {
        if self.data.remove(&handle) {
            // Decrement counter
            self.count -= 1;
        }
    }
}
