use std;

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Handle {
    _value: u32,
}

impl Handle {
    fn new(index: u32) -> Handle {
        Handle {
            _value: index,
        }
    }

    pub fn get_index(self) -> u32 {
        self._value
    }
}

impl std::fmt::Display for Handle {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "{}", self._value)
    }
}

pub struct Manager {
    data:  std::collections::HashSet<Handle>,
    index: u32,
    count: u32,
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            data:  std::collections::HashSet::with_capacity(hint),
            index: 0,
            count: 0,
        }
    }

    pub fn add(&mut self) -> Handle {
        if self.count == u32::max_value() {
            panic!("Out of space for new entities!");
        }

        loop {
            // If an entity already exists, skip
            if self.check(Handle::new(self.index)) {
                self.index = self.index.wrapping_add(1);
                continue;
            }

            break;
        }

        // Add new entity
        let handle = Handle::new(self.index);
        self.data.insert(handle);
        self.count += 1;

        // Offset for next time
        self.index = self.index.wrapping_add(1);

        handle
    }

    pub fn check(&self, handle: Handle) -> bool {
        self.data.contains(&handle)
    }

    // Idempotent--but access a handle after remove() at your own risk!
    pub fn remove(&mut self, handle: Handle) {
        if self.data.remove(&handle) {
            // Decrement counter
            self.count -= 1;
        }
    }

    pub fn count(&self) -> usize {
        self.data.len()
    }
}
