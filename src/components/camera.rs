use alg;
use entity;
use components;
use render;

use components::transform;
use components::Component;

pub struct Camera { }

pub struct Manager {
    active: usize,
    // There will likely be few cameras
    instances: Vec<(entity::Handle, Camera)>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        self.instances.push(
            (
                entity,
                Camera { },
            )
        );
    }

    fn count(&self) -> usize {
        self.instances.len()
    }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            active: 0,
            instances: Vec::with_capacity(hint),
        }
    }
}
