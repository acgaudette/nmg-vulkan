use std;
use alg;
use render;
use entity;
use components;

pub struct Manager {
    lights: std::collections::HashMap<entity::Handle, render::Light>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        self.lights.insert(
            entity,
            render::Light::none(),
        );
    }

    fn count(&self) -> usize {
        self.lights.len()
    }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            lights: std::collections::HashMap::with_capacity(hint),
        }
    }
}
