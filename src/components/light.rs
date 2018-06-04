use std;
use alg;
use render;
use entity;
use components;

use components::transform;

pub struct Manager {
    instances: std::collections::HashMap<entity::Handle, render::Light>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        self.instances.insert(
            entity,
            render::Light::none(),
        );
    }

    fn count(&self) -> usize {
        self.instances.len()
    }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            instances: std::collections::HashMap::with_capacity(hint),
        }
    }

    fn set(&mut self, entity: entity::Handle, light: render::Light) {
        debug_assert!(self.instances.contains_key(&entity));
        *self.instances.get_mut(&entity).unwrap() = light;
    }

    // Update point light positions from transform component
    pub fn update(&mut self, transforms: &transform::Manager) {
        for (entity, light) in &mut self.instances {
            if light.radius > 0.0 {
                light.vector = transforms.get_position(*entity);
            }
        }
    }

    // Given a position, return the set of lights affecting it
    pub fn cull(
        &self,
        position: alg::Vec3,
    ) -> [render::Light; render::MAX_INSTANCE_LIGHTS] {
        let mut instance_lights = [
            render::Light::none();
            render::MAX_INSTANCE_LIGHTS
        ];

        let mut i = 0;

        for (_, light) in &self.instances {
            // Directional
            if light.radius == -1.0 {
                instance_lights[i] = *light; // Set light
                i = i + 1;
            }

            // Dummy light
            else if light.radius == 0.0 {
                continue;
            }

            // Point light--check radius for containment
            else if light.radius > position.dist(light.vector) {
                instance_lights[i] = *light; // Set light
                i = i + 1;
            }

            // Exit after the number of lights per instance is exceeded
            if i == render::MAX_INSTANCE_LIGHTS {
                break;
            }
        }

        instance_lights
    }
}
