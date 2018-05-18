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

    pub fn set(&mut self, entity: entity::Handle, light: render::Light) {
        debug_assert!(self.lights.contains_key(&entity));
        *self.lights.get_mut(&entity).unwrap() = light;
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

        for (_, light) in &self.lights {
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
