use std;
use alg;
use render;
use graphics;
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

/// Builder pattern for lights
pub struct LightBuilder<'a> {
    manager: &'a mut Manager,
    light: render::Light,
}

impl<'a> LightBuilder<'a> {
    pub fn new(manager: &'a mut Manager) -> LightBuilder<'a> {
        LightBuilder {
            manager,
            light: render::Light {
                vector: alg::Vec3::zero(),
                intensity: 1.0,
                color: graphics::Color::white(),
                radius: 0.0,
            },
        }
    }

    /// Create directional light with given vector \
    /// Not compatible with `point_with_radius(...)`
    pub fn directional(
        &mut self,
        direction: alg::Vec3,
    ) -> &mut LightBuilder<'a> {
        self.light.vector = -direction.norm();
        self.light.radius = -1.0; // Sentinel
        self
    }

    /// Create point light with given radius \
    /// Position is taken from the associated transform component \
    /// Not compatible with `directional(...)`
    pub fn point_with_radius(
        &mut self,
        radius: f32,
    ) -> &mut LightBuilder<'a> {
        self.light.radius = radius;
        self
    }

    pub fn color(&mut self, color: graphics::Color) -> &mut LightBuilder<'a> {
        self.light.color = color;
        self
    }

    pub fn intensity(&mut self, intensity: f32) -> &mut LightBuilder<'a> {
        self.light.intensity = intensity;
        self
    }

    /// Finalize
    pub fn for_entity(&mut self, entity: entity::Handle) {
        debug_assert!(self.light.radius != 0.0);
        debug_assert!(
            self.light.radius > 0.0 || self.light.vector != alg::Vec3::zero()
        );

        self.manager.set(entity, self.light);
    }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            instances: std::collections::HashMap::with_capacity(hint),
        }
    }

    pub fn build(&mut self) -> LightBuilder {
        LightBuilder::new(self)
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
