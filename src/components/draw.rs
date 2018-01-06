use std;
use alg;
use ecs;
use render;

use components::transform;

pub struct Draws {
    instances: render::Instances,
    handles: std::collections::HashMap<
        ecs::EntityHandle,
        render::InstanceHandle,
    >,
}

impl Draws {
    pub fn new(hint: usize, instances: render::Instances) -> Draws {
        Draws {
            handles: std::collections::HashMap::with_capacity(hint),
            instances: instances,
        }
    }

    pub fn add(&mut self, entity: ecs::EntityHandle, model_index: usize) {
        let handle = self.instances.add(
            render::InstanceUBO::default(),
            model_index,
        );

        self.handles.insert(
            entity,
            handle,
        );
    }

    // Update
    pub fn transfer(&mut self, transforms: transform::Transforms) {
        for (entity, instance) in &self.handles {
            // Get transform component data
            let transform = transforms.get(*entity);

            // Build uniform buffer object
            let ubo = {
                let translation = alg::Mat::translation_vec(transform.0);
                let rotation = transform.1;
                let scale = alg::Mat::scale_vec(transform.2);

                let model = translation * rotation * scale;
                render::InstanceUBO::new(model)
            };

            // Update renderer
            self.instances.update(*instance, ubo);
        }
    }
}
