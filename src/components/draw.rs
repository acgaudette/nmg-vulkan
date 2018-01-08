use std;
use alg;
use entity;
use render;

use components::transform;

pub struct Manager {
    pub instances: render::Instances,
    handles: std::collections::HashMap<
        entity::Handle,
        render::InstanceHandle,
    >,
}

impl Manager {
    pub fn new(hint: usize, instances: render::Instances) -> Manager {
        Manager {
            handles: std::collections::HashMap::with_capacity(hint),
            instances: instances,
        }
    }

    pub fn register(
        &mut self,
        entity: entity::Handle,
        model_index: usize,
    ) {
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
    pub fn transfer(&mut self, transforms: &transform::Manager) {
        for (entity, instance) in &self.handles {
            // Get transform component data
            let transform = transforms.get(*entity);

            // Build uniform buffer object
            let ubo = {
                let translation = alg::Mat::translation_vec(transform.0);
                let rotation = transform.1.to_mat();
                let scale = alg::Mat::scale_vec(transform.2);

                let model = translation * rotation * scale;
                let offsets = [alg::Vec3::zero(); render::MAX_SOFTBODY_VERT];

                render::InstanceUBO::new(model, offsets)
            };

            // Update renderer
            self.instances.update(*instance, ubo);
        }
    }

    pub fn count(&self) -> usize {
        self.handles.len()
    }
}
