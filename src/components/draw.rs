use std;
use alg;
use entity;
use render;
use components;

use components::transform;
use components::softbody;
use components::light;

macro_rules! get_handle {
    ($self: ident, $entity: expr) => {{
        debug_validate_entity!($self, $entity);
        let handle = $self.handles[&$entity];

        #[cfg(debug_assertions)] {
            if handle.is_none() {
                panic!(
                    "Draw instance handle for entity {} is None. \
                    You probably attempted to use the Draw component \
                    before binding a model to it.",
                    $entity,
                );
            }
        }

        handle.unwrap()
    }}
}

pub struct Manager {
    handles: std::collections::HashMap<
        entity::Handle,
        Option<render::InstanceHandle>,
    >,
    pub instances: render::Instances,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        self.handles.insert(entity, None);
    }

    fn registered(&self, entity: entity::Handle) -> bool {
        self.handles.contains_key(&entity)
    }

    fn count(&self) -> usize {
        self.handles.len()
    }

    #[cfg(debug_assertions)] fn debug_name(&self) -> &str { "Draw" }
}

impl Manager {
    pub fn new(hint: usize, instances: render::Instances) -> Manager {
        Manager {
            instances,
            handles: std::collections::HashMap::with_capacity(hint),
        }
    }

    /// Set model that the draw component will render for this entity.
    /// For now, this can only be done once.
    pub fn bind_model(&mut self, entity: entity::Handle, model_index: usize) {
        debug_validate_entity!(self, entity);
        debug_assert!(self.handles[&entity].is_none());

        let handle = self.instances.add(
            render::InstanceUBO::default(),
            model_index,
        );

        *self.handles.get_mut(&entity).unwrap() = Some(handle);
    }

    /// Stop entity from being rendered
    pub fn hide(&mut self, entity: entity::Handle) {
        let handle = get_handle!(self, entity);
        self.instances.update_meta(
            handle,
            render::InstanceMeta::new(true),
        );
    }

    /// Resume rendering of entity (idempotent)
    pub fn unhide(&mut self, entity: entity::Handle) {
        let handle = get_handle!(self, entity);
        self.instances.update_meta(
            handle,
            render::InstanceMeta::new(false),
        );
    }

    // Update
    pub fn transfer(
        &mut self,
        transforms: &transform::Manager,
        softbodies: &softbody::Manager,
        lights: &light::Manager,
    ) {
        for (entity, instance) in &self.handles {
            debug_assert!(instance.is_some());

            // Get transform component data
            debug_validate_entity!(transforms, *entity);
            let transform = transforms.get(*entity);

            // Build uniform buffer object
            let ubo = {
                let model = alg::Mat4::transform(
                    transform.0,
                    transform.1,
                    transform.2,
                );

                let instance_lights = lights.cull(transform.0);

                render::InstanceUBO::new(
                    model,
                    instance_lights,
                    softbodies.get_position_offsets(*entity),
                    softbodies.get_normal_offsets(*entity),
                )
            };

            // Update renderer
            self.instances.update(instance.unwrap(), ubo);
        }
    }
}
