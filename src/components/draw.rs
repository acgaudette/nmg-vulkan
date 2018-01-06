use std;
use ecs;
use render;

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
}
