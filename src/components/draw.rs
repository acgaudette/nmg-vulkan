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
}
