use alg;
use entity;
use components;
use render;

use components::transform;
use components::Component;

pub const DEFAULT_FOV: f32 = 60.0;
pub const DEFAULT_NEAR: f32 = 0.01;
pub const DEFAULT_FAR: f32 = 4.0;

#[derive(Copy, Clone)]
pub struct Camera {
    fov: f32,
    near: f32,
    far: f32,
    overrule: Option<render::SharedUBO>,
}

impl Default for Camera {
    fn default() -> Camera {
        Camera {
            fov: DEFAULT_FOV,
            near: DEFAULT_NEAR,
            far: DEFAULT_FAR,
            overrule: None,
        }
    }
}

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
                Camera::default(),
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
