use alg;
use entity;
use components;
use render;

use components::transform;
use components::Component;

pub const DEFAULT_FOV: f32 = 60.0;
pub const DEFAULT_NEAR: f32 = 0.01;
pub const DEFAULT_FAR: f32 = 32.0;

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

    /// Set the main camera that will be rendered
    pub fn set_active(&mut self, camera_index: usize) {
        debug_assert!(camera_index < self.count());
        self.active = camera_index;
    }

    pub fn set_fov(&mut self, entity: entity::Handle, fov: f32) {
        self.instances.iter_mut()
            .find(|instance| instance.0 == entity)
            .expect(&format!("Entity {} not found in light manager", entity))
            .1.fov = fov;
    }

    pub fn set_near(&mut self, entity: entity::Handle, near: f32) {
        self.instances.iter_mut()
            .find(|instance| instance.0 == entity)
            .expect(&format!("Entity {} not found in light manager", entity))
            .1.near = near;
    }

    pub fn set_far(&mut self, entity: entity::Handle, far: f32) {
        self.instances.iter_mut()
            .find(|instance| instance.0 == entity)
            .expect(&format!("Entity {} not found in light manager", entity))
            .1.far = far;
    }

    /// Override a camera with a custom shared UBO
    pub fn overrule(
        &mut self,
        entity: entity::Handle,
        shared_ubo: render::SharedUBO,
    ) {
        self.instances.iter_mut()
            .find(|instance| instance.0 == entity)
            .expect(&format!("Entity {} not found in light manager", entity))
            .1.overrule = Some(shared_ubo);
    }

    /// Build a SharedUBO necessary for rendering from the active camera
    pub fn compute(
        &mut self,
        transforms: &transform::Manager,
        screen: ::ScreenData,
    ) -> render::SharedUBO {
        debug_assert!(self.active < self.count());

        // Get active entity and camera
        let (entity, camera) = self.instances[self.active];

        // Return overridden shared UBO if set
        if let Some(shared_ubo) = camera.overrule { return shared_ubo }

        // Get transform data for active camera entity
        let (position, orientation, _) = transforms.get(entity);

        /* Build view and projection matrices */

        let view = orientation.conjugate().to_mat()
            * alg::Mat4::translation_vec(-position);

        let projection = alg::Mat4::perspective(
            camera.fov,
            screen.width as f32 / screen.height as f32,
            camera.near,
            camera.far,
        );

        render::SharedUBO::new(view, projection)
    }
}
