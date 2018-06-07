use alg;
use entity;
use components;

pub struct Transform {
          position: alg::Vec3,
    local_position: alg::Vec3,
          orientation: alg::Quat,
    local_orientation: alg::Quat,
          scale: alg::Vec3,
    local_scale: alg::Vec3,

    parent: Option<usize>,
    children: Vec<usize>,
    cached_transform: alg::Mat4,
}

impl Transform {
    fn blank(child_hint: usize) -> Transform {
        Transform {
                  position: alg::Vec3::zero(),
            local_position: alg::Vec3::zero(),
                  orientation: alg::Quat::id(),
            local_orientation: alg::Quat::id(),
                  scale: alg::Vec3::one(),
            local_scale: alg::Vec3::one(),

            parent: None,
            children: Vec::with_capacity(child_hint),
            cached_transform: alg::Mat4::id(),
        }
    }

    /// Set/update transform with respect to parent. \
    /// Note that the mutable reference to self could only have been obtained
    /// unsafely.
    fn update_cached(&mut self, manager: &Manager) {
        // This method is only called internally,
        // so we can assume this will succeed
        debug_assert!(self.parent.is_some());
        let parent = manager.instances[self.parent.unwrap()].as_ref().unwrap();

        // Rebuild cached transform for this instance
        let transform =
            parent.cached_transform
            * alg::Mat4::transform(
                self.local_position,
                self.local_orientation,
                self.local_scale,
            );

        /* Assign transform data */

        let scale = transform.to_scale();
        self.scale = scale;

        self.orientation = transform.to_rotation_raw(scale).to_quat();
        self.position = transform * alg::Vec3::zero();
        self.cached_transform = transform;
    }

    /// Recursively call `update_cached()` on all children
    unsafe fn update_children(&self, manager: &mut Manager) {
        for child_index in &self.children {
            let ptr = manager.instances.as_mut_ptr()
                .offset(*child_index as isize);
            let child = (*ptr).as_mut().unwrap();

            child.update_cached(manager);
            child.update_children(manager);
        }
    }
}

// Data layout assumes that almost all entities will have this component
pub struct Manager {
    instances: Vec<Option<Transform>>,
    count: usize,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        let i = entity.get_index() as usize;

        // Resize array to fit new entity
        loop {
            if i >= self.instances.len() {
                self.instances.push(None);
                continue;
            }

            break;
        }

        self.instances[i] = Some(Transform::blank(0));
        self.count += 1;
    }

    fn count(&self) -> usize {
        self.count
    }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            instances: Vec::with_capacity(hint),
            count: 0,
        }
    }

    /// Set transform parent of `entity` to `parent`
    pub fn parent(&mut self, entity: entity::Handle, parent: entity::Handle) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.instances.len());
        debug_assert!(self.instances[i].is_some());

        let transform = unsafe {
            let ptr = self.instances.as_mut_ptr().offset(i as isize);
            (*ptr).as_mut().unwrap()
        };

        let j = parent.get_index() as usize;
        debug_assert!(j < self.instances.len());
        debug_assert!(self.instances[j].is_some());

        let parent = unsafe {
            let ptr = self.instances.as_mut_ptr().offset(j as isize);
            (*ptr).as_mut().unwrap()
        };

        transform.parent = Some(j);
        parent.children.push(i);

        // TODO: Potentially update local transform relative to the new parent

        transform.update_cached(self);
        unsafe { transform.update_children(self); }
    }

    /// Set transform data \
    /// Faster than setting the fields individually
    pub fn set(
        &mut self,
        entity: entity::Handle,
        position: alg::Vec3,
        orientation: alg::Quat,
        scale: alg::Vec3,
    ) {
        let i = entity.get_index() as usize;
        self.set_raw(i, position, orientation, scale);
    }

    /// Returns tuple of position, rotation, scale \
    /// Faster than getting the transform fields individually
    pub fn get(&self, entity: entity::Handle) -> (
        alg::Vec3,
        alg::Quat,
        alg::Vec3,
    ) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.instances.len());
        debug_assert!(self.instances[i].is_some());

        let transform = self.instances[i].as_ref().unwrap();

        (
            transform.position,
            transform.orientation,
            transform.scale,
        )
    }

    pub fn get_position(&self, entity: entity::Handle) -> alg::Vec3 {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.instances.len());
        debug_assert!(self.instances[i].is_some());

        self.instances[i].as_ref().unwrap()
            .position
    }

    pub fn get_orientation(&self, entity: entity::Handle) -> alg::Quat {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.instances.len());
        debug_assert!(self.instances[i].is_some());

        self.instances[i].as_ref().unwrap()
            .orientation
    }

    pub fn get_scale(&self, entity: entity::Handle) -> alg::Vec3 {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.instances.len());
        debug_assert!(self.instances[i].is_some());

        self.instances[i].as_ref().unwrap()
            .scale
    }

    pub fn set_position(
        &mut self,
        entity: entity::Handle,
        position: alg::Vec3,
    ) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.instances.len());
        debug_assert!(self.instances[i].is_some());

        self.instances[i].as_mut().unwrap()
            .position = position;
    }

    pub fn set_orientation(
        &mut self,
        entity: entity::Handle,
        orientation: alg::Quat,
    ) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.instances.len());
        debug_assert!(self.instances[i].is_some());

        self.instances[i].as_mut().unwrap()
            .orientation = orientation;
    }

    pub fn set_scale(
        &mut self,
        entity: entity::Handle,
        scale: alg::Vec3,
    ) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.instances.len());
        debug_assert!(self.instances[i].is_some());

        self.instances[i].as_mut().unwrap()
            .scale = scale;
    }

    /* "Unsafe" methods for components with similar data layouts.
     * These technically invalidate the ECS model but are used
     * for performance purposes.
     */

    pub(super) fn get_local_position_raw(&self, index: usize) -> alg::Vec3 {
        self.instances[index].as_ref().unwrap().local_position
    }

    pub(super) fn get_local_orientation_raw(&self, index: usize) -> alg::Quat {
        self.instances[index].as_ref().unwrap().local_orientation
    }

    /// Set transform data and update chain
    pub(super) fn set_raw(
        &mut self,
        index: usize,
        position: alg::Vec3,
        orientation: alg::Quat,
        scale: alg::Vec3,
    ) {
        debug_assert!(index < self.instances.len());
        debug_assert!(self.instances[index].is_some());

        let transform = unsafe {
            let ptr = self.instances.as_mut_ptr().offset(index as isize);
            (*ptr).as_mut().unwrap()
        };

        transform.local_position = position;
        transform.local_orientation = orientation;
        transform.local_scale = scale;

        /* Set worldspace transform data */

        // If this transform has a parent, update in chain
        if transform.parent.is_some() {
            transform.update_cached(self);
        }

        // No parent (chain root)--just set data
        else {
            transform.position = position;
            transform.orientation = orientation;
            transform.scale = scale;

            transform.cached_transform = alg::Mat4::transform(
                position,
                orientation,
                scale,
            );
        }

        // Update children transforms
        unsafe { transform.update_children(self); }
    }
}
