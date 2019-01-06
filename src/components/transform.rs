use alg;
use entity;
use components;

macro_rules! get_instance {
    ($self: ident, $entity: expr) => {{
        debug_validate_entity!($self, $entity);
        let i = $entity.get_index() as usize;
        $self.instances[i].as_ref().unwrap()
    }}
}

#[allow(unused_macros)]
macro_rules! get_mut_instance {
    ($self: ident, $entity: expr) => {{
        debug_validate_entity!($self, $entity);
        let i = $entity.get_index() as usize;
        $self.instances[i].as_mut().unwrap()
    }}
}

macro_rules! get_mut_instance_raw {
    ($self: expr, $index: expr) => {
        unsafe {
            debug_assert!($index < $self.instances.len());
            debug_assert!($self.instances[$index].is_some());

            let ptr = $self.instances.as_mut_ptr().offset($index as isize);
            (*ptr).as_mut().unwrap()
        }
    }
}

pub struct Transform {
          position: alg::Vec3,
    local_position: alg::Vec3,
          orientation: alg::Quat,
    local_orientation: alg::Quat,
          scale: alg::Vec3,
    local_scale: alg::Vec3,

    parent: Option<usize>,
    children: Vec<usize>,
    cached_transform: alg::Mat4, // Cached world transform
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
        // The method is only called internally,
        // so we can assume this will succeed
        debug_assert!(self.parent.is_some());
        let parent = manager.instances[self.parent.unwrap()].as_ref().unwrap();

        // Check for non-uniform scale at runtime
        #[cfg(debug_assertions)] {
            let parent_scale = parent.cached_transform.to_scale();
            if !parent_scale.is_uniform() {
                eprintln!(
                    "Warning: Non-uniform scale is not supported \
                    in transform hierarchy"
                );
            }
        }

        // Rebuild cached transform for this instance
        let transform = parent.cached_transform
            * alg::Mat4::transform(
                self.local_position,
                self.local_orientation,
                self.local_scale,
            );

        /* Assign transform data */

        let scale = transform.to_scale();

        self.scale = scale;
        self.orientation = transform.to_rotation_raw(scale).to_quat();
        self.position = transform.to_position();
        self.cached_transform = transform;
    }

    /// Recursively call `update_cached()` on all children
    unsafe fn update_children(&self, manager: &mut Manager) {
        for child_index in &self.children {
            #[allow(unused_unsafe)]
            let child = get_mut_instance_raw!(manager, *child_index);

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

    fn registered(&self, entity: entity::Handle) -> bool {
        let i = entity.get_index() as usize;
        i < self.instances.len() && self.instances[i].is_some()
    }

    fn count(&self) -> usize {
        self.count
    }

    #[cfg(debug_assertions)] fn debug_name(&self) -> &str { "Transform" }
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
        debug_validate_entity!(self, entity); // Child
        debug_validate_entity!(self, parent);

        #[cfg(debug_assertions)] {
            if entity == parent {
                panic!("Attempted to parent entity {} to itself", entity);
            }
        }

        let transform_index = entity.get_index() as usize;
        let transform = get_mut_instance_raw!(self, transform_index);

        let parent_index = parent.get_index() as usize;
        let parent_transform = get_mut_instance_raw!(self, parent_index);

        #[cfg(debug_assertions)] {
            if transform.children.contains(&parent_index) {
                panic!(
                    "Attemped to parent entity {} to its child {}",
                    entity,
                    parent,
                );
            }
        }

        transform.parent = Some(parent_index);

        if !parent_transform.children.contains(&transform_index) {
            parent_transform.children.push(transform_index);
        }

        // TODO: Potentially update local transform
        // relative to the new parent at the time of assignment

        transform.update_cached(self);
        unsafe { transform.update_children(self); }
    }

    /// Returns tuple of position, rotation, scale \
    /// Faster than getting the transform fields individually
    pub fn get(&self, entity: entity::Handle) -> (
        alg::Vec3,
        alg::Quat,
        alg::Vec3,
    ) {
        let transform = get_instance!(self, entity);

        (
            transform.position,
            transform.orientation,
            transform.scale,
        )
    }

    /// Returns transform data as alg::Mat4
    pub fn get_mat(&self, entity: entity::Handle) -> alg::Mat4 {
        let transform = get_instance!(self, entity);

        alg::Mat4::transform(
            transform.position,
            transform.orientation,
            transform.scale,
        )
    }

    pub fn get_position(&self, entity: entity::Handle) -> alg::Vec3 {
        let transform = get_instance!(self, entity);
        transform.position
    }

    pub fn get_orientation(&self, entity: entity::Handle) -> alg::Quat {
        let transform = get_instance!(self, entity);
        transform.orientation
    }

    pub fn get_scale(&self, entity: entity::Handle) -> alg::Vec3 {
        let transform = get_instance!(self, entity);
        transform.scale
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
        debug_validate_entity!(self, entity);
        let i = entity.get_index() as usize;
        self.set_raw(i, position, orientation, scale);
    }

    /// Set transform position
    pub fn set_position(
        &mut self,
        entity: entity::Handle,
        position: alg::Vec3,
    ) {
        debug_validate_entity!(self, entity);
        let i = entity.get_index() as usize;
        let transform = get_mut_instance_raw!(self, i);
        transform.local_position = position;

        /* Set worldspace transform data */

        // If this transform has a parent, update in chain
        if transform.parent.is_some() {
            transform.update_cached(self);
        }

        // No parent (chain root)--just set data
        else {
            transform.position = position;
            transform.cached_transform.set_translation(position);
        }

        // Update children transforms
        unsafe { transform.update_children(self); }
    }

    /// Set transform orientation
    pub fn set_orientation(
        &mut self,
        entity: entity::Handle,
        orientation: alg::Quat,
    ) {
        debug_validate_entity!(self, entity);
        let i = entity.get_index() as usize;
        let transform = get_mut_instance_raw!(self, i);
        transform.local_orientation = orientation;

        /* Set worldspace transform data */

        // If this transform has a parent, update in chain
        if transform.parent.is_some() {
            transform.update_cached(self);
        }

        // No parent (chain root)--just set data
        else {
            transform.orientation = orientation;
            transform.cached_transform = alg::Mat4::transform(
                transform.position,
                transform.orientation,
                transform.scale,
            );
        }

        // Update children transforms
        unsafe { transform.update_children(self); }
    }

    /// Set transform scale
    pub fn set_scale(
        &mut self,
        entity: entity::Handle,
        scale: alg::Vec3,
    ) {
        debug_validate_entity!(self, entity);
        let i = entity.get_index() as usize;
        let transform = get_mut_instance_raw!(self, i);

        // Check for non-uniform scale at assignment-time
        #[cfg(debug_assertions)] {
            if !transform.children.is_empty() && !scale.is_uniform() {
                panic!(
                    "Non-uniform scale is not supported \
                    in transform hierarchy"
                );
            }
        }

        transform.local_scale = scale;

        /* Set worldspace transform data */

        // If this transform has a parent, update in chain
        if transform.parent.is_some() {
            transform.update_cached(self);
        }

        // No parent (chain root)--just set data
        else {
            transform.scale = scale;
            transform.cached_transform = alg::Mat4::transform(
                transform.position,
                transform.orientation,
                transform.scale,
            );
        }

        // Update children transforms
        unsafe { transform.update_children(self); }
    }

    /* "Unsafe" methods for components with similar data layouts.
     * These technically invalidate the ECS model but are used
     * for performance purposes.
     */

    /// Set transform data and update chain
    pub(super) fn set_raw(
        &mut self,
        index: usize,
        position: alg::Vec3,
        orientation: alg::Quat,
        scale: alg::Vec3,
    ) {
        let transform = get_mut_instance_raw!(self, index);

        // Check for non-uniform scale at assignment-time
        #[cfg(debug_assertions)] {
            if !transform.children.is_empty() && !scale.is_uniform() {
                panic!(
                    "Non-uniform scale is not supported \
                    in transform hierarchy"
                );
            }
        }

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
