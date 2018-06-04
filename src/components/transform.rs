use alg;
use entity;
use components;

// Data layout assumes that almost all entities will have this component
pub struct Manager {
    positions: Vec<alg::Vec3>,
    orientations: Vec<alg::Quat>,
    scales: Vec<alg::Vec3>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        debug_assert!(self.positions.len() == self.orientations.len());
        debug_assert!(self.orientations.len() == self.scales.len());

        let i = entity.get_index() as usize;

        // Resize array to fit new entity
        loop {
            if i >= self.positions.len() {
                self.positions.push(alg::Vec3::zero());
                self.orientations.push(alg::Quat::id());
                self.scales.push(alg::Vec3::one());

                continue;
            }

            break;
        }
    }

    // TODO: This currently only returns the length of the underlying data
    // structure, not the count of the registered entities
    fn count(&self) -> usize {
        self.positions.len()
    }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            positions: Vec::with_capacity(hint),
            orientations: Vec::with_capacity(hint),
            scales: Vec::with_capacity(hint),
        }
    }

    pub fn set(
        &mut self,
        entity: entity::Handle,
        position: alg::Vec3,
        orientation: alg::Quat,
        scale: alg::Vec3,
    ) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.positions.len());

        self.positions[i] = position;
        self.orientations[i] = orientation;
        self.scales[i] = scale;
    }

    pub fn get(&self, entity: entity::Handle) -> (
        alg::Vec3,
        alg::Quat,
        alg::Vec3,
    ) {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.positions.len());

        (
            self.positions[i],
            self.orientations[i],
            self.scales[i],
        )
    }

    pub fn get_position(&self, entity: entity::Handle) -> alg::Vec3 {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.positions.len());

        self.positions[i]
    }

    pub fn get_orientation(&self, entity: entity::Handle) -> alg::Quat {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.positions.len());

        self.orientations[i]
    }

    pub fn get_scale(&self, entity: entity::Handle) -> alg::Vec3 {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.positions.len());

        self.scales[i]
    }

    /* "Unsafe" methods for components with similar data layouts.
     * These technically invalidate the ECS model but are used
     * for performance purposes.
     */

    pub(super) fn get_position_raw(&self, index: usize) -> alg::Vec3 {
        self.positions[index]
    }

    pub(super) fn get_orientation_raw(&self, index: usize) -> alg::Quat {
        self.orientations[index]
    }

    pub(super) fn set_position_raw(
        &mut self,
        index: usize,
        value: alg::Vec3,
    ) {
        self.positions[index] = value;
    }

    pub(super) fn set_orientation_raw(
        &mut self,
        index: usize,
        value: alg::Quat,
    ) {
        self.orientations[index] = value;
    }
}
