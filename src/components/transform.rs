use alg;
use entity;
use components;

// Data layout assumes that almost all entities will have this component
pub struct Manager {
    positions: Vec<alg::Vec3>,
    rotations: Vec<alg::Mat>, // Stand-in type
    scales:    Vec<alg::Vec3>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        debug_assert!(self.positions.len() == self.rotations.len());
        debug_assert!(self.rotations.len() == self.scales.len());

        let i = entity.get_index() as usize;

        // Resize array to fit new entity
        loop {
            if i >= self.positions.len() {
                self.positions.push(alg::Vec3::zero());
                self.rotations.push(alg::Mat::identity());
                self.scales.push(alg::Vec3::one());

                continue;
            }

            break;
        }
    }

    fn count(&self) -> usize {
        self.positions.len()
    }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            positions: Vec::with_capacity(hint),
            rotations: Vec::with_capacity(hint),
            scales:    Vec::with_capacity(hint),
        }
    }

    pub fn set(
        &mut self,
        entity:   entity::Handle,
        position: alg::Vec3,
        rotation: alg::Mat,
        scale:    alg::Vec3,
    ) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.positions.len());

        self.positions[i] = position;
        self.rotations[i] = rotation;
        self.scales[i] = scale;
    }

    pub fn get(&self, entity: entity::Handle) -> (
        alg::Vec3,
        alg::Mat,
        alg::Vec3,
    ) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.positions.len());

        (
            self.positions[i],
            self.rotations[i],
            self.scales[i],
        )
    }
}
