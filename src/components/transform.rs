use alg;
use ecs;

// Data layout assumes that almost all entities will have this component
pub struct Transforms {
    positions: Vec<alg::Vec3>,
    rotations: Vec<alg::Mat>, // Stand-in type
    scales:    Vec<alg::Vec3>,
}

impl Transforms {
    pub fn new(hint: usize) -> Transforms {
        Transforms {
            positions: Vec::with_capacity(hint),
            rotations: Vec::with_capacity(hint),
            scales:    Vec::with_capacity(hint),
        }
    }

    pub fn register(&mut self, entity: ecs::EntityHandle) {
        debug_assert!(self.positions.len() == self.rotations.len());
        debug_assert!(self.rotations.len() == self.scales.len());

        let i = entity.get_index() as usize;

        // Resize array to fit new entity
        loop {
            if i >= self.positions.len() {
                self.positions.push(alg::Vec3::zero());
                self.rotations.push(alg::Mat::identity());
                self.scales.push(alg::Vec3::one());
            }
        }
    }

    pub fn set(
        &mut self,
        entity:   ecs::EntityHandle,
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

    pub fn get(&self, entity: ecs::EntityHandle) -> (
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
