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
}
