// Data layout assumes many physics objects (but may still be sparse)
pub struct Manager {
    velocities: Vec<alg::Vec3>,
    masses:     Vec<f32>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        debug_assert!(self.velocities.len() == self.masses.len());

        let i = entity.get_index() as usize;

        // Resize array to fit new entity
        loop {
            if i >= self.positions.len() {
                self.velocities.push(alg::Vec3::zero());
                self.masses.push(0.);

                continue;
            }

            break;
        }
    }

    // TODO: This currently only returns the length of the underlying data
    // structure, not the count of the registered entities
    fn count(&self) -> usize {
        self.velocities.len()
    }
}

impl Manager {
    fn new(hint: usize) -> Manager {
        Manager {
            velocities: Vec::with_capacity(hint),
            masses:     Vec::with_capacity(hint),
        }
    }
}
