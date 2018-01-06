// Data layout assumes many physics objects (but may still be sparse)
pub struct Manager {
    velocities: Vec<alg::Vec3>,
    masses:     Vec<f32>,
    forces:     Vec<alg::Vec3>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        debug_assert!(self.velocities.len() == self.masses.len());
        debug_assert!(self.masses.len() == self.forces.len());

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
    pub fn new(hint: usize) -> Manager {
        Manager {
            velocities: Vec::with_capacity(hint),
            masses:     Vec::with_capacity(hint),
        }
    }

    pub fn set(&mut self, entity: entity::Handle, force: alg::Vec3) {
        let i = entity.get_index() as usize;

        debug_assert!(i < self.forces.len());

        self.forces[i] = force;
    }

    pub fn simulate(delta: f64, transforms: &transform::Manager) {
        debug_assert!(self.velocities.len() == self.masses.len());
        debug_assert!(self.masses.len() == self.forces.len());

        // Explicit Euler
        for i in 0..self.velocities.len() {
            let acceleration = self.forces[i] / self.masses[i];

            let position = {
                let mut pos = transforms.get_position_i(i);
                pos += self.velocities[i] * delta;
                pos
            };

            transforms.set_position_i(i, position);

            self.velocities[i] += acceleration * delta as f32;
        }
    }
}
