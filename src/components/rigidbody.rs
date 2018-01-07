use alg;
use entity;
use components;

// Data layout assumes many physics objects (but may still be sparse)
#[repr(C)]
pub struct Manager {
    forces: Vec<alg::Vec3>,
    masses: Vec<f32>,
    lin_velocities: Vec<alg::Vec3>,
    torques: Vec<alg::Vec3>,
    ang_velocities: Vec<alg::Vec3>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        debug_assert!(self.forces.len() == self.masses.len());
        debug_assert!(self.masses.len() == self.lin_velocities.len());
        debug_assert!(self.lin_velocities.len() == self.torques.len());
        debug_assert!(self.torques.len() == self.ang_velocities.len());

        let i = entity.get_index() as usize;

        // Resize array to fit new entity
        loop {
            if i >= self.forces.len() {
                self.forces.push(alg::Vec3::zero());
                self.masses.push(0.);
                self.lin_velocities.push(alg::Vec3::zero());
                self.torques.push(alg::Vec3::zero());
                self.ang_velocities.push(alg::Vec3::zero());

                continue;
            }

            break;
        }
    }

    // TODO: This currently only returns the length of the underlying data
    // structure, not the count of the registered entities
    fn count(&self) -> usize {
        self.forces.len()
    }
}

impl Manager {
    pub fn new(hint: usize) -> Manager {
        Manager {
            forces: Vec::with_capacity(hint),
            masses: Vec::with_capacity(hint),
            lin_velocities: Vec::with_capacity(hint),
            torques: Vec::with_capacity(hint),
            ang_velocities: Vec::with_capacity(hint),
        }
    }

    pub fn set(
        &mut self,
        entity: entity::Handle,
        mass:   f32,
        force:  alg::Vec3,
        torque: alg::Vec3,
    ) {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.masses.len());

        self.forces[i] = force;
        self.masses[i] = mass;
        self.torques[i] = torque;
    }

    pub fn simulate(
        &mut self,
        delta: f64,
        transforms: &mut components::transform::Manager,
    ) {
        debug_assert!(self.forces.len() == self.masses.len());
        debug_assert!(self.masses.len() == self.lin_velocities.len());
        debug_assert!(self.lin_velocities.len() == self.torques.len());
        debug_assert!(self.torques.len() == self.ang_velocities.len());

        // Semi-implicit Euler
        for i in 0..self.forces.len() {
            /* Linear motion */

            let lin_momentum = self.forces[i] * delta as f32;

            assert!(self.masses[i] > 0.);

            self.lin_velocities[i] = self.lin_velocities[i]
                + lin_momentum / self.masses[i];

            let position = transforms.get_position_i(i)
                + self.lin_velocities[i] * delta as f32;

            transforms.set_position_i(i, position);

            /* Angular motion */

            let ang_momentum = self.torques[i] * delta as f32; // do add here ?

            // TODO: Tensor support
            let inverse_inertia = 6. / self.masses[i];

            self.ang_velocities[i] = self.ang_velocities[i]
                + ang_momentum * inverse_inertia;

            // Slow!
            let ang_quat = alg::Quat::angle_axis_raw(
                self.ang_velocities[i].norm(),
                self.ang_velocities[i].mag(),
            );

            let last = transforms.get_orientation_i(i).norm(); // Renormalize
            let orientation = ang_quat * 0.5 * last; // Integrate

            transforms.set_orientation_i(i, orientation);
        }
    }
}
