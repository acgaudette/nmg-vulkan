use alg;
use entity;
use components;

use ::FIXED_DT; // Import from lib
use components::transform;

// Data layout assumes many physics objects (but may still be sparse)
#[repr(C)]
pub struct Manager {
    forces: Vec<alg::Vec3>,
    masses: Vec<f32>,
    drags:  Vec<f32>,
    lin_velocities: Vec<alg::Vec3>,
    torques: Vec<alg::Vec3>,
    ang_velocities: Vec<alg::Vec3>,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        debug_assert!(self.forces.len() == self.masses.len());
        debug_assert!(self.masses.len() == self.drags.len());
        debug_assert!(self.drags.len() == self.lin_velocities.len());
        debug_assert!(self.lin_velocities.len() == self.torques.len());
        debug_assert!(self.torques.len() == self.ang_velocities.len());

        let i = entity.get_index() as usize;

        // Resize array to fit new entity
        loop {
            if i >= self.forces.len() {
                self.forces.push(alg::Vec3::zero());
                self.masses.push(0.);
                self.drags.push(0.);
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
            drags:  Vec::with_capacity(hint),
            lin_velocities: Vec::with_capacity(hint),
            torques: Vec::with_capacity(hint),
            ang_velocities: Vec::with_capacity(hint),
        }
    }

    pub fn set(
        &mut self,
        entity: entity::Handle,
        mass:   f32,
        drag:   f32,
        force:  alg::Vec3,
        torque: alg::Vec3,
    ) {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.masses.len());

        self.forces[i] = force;
        self.masses[i] = mass;
        self.drags[i] = drag;
        self.torques[i] = torque;
    }

    pub fn simulate(&mut self, transforms: &mut transform::Manager) {
        debug_assert!(self.forces.len() == self.masses.len());
        debug_assert!(self.masses.len() == self.drags.len());
        debug_assert!(self.drags.len() == self.lin_velocities.len());
        debug_assert!(self.lin_velocities.len() == self.torques.len());
        debug_assert!(self.torques.len() == self.ang_velocities.len());

        // Semi-implicit Euler
        for i in 0..self.forces.len() {
            /* Linear motion */

            // Simple drag
            let lin_resistance = self.lin_velocities[i] * self.drags[i];

            let lin_momentum = (self.forces[i] - lin_resistance)
                * FIXED_DT as f32;

            assert!(self.masses[i] > 0.);

            self.lin_velocities[i] = self.lin_velocities[i]
                + lin_momentum / self.masses[i];

            let position = transforms.get_position_i(i)
                + self.lin_velocities[i] * FIXED_DT as f32;

            transforms.set_position_i(i, position);

            /* Angular motion */

            // Simple drag
            let ang_resistance = self.ang_velocities[i] * self.drags[i];

            let ang_momentum = (self.torques[i] - ang_resistance)
                * FIXED_DT as f32;

            // TODO: Tensor support
            let inverse_inertia = 6. / self.masses[i];

            self.ang_velocities[i] = self.ang_velocities[i]
                + ang_momentum * inverse_inertia;

            // 4D derivative vector
            let derivative = alg::Quat::new(
                self.ang_velocities[i].x,
                self.ang_velocities[i].y,
                self.ang_velocities[i].z,
                0.,
            );

            let last = transforms.get_orientation_i(i).norm(); // Renormalize
            let orientation = last + last * 0.5 * derivative * FIXED_DT as f32;

            transforms.set_orientation_i(i, orientation);
        }
    }
}
