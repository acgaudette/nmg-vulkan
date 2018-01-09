use alg;
use entity;
use render;
use components;

use ::FIXED_DT; // Import from lib
use components::transform;

const ITERATIONS: usize = 1;
const BOUNCE: f32 = 0.1; // "Realistic" = 2

struct Particle {
    position: alg::Vec3,
    last: alg::Vec3,
}

impl Particle {
    fn new(position: alg::Vec3) -> Particle {
        Particle {
            position: position,
            last: position,
        }
    }
}

struct Rod {
    left: usize,
    right: usize,
    length: f32,
}

impl Rod {
    fn new(left: usize, right: usize, particles: &[Particle]) -> Rod {
        debug_assert!(left < particles.len());
        debug_assert!(right < particles.len());

        let length = alg::Vec3::dist(
            particles[left].position,
            particles[right].position,
        );

        Rod {
            left,
            right,
            length,
        }
    }
}

#[repr(C)]
struct Instance {
    particles: Vec<Particle>,
    rods: Vec<Rod>,
    mass: f32,
    force: alg::Vec3,
    accel_dt: alg::Vec3,
    center: alg::Vec3,
    model: Vec<alg::Vec3>,
}

impl Instance {
    fn new(
        mass: f32,
        points: &[alg::Vec3],
        bindings: &[(usize, usize)],
    ) -> Instance {
        debug_assert!(mass > 0.);

        let mut particles = Vec::with_capacity(points.len());
        let mut model = Vec::with_capacity(points.len());
        let mut rods = Vec::with_capacity(bindings.len());

        for point in points {
            particles.push(Particle::new(*point));
            model.push(*point);
        }

        for binding in bindings {
            rods.push(Rod::new(binding.0, binding.1, &particles));
        }

        Instance {
            particles: particles,
            rods: rods,
            mass: mass,
            force: alg::Vec3::zero(),
            accel_dt: alg::Vec3::zero(),
            center: alg::Vec3::zero(),
            model,
        }
    }

    fn offset(&self, index: usize) -> alg::Vec3 {
        self.particles[index].position - self.center - self.model[index]
    }

    #[inline]
    fn set_force(&mut self, force: alg::Vec3) {
        self.force = force;
        self.accel_dt = (force / self.mass) * FIXED_DT * FIXED_DT;
    }
}

// Data layout assumes many physics objects (but may still be sparse)
pub struct Manager {
    instances: Vec<Option<Instance>>,
    planes: Vec<alg::Plane>,
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
    }

    // TODO: This currently only returns the length of the underlying data
    // structure, not the count of the registered entities
    fn count(&self) -> usize {
        self.instances.len()
    }
}

impl Manager {
    pub fn new(instance_hint: usize, plane_hint: usize) -> Manager {
        Manager {
            instances: Vec::with_capacity(instance_hint),
            planes: Vec::with_capacity(plane_hint),
        }
    }

    pub fn init_instance(
        &mut self,
        entity: entity::Handle,
        mass: f32,
        points: &[alg::Vec3],
        bindings: &[(usize, usize)],
    ) {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.instances.len());

        self.instances[i] = Some(Instance::new(mass, points, bindings));
    }

    pub fn set(
        &mut self,
        entity: entity::Handle,
        force: alg::Vec3,
    ) {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.instances.len());

        if let Some(ref mut instance) = self.instances[i] {
            instance.set_force(force);
        }
    }

    pub fn get_offsets(
        &self,
        entity: entity::Handle,
    ) -> [render::PaddedVec3; render::MAX_SOFTBODY_VERT] {
        let i = entity.get_index() as usize;

        // Default to no offsets (identity)
        let mut offsets = [
            render::PaddedVec3::default();
            render::MAX_SOFTBODY_VERT
        ];

        // Space has not been allocated for this component (does not exist)
        if i >= self.instances.len() {
            return offsets;
        }

        // If the entity has a softbody component, fill the offsets array
        if let Some(ref instance) = self.instances[i] {
            for i in 0..instance.particles.len() {
                offsets[i] = render::PaddedVec3::new(instance.offset(i));
            }
        }

        offsets
    }

    pub fn add_plane(&mut self, plane: alg::Plane) {
        self.planes.push(plane);
    }

    pub fn simulate(&mut self, transforms: &mut transform::Manager) {
        // Position Verlet
        for i in 0..self.instances.len() {
            let mut instance = match self.instances[i] {
                Some(ref mut instance) => instance,
                None => continue,
            };

            // Update particles
            for particle in &mut instance.particles {
                let velocity = particle.position - particle.last;
                particle.last = particle.position;

                particle.position = particle.position + velocity
                    + instance.accel_dt;
            }

            // Constraints
            for _ in 0..ITERATIONS {
                // Rods
                for rod in &instance.rods {
                    let left = instance.particles[rod.left].position;
                    let right = instance.particles[rod.right].position;
                    let difference = right - left;

                    let distance = difference.mag();
                    let percent = 0.5 * (rod.length - distance) / distance;
                    let offset = difference * percent;

                    instance.particles[rod.left].position = left - offset;
                    instance.particles[rod.right].position = right + offset;
                }

                // Planes
                for plane in &self.planes {
                    for particle in &mut instance.particles {
                        let distance = plane.normal.dot(particle.position)
                            + plane.offset;

                        if distance > 0. {
                            continue;
                        }

                        particle.position = particle.position
                            - plane.normal * BOUNCE * distance;
                    }
                }
            }

            // Compute average position
            let average = {
                let mut sum = alg::Vec3::zero();

                for particle in &instance.particles {
                    sum = sum + particle.position;
                }

                sum / instance.particles.len() as f32
            };

            // Update instance position
            instance.center = average;

            transforms.set_position_i(i, average);
        }
    }
}
