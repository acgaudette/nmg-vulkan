use alg;
use entity;
use components;

const ITERATIONS: usize = 10;

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
    center: alg::Vec3,
    model: Vec<alg::Vec3>,
}

impl Instance {
    fn new(
        mass: f32,
        points: &[alg::Vec3],
        bindings: &[(usize, usize)],
    ) -> Instance {
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

        let force = alg::Vec3::zero();
        let center = alg::Vec3::zero();

        Instance {
            particles,
            rods,
            mass,
            force,
            center,
            model,
        }
    }
}

// Data layout assumes many physics objects (but may still be sparse)
pub struct Manager {
    instances: Vec<Option<Instance>>,
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
    pub fn new(hint: usize) -> Manager {
        Manager {
            instances: Vec::with_capacity(hint),
        }
    }

    pub fn init_instance(
        &mut self,
        entity: entity::Handle,
        mass: f32,
        points: Vec<alg::Vec3>,
        bindings: Vec<(usize, usize)>,
    ) {
        let i = entity.get_index() as usize;
        debug_assert!(i < self.instances.len());

        self.instances[i] = Some(Instance::new(mass, points, bindings));
    }

    pub fn simulate(
        &mut self,
        delta: f64,
        transforms: &mut components::transform::Manager,
    ) {
        // Position Verlet
        let mut itr = self.instances.iter_mut();
        while let Some(&mut Some(ref mut instance)) = itr.next() {
            // Update particles
            for particle in instance.particles.iter_mut() {
                let velocity = particle.position - particle.last;
                particle.last = particle.position;

                particle.position = particle.position + velocity;
            }

            for _ in 0..ITERATIONS {
                // Update rods
                for rod in &instance.rods {
                    let left = instance.particles[rod.left].position;
                    let right = instance.particles[rod.right].position;
                    let offset = right - left;

                    let distance = offset.mag();
                    let percent = 0.5 * (rod.length - distance) / distance;
                    let offset = offset * percent;

                    instance.particles[rod.left].position = left - offset;
                    instance.particles[rod.right].position = right + offset;
                }

                /* Do constraints here */
            }
        }
    }
}
