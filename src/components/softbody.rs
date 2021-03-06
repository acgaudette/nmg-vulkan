extern crate fnv;

use std;
use alg;
use entity;
use render;
use components;
use debug;

#[cfg(debug_assertions)] use graphics;

use components::transform;
const EXTRACT_ITER: usize = 16;

/// Default instance mass
pub const INST_DEFAULT_MASS: f32 = 1.0;

/// Default instance rigidity
pub const INST_DEFAULT_RIGID: f32 = 1.0;

/// Default system (softbody manager) bounce
pub const MNGR_DEFAULT_BOUNCE: f32 = 2.0;

/// Default system (softbody manager) friction
pub const MNGR_DEFAULT_FRICTION: f32 = 0.02;

/// Default system (softbody manager) friction
pub const MNGR_DEFAULT_DRAG: f32 = 0.001;

// Default constraint solver iterations
const MNGR_DEFAULT_ITER: usize = 10;

// Range 0 - 1; 1.0 = cannot be deformed
// A value of zero nullifies all rods in the instance
const ROD_DEFORM: f32 = 1.000;

// Range 0 - 1.0; "Rigid" = 1.0
// A value of zero nullifies the translational constraints of all joints
// However, because other constraints are still in place,
// values other than zero produce largely undesired behavior
const JOINT_POS_RIGID: f32 = 1.0;

// Range 0 - 1.0; "Rigid" = 1.0
// Lower values produce springier joints
// A value of zero nullifies the angular constraints of all joints
const JOINT_ANG_RIGID: f32 = 1.0;

// Multiples of std::f32::EPSILON
// A value > 0 is needed to correctly test for intersection containment
// Large values will produce spurious intersections, especially with small
// constraint angles--this leads to "softer" and eventually degenerate
// joints
const JOINT_CONTAINS_BIAS: f32 = 1.0;

// Range 0 - 1.0; "Noisy" = 0.0
// Larger values bias instance velocity and acceleration values to the past
// (low pass filter).
const INTEGRAL_SMOOTH_BIAS: f32 = 0.0;

macro_rules! debug_validate_instance {
    ($instance: expr, $entity: expr) => {
        #[cfg(debug_assertions)] {
            if $instance.is_none() {
                panic!(
                    "Softbody instance for entity {} is None. \
                    You probably attempted to use the Softbody component \
                    before building its instance.",
                    $entity,
                );
            }
        }
    }
}

macro_rules! get_instance {
    ($self: ident, $entity: expr) => {{
        debug_validate_entity!($self, $entity);
        let i = $entity.get_index() as usize;
        debug_validate_instance!($self.instances[i], $entity);
        $self.instances[i].as_ref().unwrap()
    }}
}

macro_rules! get_mut_instance {
    ($self: ident, $entity: expr) => {{
        debug_validate_entity!($self, $entity);
        let i = $entity.get_index() as usize;
        debug_validate_instance!($self.instances[i], $entity);
        $self.instances[i].as_mut().unwrap()
    }}
}

pub trait Iterate {
    #[allow(unused_variables)]
    fn iterate(
        &mut self,
        fixed_delta: f32,
        iterations: usize,
        &mut Manager,
    ) { }
}

struct Particle {
    position: alg::Vec3,
    last: alg::Vec3,
}

impl Particle {
    fn new(position: alg::Vec3) -> Particle {
        Particle {
            position,
            last: position,
        }
    }

    fn displacement(&self) -> alg::Vec3 {
        // TODO: remove usages
        self.position - self.last
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

#[derive(Clone, Copy)]
struct Range {
    min: f32,
    max: f32,
}

impl Range {
    fn zero() -> Range {
        Range { min: 0.0, max: 0.0 }
    }
}

#[derive(Clone, Copy)]
struct ReachPlane {
    normal: alg::Vec3,
}

// Specialized plane struct for joint constraints
impl ReachPlane {
    fn new(left: alg::Vec3, right: alg::Vec3) -> ReachPlane {
        ReachPlane {
            normal: left.cross(right).norm(),
        }
    }

    #[inline]
    fn contains(self, point: alg::Vec3) -> bool {
        self.normal.dot(point) >= 0.0
    }

    #[inline]
    fn contains_biased(self, point: alg::Vec3) -> bool {
        // Allow points just near the plane
        self.normal.dot(point) >= 0.0 - JOINT_CONTAINS_BIAS * std::f32::EPSILON
    }

    #[inline]
    fn intersects(self, ray: alg::Vec3) -> bool {
        self.normal.dot(ray) < 0.0
    }

    #[inline]
    #[allow(dead_code)]
    fn intersection(self, ray: alg::Vec3) -> alg::Vec3 {
        let div = self.normal.dot(ray) + std::f32::EPSILON;
        let scalar = (-alg::Vec3::fwd()).dot(self.normal) / div;
        alg::Vec3::fwd() + ray * scalar
    }

    #[inline]
    fn closest(self, point: alg::Vec3) -> alg::Vec3 {
        let signed_dist = self.normal.dot(point);
        point - self.normal * signed_dist
    }
}

struct ReachCone {
    lower_left:  ReachPlane,
    lower_right: ReachPlane,
    upper_right: ReachPlane,
    upper_left:  ReachPlane,
}

impl ReachCone {
    fn new(x_limit: Range, y_limit: Range) -> ReachCone {
        // Build cone
        let (lower, right, upper, left) = {
            // Linear "approximation" to angles, requires normalization
            let x_min = 2.0 * y_limit.min / std::f32::consts::PI;
            let x_max = 2.0 * y_limit.max / std::f32::consts::PI;
            let y_min = 2.0 * x_limit.min / std::f32::consts::PI;
            let y_max = 2.0 * x_limit.max / std::f32::consts::PI;

            (
                alg::Vec3::new(0.0, y_min, 1.0 - y_min.abs()).norm(), // Lower
                alg::Vec3::new(x_max, 0.0, 1.0 - x_max.abs()).norm(), // Right
                alg::Vec3::new(0.0, y_max, 1.0 - y_max.abs()).norm(), // Upper
                alg::Vec3::new(x_min, 0.0, 1.0 - x_min.abs()).norm(), // Left
            )
        };

        // Build planes
        let lower_left  = ReachPlane::new(left, lower);
        let lower_right = ReachPlane::new(lower, right);
        let upper_right = ReachPlane::new(right, upper);
        let upper_left  = ReachPlane::new(upper, left);

        ReachCone {
            lower_left,
            lower_right,
            upper_right,
            upper_left,
        }
    }
}

struct Joint {
    child: usize,
    x_limit: Range,
    y_limit: Range,
    z_limit: Range,
    unlocked: bool,
    transform: alg::Quat,
    offset: alg::Vec3,
    cone: ReachCone,
}

impl Joint {
    fn new(
        child: usize,
        x_limit: Range,
        y_limit: Range,
        z_limit: Range,
        unlocked: bool,
        transform: alg::Quat,
        offset: alg::Vec3,
    ) -> Joint {
        let cone = ReachCone::new(
            x_limit,
            y_limit,
        );

        Joint {
            child,
            x_limit,
            y_limit,
            z_limit,
            unlocked,
            transform,
            offset,
            cone,
        }
    }

    #[allow(dead_code)]
    fn update_cone(&mut self) {
        self.cone = ReachCone::new(
            self.x_limit,
            self.y_limit,
        );
    }
}

/// Builder pattern for instance joints
pub struct JointBuilder<'a> {
    manager: &'a mut Manager,
    parent: Option<entity::Handle>,
    x_limit: Option<Range>,
    y_limit: Option<Range>,
    z_limit: Option<Range>,
    unlocked: bool,
    fwd: alg::Vec3,
    up: alg::Vec3,
    offset: alg::Vec3,
}

impl<'a> JointBuilder<'a> {
    pub fn new(manager: &'a mut Manager) -> JointBuilder {
        JointBuilder {
            manager,
            parent: None,
            x_limit: None,
            y_limit: None,
            z_limit: None,
            unlocked: false,
            fwd: alg::Vec3::fwd(),
            up: alg::Vec3::up(),
            offset: alg::Vec3::zero(),
        }
    }

    pub fn with_parent(
        &mut self,
        parent: entity::Handle,
    ) -> &'a mut JointBuilder {
        self.parent = Some(parent);
        self
    }

    /// Joint x-axis limit range, in degrees \
    /// Limits greater than 90 degrees on the x axis are not supported
    pub fn x(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        let (min, max) = (min.to_radians(), max.to_radians());
        self.x_limit = Some(Range { min, max });
        self
    }

    /// Joint y-axis limit range, in degrees \
    /// Limits greater than 90 degrees on the y axis are not supported
    pub fn y(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        let (min, max) = (min.to_radians(), max.to_radians());
        self.y_limit = Some(Range { min, max });
        self
    }

    /// Joint z-axis limit range, in degrees \
    /// Full range of motion on the z axis is supported
    pub fn z(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        let (min, max) = (min.to_radians(), max.to_radians());
        self.z_limit = Some(Range { min, max });
        self
    }

    /* Swizzles */

    pub fn xy(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        self.x(min, max).y(min, max)
    }

    pub fn xz(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        self.x(min, max).z(min, max)
    }

    pub fn yz(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        self.y(min, max).z(min, max)
    }

    pub fn xyz(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        self.x(min, max).y(min, max).z(min, max)
    }

    /* Locking */

    /// Unlocked joints will ignore all rotational limits
    pub fn unlock(&mut self) -> &'a mut JointBuilder {
        self.unlocked = true;
        self
    }

    /* Joint transform */

    /// Forward vector of the joint (parent) transform \
    /// Joint children use `start` and `end` instead
    pub fn fwd(&mut self, fwd: alg::Vec3) -> &'a mut JointBuilder {
        self.fwd = fwd;
        self
    }

    /// Up vector of the joint (parent) transform \
    /// Joint children use `start` and `end` instead
    pub fn up(&mut self, up: alg::Vec3) -> &'a mut JointBuilder {
        self.up = up;
        self
    }

    /// Joint offset is relative to the model orientation of the softbody
    /// instance (not the joint transform created with `fwd` and `up`)
    pub fn offset(&mut self, offset: alg::Vec3) -> &'a mut JointBuilder {
        self.offset = offset;
        self
    }

    /// Finalize
    pub fn for_child(&mut self, child: entity::Handle) {
        let parent = self.parent.expect(
            "No parent specified to joint builder"
        );

        debug_assert!(parent != child);

        let expected = !self.unlocked
            || self.x_limit.is_some()
            || self.y_limit.is_some()
            || self.z_limit.is_some();

        let limits = if expected {
            (
                self.x_limit.expect(
                    "No x-axis limit range specified in joint builder"
                ),
                self.y_limit.expect(
                    "No y-axis limit range specified in joint builder"
                ),
                self.z_limit.expect(
                    "No z-axis limit range specified in joint builder"
                ),
            )
        } else {
            (
                Range::zero(),
                Range::zero(),
                Range::zero(),
            )
        };

        let transform = alg::Quat::from_vecs(self.fwd, self.up);

        self.manager.add_joint(
            parent,
            child,
            transform,
            self.offset,
            limits,
            self.unlocked,
        );
    }
}

pub fn extract_orient(prev: alg::Quat, transform: alg::Mat3) -> alg::Quat {
    let mut result = prev;

    for _ in 0..EXTRACT_ITER {
        let rot = result.to_mat();
        let omega = (
              rot.col(0).cross(transform.col(0))
            + rot.col(1).cross(transform.col(1))
            + rot.col(2).cross(transform.col(2))
        ) * (
            1.0 / (
                  rot.col(0).dot(transform.col(0))
                + rot.col(1).dot(transform.col(1))
                + rot.col(2).dot(transform.col(2))
                + std::f32::EPSILON
            ).abs() + 1.0e-9
        );

        let a = omega.mag();
        if 1.0e-9 >= a { break; }

        let dir = omega / a;
        result = alg::Quat::axis_angle(dir, a) * result;
        result = result.norm();
    }

    result
}

#[cfg(debug_assertions)]
struct InstanceIterationDebug {
    plane: f32,
    shape: f32,
}

#[cfg(debug_assertions)]
struct InstanceFrameDebug {
    pos_delta: f32,
    internal_vel: f32,
    err: Vec<InstanceIterationDebug>,
    age: usize,
}

#[cfg(debug_assertions)]
impl InstanceFrameDebug {
    fn new() -> InstanceFrameDebug {
        InstanceFrameDebug {
            pos_delta: 0.0,
            internal_vel: 0.0,
            err: Vec::with_capacity(16),
            age: 0,
        }
    }

    fn log_frame(&self) {
        println!(
            "         pos-delta : {:.2} m\n  \
                   internal-vel : {:.2} m/s",
            self.pos_delta,
            self.internal_vel,
        );

        for (i, e) in self.err.iter().enumerate() {
            println!(
                "   plane-err [{:03}] : {:.2} m\n   \
                    shape-err [{:03}] : {:.2} m",
                i, e.plane,
                i, e.shape,
            );
        }
    }
}

/* TODO: Refactor Instance data structure for memory performance
 * now that you have converged on how/where it is actually used.
 */

pub struct Instance {
    particles: Vec<Particle>,
    rods: Vec<Rod>,
    match_shape: bool, // Actively match shape at runtime

    force: alg::Vec3,
    accel_dt: alg::Vec3, // Cached value, dependent on force

    /* Updated per-frame */

    frame_position: alg::Vec3,
    frame_orient_conj: alg::Quat,
    frame_orient_diff: alg::Quat,
    frame_vel: alg::Vec3,
    frame_accel: alg::Vec3,
    frame_ang_vel: (alg::Vec3, f32),

    /* "Constants" */

    pub mass: f32,
    pub inv_pt_mass: f32, // Cached inverse mass per particle
    end_offset: f32, // Distance from center to simple endpoint
    start_indices: Vec<usize>, // Optional joint start highlight
    end_indices: Vec<usize>, // Optional joint end highlight
    model: Model,

    // Range 0 - 0.5; "Rigid" = 0.5
    // Lower values produce springier meshes
    // A value of zero nullifies all rods in the instance
    pub rigidity: f32,

    #[cfg(debug_assertions)] debug: InstanceFrameDebug,
}

/// Source mesh reference structure.
/// All data is preserved from the original model input (including duplicates).
pub struct Model {
    positions: Vec<alg::Vec3>,
    positions_override: Option<Vec<alg::Vec3>>, // Override offset computations
    com: alg::Vec3, // Point-evaluated (not volume) center of mass
    indices: Vec<usize>, // For computing normals
    normals: Vec<alg::Vec3>,
    duplicates: Vec<usize>, // Mapping from "true" input mesh
}

impl Instance {
    fn new(
        points: &[alg::Vec3],
        indices: &[usize],
        model_override: Option<&[alg::Vec3]>,
        bindings: &[(usize, usize)],
        match_shape: bool,
        mass: f32,
        rigidity: f32,
        initial_pos: alg::Vec3,
        initial_accel_dt: alg::Vec3,
        end_offset: f32,
        start_indices: &[usize],
        end_indices: &[usize],
    ) -> Instance {
        debug_assert!(mass > 0.0);
        debug_assert!(rigidity > 0.0 && rigidity <= 1.0);

        let points_len = points.len();

        /* Initialize particles and base comparison model */

        let (particles, model, duplicates) = {
            let mut particles = Vec::with_capacity(points_len);
            let mut model = Vec::with_capacity(points_len);
            let mut duplicates = Vec::with_capacity(points_len);

            for (i, point) in points.iter().enumerate() {
                particles.push(Particle::new(initial_pos + *point));
                model.push(*point);
                duplicates.push(i); // NOOP
            }

            (particles, model, duplicates)
        };

        let com = model.iter().fold(
            alg::Vec3::zero(),
            |sum, position| sum + *position
        ) / model.len() as f32;

        // Compute base comparison normals for instance
        let normals = if indices.len() > 1 {
            debug_assert!(indices.len() % 3 == 0);
            Instance::compute_normals(
                &particles,
                &indices,
                duplicates.len(),
            )
        } else {
            vec![alg::Vec3::zero()]
        };

        // Initialize rods
        let mut rods = Vec::with_capacity(bindings.len());
        for binding in bindings {
            rods.push(Rod::new(binding.0, binding.1, &particles));
        }

        debug_assert!(points.len() == particles.len());
        debug_assert!(particles.len() == model.len());
        debug_assert!(model.len() == duplicates.len());

        Instance {
            particles,
            rods,
            match_shape,

            force: alg::Vec3::zero(),
            accel_dt: initial_accel_dt,

            frame_position: alg::Vec3::zero(),
            frame_orient_conj: alg::Quat::id(),
            frame_orient_diff: alg::Quat::id(),
            frame_vel: alg::Vec3::zero(),
            frame_accel: alg::Vec3::zero(),
            frame_ang_vel: (alg::Vec3::up(), 0.0),

            mass,
            inv_pt_mass: 1.0 / (mass / points_len as f32),
            model: Model {
                positions: model,
                com,
                positions_override: model_override
                    .map(|positions| positions.to_vec()),
                indices: indices.to_vec(),
                normals,
                duplicates,
            },
            end_offset,
            start_indices: start_indices.to_vec(),
            end_indices: end_indices.to_vec(),
            rigidity,
            #[cfg(debug_assertions)] debug: InstanceFrameDebug::new(),
        }
    }

    fn new_from_model(
        input: &render::ModelData,
        mass: f32,
        rigidity: f32,
        initial_pos: alg::Vec3,
        initial_accel_dt: alg::Vec3,
        end_offset: f32,
        start_indices: &[usize],
        end_indices: &[usize],
    ) -> Instance {
        debug_assert!(mass > 0.0);
        debug_assert!(rigidity > 0.0 && rigidity <= 1.0);

        let vertices_len = input.vertices.len();

        /* Initialize particles and base comparison model */

        let (particles, model, duplicates) = {
            let mut particles: Vec<Particle>
                = Vec::with_capacity(vertices_len); // Overfill
            let mut model: Vec<alg::Vec3>
                = Vec::with_capacity(vertices_len); // Overfill
            let mut duplicate_count = 0;

            // Mapping array from input indices to result indices
            let mut duplicates = Vec::with_capacity(vertices_len);

            let mut i = 0;
            for point in input.vertices.iter().map(|vertex| vertex.position) {
                let mut valid = true;

                // Search previous entries for duplicate position
                for j in 0..i {
                    let compare = model[j];
                    let x = compare.x - point.x;
                    let y = compare.y - point.y;
                    let z = compare.z - point.z;

                    if x*x + y*y + z*z < std::f32::EPSILON {
                        duplicates.push(j);
                        duplicate_count += 1;
                        valid = false;
                        break;
                    }
                }

                if valid {
                    particles.push(Particle::new(initial_pos + point));
                    model.push(point);
                    duplicates.push(i);
                    i += 1;
                }
            }

            if duplicate_count > 0 {
                println!(
                    "{} duplicates found in input model \"{}\"",
                    duplicate_count,
                    input.name,
                );
            }

            debug_assert!(duplicates.len() == input.vertices.len());
            debug_assert!(
                particles.len() == input.vertices.len() - duplicate_count
            );

            (particles, model, duplicates)
        };

        // Convert indices
        let indices: Vec<usize> = input.indices.iter()
            .map(|index| duplicates[*index as usize])
            .collect();

        // Compute center from stored model instead of input model;
        // duplicates throw off the result.
        let com = model.iter().fold(
            alg::Vec3::zero(),
            |sum, position| sum + *position,
        ) / model.len() as f32;

        // Softbodies only support computed normals.
        debug_assert!(input.computed_normals);

        // Compute base comparison normals for instance
        debug_assert!(indices.len() % 3 == 0);
        let normals = Instance::compute_normals(
            &particles,
            &indices,
            duplicates.len(),
        );

        /* Remap start and end indices */

        let mut start_indices: Vec<usize> = start_indices.iter()
            .map(|index| duplicates[*index])
            .collect();

        start_indices.sort_unstable();
        start_indices.dedup();

        let mut end_indices: Vec<usize> = end_indices.iter()
            .map(|index| duplicates[*index])
            .collect();

        end_indices.sort_unstable();
        end_indices.dedup();

        Instance {
            particles,
            rods: Vec::with_capacity(0),
            match_shape: true,

            force: alg::Vec3::zero(),
            accel_dt: initial_accel_dt,

            frame_position: alg::Vec3::zero(),
            frame_orient_conj: alg::Quat::id(),
            frame_orient_diff: alg::Quat::id(),
            frame_vel: alg::Vec3::zero(),
            frame_accel: alg::Vec3::zero(),
            frame_ang_vel: (alg::Vec3::up(), 0.0),

            mass,
            inv_pt_mass: 1.0 / (mass / vertices_len as f32),
            end_offset,
            start_indices,
            end_indices,
            model: Model {
                positions: model,
                com,
                positions_override: None,
                indices,
                normals,
                duplicates,
            },
            rigidity,
            #[cfg(debug_assertions)] debug: InstanceFrameDebug::new(),
        }
    }

    /* Calculate normals for implicit softbody mesh,
     * useful for blending rendered mesh normals
     * and/or determining softbody topology.
     * Note: returns Vec with length equal to the input model
     * (including duplicates), i.e. `actual_len`
     */
    fn compute_normals(
        particles: &[Particle],
        indices: &[usize],
        actual_len: usize,
    ) -> Vec<alg::Vec3> {
        let mut result = vec![alg::Vec3::zero(); actual_len];

        for (i, j, k) in indices.chunks(3)
            .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        {
            let a = particles[i].position;
            let b = particles[j].position;
            let c = particles[k].position;
            let normal = alg::Vec3::normal(a, b, c);

            // Sum normal contributions
            result[i] = result[i] + normal;
            result[j] = result[j] + normal;
            result[k] = result[k] + normal;
        }

        // Rescale
        result.iter().map(|raw| raw.norm_safe()).collect()
    }

    // Must be called when gravity or force changes
    #[inline]
    fn update_cache(&mut self, gravity: alg::Vec3, fixed_dt: f32) {
        self.accel_dt = (self.force * self.inv_pt_mass + gravity)
            * fixed_dt * fixed_dt;
    }

    /* General instance methods */

    pub fn center(&self) -> alg::Vec3 {
        self.particles.iter().fold(
            alg::Vec3::zero(),
            |sum, particle| sum + particle.position
        ) / self.particles.len() as f32
    }

    fn compute_velocity(&self, fixed_dt: f32) -> alg::Vec3 {
        // TODO: mass weighted result
        let new = self.particles.iter().fold(
            alg::Vec3::zero(),
            |sum, particle| sum + particle.displacement(),
        ) / (self.particles.len() as f32 * fixed_dt);

       new.lerp(self.frame_vel, INTEGRAL_SMOOTH_BIAS)
    }

    fn compute_accel(&self, new_vel: alg::Vec3, fixed_dt: f32) -> alg::Vec3 {
        let diff = new_vel - self.frame_vel;
        let new = diff / fixed_dt;
        new.lerp(self.frame_accel, INTEGRAL_SMOOTH_BIAS)
    }

    fn compute_ang_vel(&self, fixed_dt: f32) -> (alg::Vec3, f32) {
        debug_assert!(self.match_shape);

        // Computing via the torque equation
        // grows in error with the magnitude of the angular velocity
        // (consider the 2D case), so we use the frame diff instead.

        let q = self.frame_orient_diff;
        let a = q.abs_angle();
        debug_assert!(a <= std::f32::consts::PI);

        let vec = q.vec();
        let mag = vec.mag();
        let n = if mag < std::f32::EPSILON
            { alg::Vec3::up() } else { vec / mag };

        // Return separately for increased accuracy
        (n, a / fixed_dt)
    }

    /// Returns velocity of instance in meters per second.
    pub fn velocity(&self) -> alg::Vec3 { self.frame_vel }

    /// Returns acceleration of instance in meters per second squared.
    pub fn acceleration(&self) -> alg::Vec3 { self.frame_accel }

    /// Returns axis and angular velocity of instance in radians per second.
    pub fn ang_vel(&self) -> (alg::Vec3, f32) { self.frame_ang_vel }

    /// Returns instance transform using least squares fit. \
    /// `center` is a parameter for optional caching.
    pub fn matched_transform(&self, center: alg::Vec3) -> alg::Mat3 {
        let mut transform = alg::Mat3::zero();

        // Sum product of actual and model particle positions
        for i in 0..self.particles.len() {
            let actual = self.particles[i].position - center;
            let model = self.model.positions[i] - self.model.com;
            transform = transform + (actual * model);
            // Ignore [inverse] model matrix
        }

        transform
    }

    /// Returns instance orientation using least squares fit.
    pub fn matched_orient_polar(&self, center: alg::Vec3) -> alg::Mat3 {
        let transform = self.matched_transform(center);

        // If the mesh self-intersects (e.g. if the rigidity is too low),
        // the transform cannot be fully described by a rotation,
        // which causes it to invert.
        #[cfg(debug_assertions)] {
            if transform.det() < 0.0 {
                eprintln!("Warning: invalid shape matching transform");
            }
        }

        // Compute rotation component using polar decomposition
        let s = (transform.transpose() * transform).sqrt();
        transform * s.inverse()
    }

    /// Returns instance orientation using least squares fit.
    pub fn matched_orient(&self, center: alg::Vec3) -> alg::Quat {
        let result = self.frame_orient_conj.conjugate();
        let transform = self.matched_transform(center);
        extract_orient(result, transform)
    }

    // Call with point == center for a general rotate method
    #[inline]
    pub fn rotate_around(&mut self, rotation: alg::Quat, point: alg::Vec3) {
        // Rotate
        for i in 0..8 {
            self.particles[i].position = rotation
                * (self.particles[i].position - point) // Center rotation axis
                + point; // Move back to world space
        }
    }

    #[inline]
    pub fn translate(&mut self, offset: alg::Vec3) {
        for particle in self.particles.iter_mut() {
            particle.position = particle.position + offset;
        }
    }

    /// Nullify velocity by setting particle last positions \
    /// to current positions.
    #[inline]
    pub fn lock(&mut self) {
        self.particles.iter_mut()
            .for_each(|particle| particle.last = particle.position);
    }

    /// Pin instance position to target. \
    /// Equivalent to calling `translate(target - center)` and `lock()`. \
    /// `center` is a parameter for optional caching.
    pub fn pin(&mut self, target: alg::Vec3, center: alg::Vec3) {
        for particle in self.particles.iter_mut() {
            particle.position = particle.position - center + target;
            particle.last = particle.position;
        }
    }

    // optional caching
    #[inline]
    fn start(&self, center: alg::Vec3, orientation: alg::Mat3) -> alg::Vec3 {
        center + orientation * alg::Vec3::fwd() * self.end_offset * -1.0
    }

    #[inline]
    fn end(&self, center: alg::Vec3, orientation: alg::Mat3) -> alg::Vec3 {
        center + orientation * alg::Vec3::fwd() * self.end_offset * 1.0
    }

    // Convert local point in instance to global.
    // This function is included for readability;
    // often orientation and center are cached and reused.
    #[inline] fn extend(
        &self,
        offset: alg::Vec3,
        orientation: alg::Mat3,
        center: alg::Vec3,
    ) -> alg::Vec3 {
        center + orientation * offset
    }
}

/// Builder pattern for softbody instances
pub struct InstanceBuilder<'a> {
    manager: &'a mut Manager,
    scale: Option<alg::Vec3>, // For optional box limb creation
    model: Option<&'a render::ModelData>, // For optional model starter
    mass: f32,
    rigidity: f32,
    particles: Option<&'a [alg::Vec3]>,
    indices: Option<&'a [usize]>,
    bindings: Option<&'a [(usize, usize)]>,
    initial_pos: alg::Vec3,
    match_shape: bool,
    end_offset: Option<f32>,
    start_indices: Option<&'a [usize]>,
    end_indices: Option<&'a [usize]>,
}

impl<'a> InstanceBuilder<'a> {
    // Initialize with manager
    pub fn new(manager: &mut Manager) -> InstanceBuilder {
        InstanceBuilder {
            manager,
            scale: None,
            model: None,
            mass: INST_DEFAULT_MASS,
            rigidity: INST_DEFAULT_RIGID,
            particles: None,
            indices: None,
            bindings: None,
            initial_pos: alg::Vec3::zero(),
            match_shape: false,
            end_offset: None, // Default to no simple endpoint
            start_indices: None,
            end_indices: None,
        }
    }

    /// Override general instance creation with box limb preset.
    /// Takes in the box scale as an argument.
    pub fn make_box_limb(
        &mut self,
        scale: alg::Vec3,
    ) -> &mut InstanceBuilder<'a> {
        self.scale = Some(scale);
        self
    }

    /// Override general instance creation with mesh.
    pub fn from_model(
        &mut self,
        model: &'a render::ModelData,
    ) -> &mut InstanceBuilder<'a> {
        self.model = Some(model);
        self
    }

    pub fn mass(&mut self, mass: f32) -> &mut InstanceBuilder<'a> {
        self.mass = mass;
        self
    }

    /// Rigidity is in the range (0, 1]
    pub fn rigidity(&mut self, rigidity: f32) -> &mut InstanceBuilder<'a> {
        debug_assert!(rigidity > 0.0 && rigidity <= 1.0);
        self.rigidity = rigidity;
        self
    }

    pub fn particles(
        &mut self,
        particles: &'a [alg::Vec3],
    ) -> &mut InstanceBuilder<'a> {
        self.particles = Some(particles);
        self
    }

    pub fn indices(
        &mut self,
        indices: &'a [usize],
    ) -> &mut InstanceBuilder<'a> {
        self.indices = Some(indices);
        self
    }

    pub fn bindings(
        &mut self,
        bindings: &'a [(usize, usize)],
    ) -> &mut InstanceBuilder<'a> {
        self.bindings = Some(bindings);
        self
    }

    pub fn initial_pos(
        &mut self,
        position: alg::Vec3,
    ) -> &mut InstanceBuilder<'a> {
        self.initial_pos = position;
        self
    }

    /// Enable active shape matching
    pub fn match_shape(&mut self) -> &mut InstanceBuilder<'a> {
        self.match_shape = true;
        self
    }

    /// Distance from center of limb to simple endpoint (start and end).
    /// This is only necessary for instances that will be joint children.
    pub fn end_offset(&mut self, offset: f32) -> &mut InstanceBuilder<'a> {
        self.end_offset = Some(offset);
        self
    }

    /// Highlight indices to use for the joint start target.
    /// This is only necessary for instances that will be joint children. \
    /// The instance's `start` point will be taken as the aggregate position
    /// of the vertices at these indices. \
    /// Input indices are with respect to the input model (if one exists),
    /// and as such must include duplicates.
    pub fn start(&mut self, indices: &'a [usize]) -> &mut InstanceBuilder<'a> {
        self.start_indices = Some(indices);
        self
    }

    /// Highlight indices to use for the joint end target.
    /// This is only necessary for instances that will be joint children. \
    /// The instance's `end` point will be taken as the aggregate position
    /// of the vertices at these indices. \
    /// Input indices are with respect to the input model (if one exists),
    /// and as such must include duplicates.
    pub fn end(&mut self, indices: &'a [usize]) -> &mut InstanceBuilder<'a> {
        self.end_indices = Some(indices);
        self
    }

    /// Finalize
    pub fn for_entity(&mut self, entity: entity::Handle) {
        let initial_accel_dt = self.manager.gravity // Initialize with gravity
            * self.manager.fixed_dt * self.manager.fixed_dt;

        #[cfg(debug_assertions)] {
            if self.start_indices.is_some() && !self.end_indices.is_some() {
                panic!("Start indices were set but end indices were not set");
            }
            if self.end_indices.is_some() && !self.start_indices.is_some() {
                panic!("End indices were set but start indices were not set");
            }
        }

        /* Box limb instance */

        let instance = if let Some(scale) = self.scale {
            let scale = scale * 0.5;

            debug_assert!(self.model.is_none());
            debug_assert!(self.particles.is_none());
            debug_assert!(self.indices.is_none());
            debug_assert!(self.bindings.is_none());
            debug_assert!(self.start_indices.is_none());
            debug_assert!(self.end_indices.is_none());

            // Build 8-particle scaled box
            Instance::new(
                &[
                    // Front face (CW)
                    alg::Vec3::new(-scale.x,  scale.y, -scale.z), // 0
                    alg::Vec3::new( scale.x,  scale.y, -scale.z), // 1
                    alg::Vec3::new( scale.x, -scale.y, -scale.z), // 2
                    alg::Vec3::new(-scale.x, -scale.y, -scale.z), // 3

                    // Back face (CW)
                    alg::Vec3::new(-scale.x,  scale.y, scale.z), // 4
                    alg::Vec3::new( scale.x,  scale.y, scale.z), // 5
                    alg::Vec3::new( scale.x, -scale.y, scale.z), // 6
                    alg::Vec3::new(-scale.x, -scale.y, scale.z), // 7
                ],
                // CW triangle indices
                &[
                    0, 1, 2, // Front face
                    2, 3, 0,
                    4, 7, 6, // Back face
                    6, 5, 4,
                    0, 4, 5, // Top face
                    5, 1, 0,
                    3, 2, 6, // Bottom face
                    6, 7, 3,
                    0, 3, 7, // Left face
                    7, 4, 0,
                    6, 2, 1, // Right face
                    1, 5, 6,
                ],
                /* Override scaled model with unit cube;
                 * enables offsets to work properly with multiple scaled
                 * versions of the same mesh.
                 * While this can change the magnitude of the scale matrix
                 * during shape matching, this doesn't actually affect the
                 * output.
                 */
                Some(&[
                    // Front face (CW)
                    alg::Vec3::new(-0.5,  0.5, -0.5),
                    alg::Vec3::new( 0.5,  0.5, -0.5),
                    alg::Vec3::new( 0.5, -0.5, -0.5),
                    alg::Vec3::new(-0.5, -0.5, -0.5),

                    // Back face (CW)
                    alg::Vec3::new(-0.5,  0.5,  0.5),
                    alg::Vec3::new( 0.5,  0.5,  0.5),
                    alg::Vec3::new( 0.5, -0.5,  0.5),
                    alg::Vec3::new(-0.5, -0.5,  0.5),
                ]),
                &[], // Ignore bindings
                true, // Match shape
                self.mass,
                self.rigidity,
                self.initial_pos,
                initial_accel_dt,
                self.end_offset.unwrap_or(0.5),
                &[0, 1, 2, 3], // Start indices
                &[4, 5, 6, 7], // End indices
            )
        }

        /* Mesh */

        else if let Some(model) = self.model {
            debug_assert!(self.particles.is_none());
            debug_assert!(self.indices.is_none());
            debug_assert!(self.bindings.is_none());

            Instance::new_from_model(
                model,
                self.mass,
                self.rigidity,
                self.initial_pos,
                initial_accel_dt,
                self.end_offset.unwrap_or(0.0),
                self.start_indices.unwrap_or(&[]),
                self.end_indices.unwrap_or(&[]),
            )
        }

        /* Generic instance */

        else {
            debug_assert!(self.particles.is_some());
            debug_assert!(self.indices.is_some());

            Instance::new(
                self.particles.unwrap(),
                self.indices.unwrap(),
                None, // No model override
                self.bindings.unwrap_or(&[]),
                self.match_shape,
                self.mass,
                self.rigidity,
                self.initial_pos,
                initial_accel_dt,
                self.end_offset.unwrap_or(0.0),
                self.start_indices.unwrap_or(&[]),
                self.end_indices.unwrap_or(&[]),
            )
        };

        // Register with manager
        self.manager.add_instance(instance, entity);
    }
}

// Data layout assumes many physics objects (but may still be sparse)
pub struct Manager {
    handles: Vec<Option<entity::Handle>>,
    instances: Vec<Option<Instance>>,
    count: usize,

    joints: fnv::FnvHashMap<usize, Vec<Joint>>,
    planes: Vec<alg::Plane>,

    pub iterations: usize,
    fixed_dt: f32,
    gravity: alg::Vec3,
    drag: f32,
    friction: f32,
    bounce: f32,
}

impl components::Component for Manager {
    fn register(&mut self, entity: entity::Handle) {
        let i = entity.get_index() as usize;

        // Resize array to fit new entity
        loop {
            if i >= self.instances.len() {
                self.handles.push(None);
                self.instances.push(None);
                continue;
            }

            break;
        }

        self.handles[i] = Some(entity);
        self.count += 1;

        debug_assert!(self.handles.len() == self.instances.len());
    }

    fn registered(&self, entity: entity::Handle) -> bool {
        let i = entity.get_index() as usize;
        i < self.instances.len() && self.handles[i].is_some()
    }

    fn count(&self) -> usize {
        self.count
    }

    #[cfg(debug_assertions)] fn debug_name(&self) -> &str { "Softbody" }
}

impl Manager {
    pub fn new(
        fixed_dt: f32,
        instance_hint: usize,
        joint_hint: usize,
        plane_hint: usize,
    ) -> Manager {
        let joint_map = fnv::FnvHashMap::with_capacity_and_hasher(
            joint_hint,
            Default::default(),
        );

        Manager {
            handles: Vec::with_capacity(instance_hint),
            instances: Vec::with_capacity(instance_hint),
            count: 0,
            joints: joint_map,
            planes: Vec::with_capacity(plane_hint),
            iterations: MNGR_DEFAULT_ITER,
            fixed_dt: fixed_dt,
            gravity: alg::Vec3::new(0., -9.8, 0.), // Default gravity
            drag: MNGR_DEFAULT_DRAG,
            friction: MNGR_DEFAULT_FRICTION,
            bounce: MNGR_DEFAULT_BOUNCE,
        }
    }

    /// Get instance builder that can be used to initialize the softbody
    /// instance for this entity.
    pub fn build_instance(&mut self) -> InstanceBuilder {
        InstanceBuilder::new(self)
    }

    fn add_instance(&mut self, instance: Instance, entity: entity::Handle) {
        debug_validate_entity!(self, entity);
        let i = entity.get_index() as usize;
        self.instances[i] = Some(instance);
    }

    pub fn get_instance(&self, entity: entity::Handle) -> &Instance {
        get_instance!(self, entity)
    }

    pub fn get_mut_instance(
        &mut self,
        entity: entity::Handle,
    ) -> &mut Instance {
        get_mut_instance!(self, entity)
    }

    pub fn energy(&self) -> f32 {
        self.instances.iter().filter_map(|i| i.as_ref())
            .map(
                |i| i.particles.iter()
                    .map(
                        move |p|
                        (1. / i.inv_pt_mass, p.displacement().mag() / self.fixed_dt)
                    )
            ).flatten()
                .fold(0., |acc, p| acc + 0.5 * p.0 * p.1 * p.1) // KE
    }

    /// Returns closest particle in specified direction. \
    /// `center` is a parameter for optional caching.
    pub fn closest_point(
        &self,
        entity: entity::Handle,
        direction: alg::Vec3, // Does not need to be normalized
        center: alg::Vec3,
    ) -> alg::Vec3 {
        let instance = get_instance!(self, entity);
        instance.particles.iter().fold(
            (std::f32::MIN, alg::Vec3::zero()),
            |result, particle| {
                let dot = (particle.position - center).dot(direction);
                if dot > result.0 { (dot, particle.position) } else { result }
            }
        ).1
    }

    /// Returns closest point on bounding box in specified direction. \
    /// `center` is a parameter for optional caching.
    pub fn closest_point_bounded(
        &self,
        entity: entity::Handle,
        direction: alg::Vec3,
        center: alg::Vec3,
    ) -> alg::Vec3 {
        let point = self.closest_point(entity, direction, center);
        let direction_norm = direction.norm();
        center + direction_norm * direction_norm.dot(point - center)
    }

    /// Returns joint pivot point in worldspace between two instances.
    pub fn pivot(
        &self,
        parent: entity::Handle,
        child: entity::Handle,
    ) -> alg::Vec3 {
        debug_validate_entity!(self, parent);
        debug_validate_entity!(self, child);

        let i = parent.get_index() as usize;
        debug_validate_instance!(self.instances[i], parent);

        let j = child.get_index() as usize;
        debug_validate_instance!(self.instances[j], child);

        let parent_instance = self.instances[i].as_ref().unwrap();

        match self.joints.get(&i) {
            Some(joints) => for joint in joints {
                if joint.child == j {
                    let center = parent_instance.center();
                    let orient = parent_instance.matched_orient(center).to_mat();

                    return parent_instance.extend(
                        joint.offset,
                        orient,
                        center,
                    );
                }
            },
            None => panic!(
                "Softbody instance for entity {} is not a joint parent.",
                parent,
            ),
        }

        panic!(
            "Softbody instance for entity {} does not have a joint child {}",
            parent, child,
        )
    }

    /// Returns weighted center of mass of a slice of instances.
    pub fn com(&self, entities: &[entity::Handle]) -> alg::Vec3 {
        let sum = entities.iter()
            .map(|handle| get_instance!(self, *handle))
            .map(|instance| (instance.center(), instance.mass))
            .fold(
                (alg::Vec3::zero(), 0f32),
                |sum, (center, mass)| (sum.0 + center * mass, sum.1 + mass)
            );

        sum.0 / sum.1
    }

    /// Returns weighted velocity of a slice of instances.
    pub fn velocity(&self, entities: &[entity::Handle]) -> alg::Vec3 {
        let sum = entities.iter()
            .map(|handle| get_instance!(self, *handle))
            .map(|instance| (instance.velocity(), instance.mass))
            .fold(
                (alg::Vec3::zero(), 0f32),
                |sum, (v, mass)| (sum.0 + v * mass, sum.1 + mass)
            );

        sum.0 / sum.1
    }

    pub fn set_force(&mut self, entity: entity::Handle, force: alg::Vec3) {
        let instance = get_mut_instance!(self, entity);
        instance.force = force;
        instance.update_cache(self.gravity, self.fixed_dt);
    }

    /// Get instance particle offsets from the model.
    pub(super) fn get_position_offsets(
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
            #[cfg(debug_assertions)] {
                if instance.particles.len() > render::MAX_SOFTBODY_VERT {
                    panic!(
                        "Softbody instance for entity {} \
                        has too many vertices!",
                        entity,
                    );
                }
            }

            // Duplicates will cause repeat computations
            for (i, j) in instance.model.duplicates.iter()
                .map(|index| *index as usize)
                .enumerate()
            {
                // Get offset from center; compare current transform against
                // model reference
                let offset = instance.frame_orient_conj * (
                    instance.particles[j].position - instance.frame_position
                ) - instance.model.positions_override.as_ref()
                    .unwrap_or(&instance.model.positions)[j];

                offsets[i] = render::PaddedVec3::new(offset);
            }
        }

        offsets
    }

    /// Get instance particle offsets from the normals model.
    pub(super) fn get_normal_offsets(
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
            #[cfg(debug_assertions)] {
                if instance.particles.len() > render::MAX_SOFTBODY_VERT {
                    panic!(
                        "Softbody instance for entity {} \
                        has too many vertices!",
                        entity,
                    );
                }
            }

            let new = Instance::compute_normals(
                &instance.particles,
                &instance.model.indices,
                instance.model.duplicates.len(),
            );

            // Compute offsets
            for i in 0..new.len() {
                offsets[i] = render::PaddedVec3::new(
                    instance.frame_orient_conj * new[i]
                        - instance.model.normals[i]
                );
            }
        }

        offsets
    }

    /// Get joint builder that can be used to add a joint to the softbody
    /// instance for this entity.
    pub fn build_joint(&mut self) -> JointBuilder {
        JointBuilder::new(self)
    }

    fn add_joint(
        &mut self,
        parent: entity::Handle,
        child: entity::Handle,
        transform: alg::Quat,
        offset: alg::Vec3,
        limits: (Range, Range, Range),
        unlocked: bool,
    ) {
        debug_validate_entity!(self, parent);
        debug_validate_entity!(self, child);

        let (i, j) = (
            parent.get_index() as usize,
            child.get_index() as usize,
        );

        debug_validate_instance!(self.instances[i], parent);
        debug_validate_instance!(self.instances[j], child);

        let joint = Joint::new(
            j, // Child index
            limits.0,
            limits.1,
            limits.2,
            unlocked,
            transform,
            offset,
        );

        { // Initialize new instance in optimal position and orientation
            let (parent, child) = unsafe {
                let p = self.instances.as_ptr().offset(i as isize);
                let c = self.instances.as_mut_ptr().offset(j as isize);

                (
                    (*p).as_ref().unwrap(),
                    (*c).as_mut().unwrap(), // Get child as mutable
                )
            };

            /* Align child with parent and joint transform */

            let parent_center = parent.center();
            let parent_orient = parent.matched_orient(parent_center);
            let rotation = parent_orient * transform;

            let child_center = child.center();
            let child_orient = child.matched_orient(child_center).to_mat();

            let child_start = child.start(
                child_center,
                child_orient,
            );

            let child_end = child.end(
                child_center,
                child_orient,
            );

            let end = (child_end - child_start) * 0.5;
            let position = parent.extend(offset, parent_orient.to_mat(), parent_center)
                + rotation * end;

            child.rotate_around(rotation, child_center);
            child.translate(position - child_center);

            // Reset child position for integrator
            child.lock();
        }

        // Check if this parent already has a joint
        if let Some(entry) = self.joints.get_mut(&i) {
            entry.push(joint);
            return;
        }

        // Otherwise, create a new Vec
        self.joints.insert(i, vec![joint]);
    }

    pub fn add_plane(&mut self, plane: alg::Plane) {
        self.planes.push(plane);
    }

    pub fn add_planes(&mut self, planes: &[alg::Plane]) {
        planes.iter().for_each(|plane| self.add_plane(*plane));
    }

    /// Set gravity for all instances. \
    /// Heavier call than `set_gravity_raw(...)`, \
    /// but will force-update all instances.
    pub fn set_gravity(&mut self, gravity: alg::Vec3) {
        self.gravity = gravity;

        for i in 0..self.instances.len() {
            if let Some(ref mut instance) = self.instances[i] {
                instance.update_cache(self.gravity, self.fixed_dt);
            }
        }
    }

    /// Set gravity for all instances. \
    /// May not immediately affect all instances.
    pub fn set_gravity_raw(&mut self, gravity: alg::Vec3) {
        self.gravity = gravity;
    }

    /// Range 0 - 1; 0 = no drag; 1 = nothing moves
    pub fn set_drag(&mut self, drag: f32) {
        self.drag = drag;
    }

    /// Range 0 - 1; 0 = no planar friction
    pub fn set_friction(&mut self, friction: f32) {
        self.friction = friction;
    }

    /// Range 0 - inf; "Realistic" = 2.0 \
    /// Values < 2 become force zones, values > 2 add impossible force. \
    /// A value of zero nullifies all collisions.
    pub fn set_bounce(&mut self, bounce: f32) {
        self.bounce = bounce;
    }

    pub(crate) fn simulate<T>(
        &mut self,
        game: &mut T,
        transforms: &mut transform::Manager
    ) where T: Iterate {
        let mut dbg_idx = 0;

        // Update instance particles
        for i in 0..self.instances.len() {
            let instance = match self.instances[i] {
                Some(ref mut instance) => instance,
                None => continue,
            };

            // Position Verlet
            for particle in &mut instance.particles {
                let next_position = particle.position * (2.0 - self.drag)
                    + (instance.accel_dt - particle.last)
                    * (1.0 - self.drag);

                particle.last = particle.position;
                particle.position = next_position;
            }

            #[cfg(debug_assertions)] {
                instance.debug.err.clear();
            }
        }

        // Solve constraints
        for _ in 0..self.iterations {
            for i in 0..self.instances.len() {
                let instance = match self.instances[i] {
                    Some(ref mut instance) => instance,
                    None => continue,
                };

                #[cfg(debug_assertions)]
                instance.debug.err.push(
                    InstanceIterationDebug {
                        plane: 0.0,
                        shape: 0.0,
                    }
                );

                #[cfg(debug_assertions)] {
                    dbg_idx = instance.debug.err.len() - 1;
                }

                // Rods
                for rod in &instance.rods {
                    let left = instance.particles[rod.left].position;
                    let right = instance.particles[rod.right].position;

                    let difference = right - left;
                    let distance = difference.mag();

                    let offset = difference * instance.rigidity
                        * (rod.length / distance - 1.);

                    instance.particles[rod.left].position = left - offset;
                    instance.particles[rod.right].position = right + offset;
                }

                // Shape matching
                if instance.match_shape {
                    let center = instance.center();
                    let orientation = instance.matched_orient(center);

                    /* TODO: currently disabled
                     *       so that matching is always predicted
                     *       from the previous good _frame_ (not substep).
                    instance.frame_orient_conj = orientation.conjugate();
                    */

                    for (particle, model_position) in instance.particles
                        .iter_mut().zip(&instance.model.positions)
                    {
                        let target = orientation
                            * (*model_position - instance.model.com)
                            + center;

                        let offset = target - particle.position;
                        particle.position = particle.position
                            + offset * instance.rigidity;

                        #[cfg(debug_assertions)] {
                            instance.debug.err[dbg_idx].shape
                                += offset.mag().powf(2.0);
                        }
                    }

                    #[cfg(debug_assertions)] {
                        let len = instance.particles.len() as f32;
                        instance.debug.err[dbg_idx].shape = (
                            instance.debug.err[dbg_idx].shape / len
                        ).sqrt();
                    }
                }

                // Deformity
                for rod in &mut instance.rods {
                    let left = instance.particles[rod.left].position;
                    let right = instance.particles[rod.right].position;

                    rod.length = f32::min(
                        f32::max(left.dist(right), rod.length * ROD_DEFORM),
                        rod.length,
                    );
                }
            }
        }

        // Solve abstracted constraints
        // Note: "true" delta time is FIXED_DT / ITERATIONS
        for _ in 0..self.iterations {
            // External constraints
            game.iterate(self.fixed_dt, self.iterations, self);

            // Joint constraints
            self.solve_joints();
        }

        #[cfg(debug_assertions)] { dbg_idx = 0 }

        // Apply velocity constraints in one pass
        for i in 0..self.instances.len() {
            let instance = match self.instances[i] {
                Some(ref mut instance) => instance,
                None => continue,
            };

            for plane in &self.planes {
                for particle in &mut instance.particles {
                    let distance = plane.dist(particle.position);

                    if distance > 0. {
                        continue;
                    }

                    let disp = particle.displacement();

                    /* Plane collision */

                    if distance < 0. {
                        particle.position = particle.position
                            - plane.normal * self.bounce * distance;

                        if self.bounce >= 1. {
                            let b = self.bounce - 1.0;
                            let v = plane.normal.dot(disp);
                            let last_distance = plane.dist(particle.last);
                            particle.last = particle.last + plane.normal
                                * (b * (v - distance) - last_distance);
                        }

                        #[cfg(debug_assertions)] {
                            instance.debug.err[dbg_idx].plane -= distance;
                        }
                    }

                    /* Plane friction */

                    let direction = disp.norm_safe();
                    let tangent = direction
                        .cross( plane.normal)
                        .cross(-plane.normal)
                        .norm_safe();

                    let factor = tangent.dot(disp);
                    let projected = tangent * factor;
                    particle.position = particle.position
                        - projected * self.friction;
                }

                #[cfg(debug_assertions)] {
                    instance.debug.err[dbg_idx].plane /=
                        instance.particles.len() as f32;
                }
            }

            #[cfg(debug_assertions)] { dbg_idx += 1 }
        }

        // Finalize instances
        for i in 0..self.instances.len() {
            let mut instance = match self.instances[i] {
                Some(ref mut instance) => instance,
                None => continue,
            };

            // Compute average position and best fit orientation
            let center = instance.center();
            let orientation = instance.matched_orient(center);

            #[cfg(debug_assertions)] {
                instance.debug.pos_delta = if instance.debug.age > 0 {
                    (center - instance.frame_position).mag()
                } else { 0.0 };
            }

            // Update instance position and orientation
            instance.frame_position = center;
            instance.frame_orient_diff = (
                instance.frame_orient_conj * orientation
            ).norm();
            instance.frame_orient_conj = orientation.conjugate();

            // Update transform
            debug_validate_entity!(transforms, self.handles[i].unwrap());
            transforms.set_raw(i, center, orientation, alg::Vec3::one());

            #[cfg(debug_assertions)] {
                instance.debug.internal_vel = 0.0;
            }

            #[cfg(debug_assertions)] {
                for particle in &mut instance.particles {
                    let disp = particle.displacement(); // Meters per FIXED_DT
                    instance.debug.internal_vel += disp.mag() / self.fixed_dt;
                }
            }

            #[cfg(debug_assertions)] {
                instance.debug.internal_vel /= instance.particles.len() as f32;
                instance.debug.age += 1;
            }

            let new_vel = instance.compute_velocity(self.fixed_dt);
            instance.frame_accel = instance.compute_accel(new_vel, self.fixed_dt);
            instance.frame_vel = new_vel;

            if instance.match_shape {
                instance.frame_ang_vel = instance.compute_ang_vel(self.fixed_dt);
            }
        }
    }

    #[inline]
    fn solve_joints(&mut self) {
        for (parent_index, joints) in &self.joints {
            /* Unsafely acquire mutable references to vector elements.
             * Unfortunately, the only safe Rust alternative (split_at_mut())
             * is slower.
             */

            let parent = unsafe {
                let ptr = self.instances.as_mut_ptr()
                    .offset(*parent_index as isize);

                (*ptr).as_mut().unwrap()
            };

            let mut children = Vec::with_capacity(joints.len());

            for joint in joints {
                let child = unsafe {
                    let ptr = self.instances.as_mut_ptr()
                        .offset(joint.child as isize);

                    (*ptr).as_mut().unwrap()
                };

                children.push(child);
            }

            debug_assert!(children.len() == joints.len());

            /* Constrain orientations to joint connection */

            for i in 0..children.len() {
                // Calculate mass imbalance
                let weight = 1. / (children[i].mass / parent.mass + 1.);

                /* Recompute parent center, orientation, start/end */

                let parent_center = parent.center();
                let parent_orient = parent.matched_orient(parent_center).to_mat();

                let parent_start = parent.extend(
                    -joints[i].offset,
                    parent_orient,
                    parent_center,
                );

                let parent_end = parent.extend(
                    joints[i].offset,
                    parent_orient,
                    parent_center,
                );

                /* Recompute child center, orientation, start/end */

                let child_center = children[i].center();
                let child_orient = children[i].matched_orient(child_center).to_mat();

                let child_start = children[i].start(
                    child_center,
                    child_orient,
                );

                let child_end = children[i].end(
                    child_center,
                    child_orient,
                );

                // Find midpoint for initial correction
                let midpoint = child_start.lerp(parent_end, weight);

                /* Rotate child towards midpoint */

                let child_fwd = child_orient * alg::Vec3::fwd();

                // Ensure the child endpoints are aligned with the child
                // forward direction
                #[cfg(debug_assertions)] {
                    let compare = (child_end - child_start).norm();
                    if child_fwd.dot(compare) < 0.99 {
                        panic!(
                            "Softbody instance orientation \
                            and start/end do not match!",
                        );
                    }
                }

                let child_correction = alg::Quat::from_to(
                    child_fwd,
                    (child_end - midpoint).norm(),
                );

                children[i].rotate_around(child_correction, child_end);

                /* Rotate parent towards midpoint,
                 * taking joint transform into account
                 */

                let parent_correction = alg::Quat::from_to(
                    (parent_end - parent_start).norm(),
                    (midpoint - parent_start).norm(),
                );

                // Rotate around joint "start" position
                parent.rotate_around(parent_correction, parent_start);

                /* Constrain positions */

                // Recompute child orientation
                let child_center = children[i].center();
                let child_orient = children[i].matched_orient(child_center).to_mat();

                let child_start = children[i].start(
                    child_center,
                    child_orient,
                );

                let offset = (child_start - parent_end)
                    * -JOINT_POS_RIGID;

                children[i].translate(offset * weight);
                parent.translate(-offset * (1. - weight));
            }

            /* Constrain orientations to limits */

            for i in 0..children.len() {
                // Correct parent and child
                Manager::solve_joint_rotation(
                    parent,
                    &mut children[i],
                    &joints[i],
                );
            }
        }
    }

    fn solve_joint_rotation(
        parent: &mut Instance,
        child: &mut Instance,
        joint: &Joint,
    ) {
        // Ignore unlocked joints (slow)
        if joint.unlocked { return }

        let parent_center = parent.center();
        let parent_orient = parent.matched_orient(parent_center).to_mat();
        let child_center = child.center();
        let child_orient = child.matched_orient(child_center).to_mat();
        let child_orient_inv = child_orient.transpose();

        // Joint transform is treated as child of parent limb
        let joint_global = parent_orient * joint.transform.to_mat();
        let joint_global_inv = joint_global.transpose();

        // Get child in local space
        let local_child = (joint_global_inv * child_orient).to_quat();
        let local_child_fwd = local_child * alg::Vec3::fwd();

        // Compute simple and twist rotations
        let simple = alg::Quat::simple(alg::Vec3::fwd(), local_child_fwd);
        let twist = simple.conjugate() * local_child;

        // Is the rotation inside the cone?
        let inside = joint.cone.lower_left.contains(local_child_fwd)
            && joint.cone.lower_right.contains(local_child_fwd)
            && joint.cone.upper_right.contains(local_child_fwd)
            && joint.cone.upper_left.contains(local_child_fwd);

        // Rebind (limit) simple
        let simple = if !inside {
            Manager::limit_simple_joint(&joint.cone, local_child_fwd)
        } else { simple };

        let simple = if joint.x_limit.min == joint.x_limit.max
                     || joint.y_limit.min == joint.y_limit.max {
            alg::Quat::id()
        } else { simple };

        /* Rebind (limit) twist */

        let (axis, angle) = twist.to_axis_angle();

        // Test to see if axis has flipped
        // (forcing positive angles)
        let sign = if axis.dot(alg::Vec3::fwd()) > 0.0 { -1.0 } else { 1.0 };

        let limited = (angle * sign)
            .max(joint.z_limit.min)
            .min(joint.z_limit.max);

        // Flip sign again to undo the limit comparison sign change
        let twist = alg::Quat::axis_angle(alg::Vec3::fwd(), -limited);

        // Calculate mass imbalance
        let weight = 1. / (child.mass / parent.mass + 1.);

        /* Correct child */

        let point = child.start(child_center, child_orient);

        // Clear child, apply new rotation, apply parent joint
        let child_correction = joint_global
            * (simple * twist).to_mat()
            * child_orient_inv;

        child.rotate_around(
            child_correction.to_quat()
                .pow(weight * JOINT_ANG_RIGID),
            point,
        );

        /* Correct parent */

        // Transform from the same position as the midpoint correction
        // Reuse parent center and orientation from above
        let point = parent.extend(-joint.offset, parent_orient, parent_center);

        let parent_correction = child_orient
            * (simple * twist).conjugate().to_mat()
            * joint_global_inv;

        parent.rotate_around(
            parent_correction.to_quat()
                .pow((1. - weight) * JOINT_ANG_RIGID),
            point,
        );

        // TODO: Add back transform decomposition code
    }

    // Returns limited simple rotation given cone and forward vector
    fn limit_simple_joint(
        cone: &ReachCone,
        local_child_fwd: alg::Vec3,
    ) -> alg::Quat {
        // Calculate intersection of ray with cone
        let intersection = {
            let mut candidates = Vec::with_capacity(2);
            let mut plane = cone.lower_left;

            // Linear rotation path (ray) is
            // (alg::Vec3::fwd() + local_child_fwd - alg::Vec3::fwd()).norm()
            // which can be simplified to local_child_fwd

            if cone.lower_left.intersects(local_child_fwd) {
                candidates.push(
                    cone.lower_left.closest(local_child_fwd)
                );
            }

            if cone.lower_right.intersects(local_child_fwd) {
                if candidates.is_empty() {
                    plane = cone.lower_right;
                }

                candidates.push(
                    cone.lower_right.closest(local_child_fwd)
                );
            }

            if cone.upper_right.intersects(local_child_fwd) {
                if candidates.is_empty() {
                    plane = cone.upper_right;
                }

                candidates.push(
                    cone.upper_right.closest(local_child_fwd)
                );
            }

            if cone.upper_left.intersects(local_child_fwd) {
                if candidates.is_empty() {
                    plane = cone.upper_left;
                }

                candidates.push(
                    cone.upper_left.closest(local_child_fwd)
                );
            }

            debug_assert!(!candidates.is_empty());
            let mut result = candidates[0];

            if candidates.len() > 1 {
                let compare = if candidates.len() == 2
                    || candidates[0] == candidates[2] // X: -90 to 90
                {
                    candidates[1]
                } else {
                    // Y: -90 to 90
                    candidates[2]
                };

                // Solution should be inside all four
                let inside = cone.lower_left.contains_biased(compare)
                    && cone.lower_right.contains_biased(compare)
                    && cone.upper_right.contains_biased(compare)
                    && cone.upper_left.contains_biased(compare);

                if inside {
                    result = compare;
                }

                // Both candidates are outside
                else if !cone.lower_left.contains_biased(candidates[0])
                    || !cone.lower_right.contains_biased(candidates[0])
                    || !cone.upper_right.contains_biased(candidates[0])
                    || !cone.upper_left.contains_biased(candidates[0])
                {
                    result = plane.closest(compare);
                }
            }

            result
        };

        // Calculate rotation midpoint
        let midpoint = intersection.norm();

        // Limit rotation
        alg::Quat::simple(alg::Vec3::fwd(), midpoint)
    }

    #[cfg(debug_assertions)]
    pub fn dump_config(&self) {
        let mut str = String::new();
        str += &format!("\n # # # nmg/softbody/config # # # \n\n");

        str += &format!(
            "iter={}\n\
             grav={}\n\
             drag={}\n\
             frct={}\n\
             bnce={}\n",
            self.iterations,
            self.gravity,
            self.drag,
            self.friction,
            self.bounce,
        );

        for (i, inst) in self.instances.iter()
            .filter_map(|i| i.as_ref())
            .enumerate()
        {
            str += &format!("\n\t-- instance {} --\n", i);
            str += &format!(
                " mass={}\n\
                 rigid={}\n\
                 match={}\n\
                 force={}\n",
                inst.mass,
                inst.rigidity,
                if inst.match_shape { 1 } else { 0 },
                inst.force,
            );

            str += &format!("\n\t-- particles={} --\n", inst.particles.len());
            for (j, p) in inst.particles.iter().enumerate() {
                str += &format!("\t{}\n", j);
                str += &format!(
                    "curr={}\n\
                     prev={}\n\
                     disp={}\n",
                    p.position,
                    p.last,
                    p.displacement(),
                );
            }

            /* TODO: rods, joint data */
        }

        /* TODO: joints, planes */

        str += &format!("\n # # # nmg/softbody/config # # # \n\n");
        eprint!("{}", str);
    }

    #[allow(unused_variables)]
    pub fn draw_all(&self, debug: &mut debug::Handler) {
        #[cfg(debug_assertions)] {
            self.draw_all_instances(debug);
            self.draw_all_joints(debug);
        }
    }

    #[allow(unused_variables)]
    pub fn draw_entity(
        &self,
        entity: entity::Handle,
        draw_normals: bool,
        draw_endpoints: bool,
        debug: &mut debug::Handler,
    ) {
        #[cfg(debug_assertions)] {
            let i = entity.get_index() as usize;
            self.draw_instance(i, draw_normals, draw_endpoints, debug);
        }
    }

    #[allow(unused_variables)]
    pub fn draw_all_instances(&self, debug: &mut debug::Handler) {
        #[cfg(debug_assertions)] {
            for i in 0..self.instances.len() {
                self.draw_instance(i, false, true, debug);
            }
        }
    }

    #[allow(unused_variables)]
    fn draw_instance(
        &self,
        index: usize,
        draw_normals: bool,
        draw_endpoints: bool,
        debug: &mut debug::Handler,
    ) {
        #[cfg(debug_assertions)] {
            debug_assert!(index < self.instances.len());

            if let Some(ref instance) = self.instances[index] {
                if draw_endpoints && instance.end_offset > 0.0 {
                    let center = instance.center();
                    let orientation = instance.matched_orient(center).to_mat();

                    debug.add_cross(
                        instance.start(center, orientation),
                        0.33,
                        graphics::Color::yellow(),
                    );

                    debug.add_cross(
                        instance.end(center, orientation),
                        0.33,
                        graphics::Color::blue(),
                    );
                }

                if draw_normals {
                    let normals = Instance::compute_normals(
                        &instance.particles,
                        &instance.model.indices,
                        instance.model.duplicates.len(),
                    );

                    for (position, normal) in instance.model.indices.iter()
                        .map(|index| {
                            (
                                instance.particles[*index].position,
                                normals[*index],
                            )
                        })
                    {
                        debug.add_ray(
                            position,
                            normal * 0.5,
                            graphics::Color::green(),
                        );
                    }
                }

                // Draw instance bindings
                for rod in &instance.rods {
                    let left = instance.particles[rod.left].position;
                    let right = instance.particles[rod.right].position;

                    let lerp = (rod.length - left.dist(right)).abs()
                        / (0.1 * rod.length);

                    debug.add_line(
                        alg::Line::new(left, right),
                        graphics::Color::lerp(
                            graphics::Color::green(),
                            graphics::Color::red(),
                            lerp,
                        ),
                    );
                }

                if instance.match_shape {
                    let mut draw = |triangle: &[usize], a: usize, b: usize| {
                        debug.add_line(
                            alg::Line::new(
                                instance.particles[triangle[a]].position,
                                instance.particles[triangle[b]].position,
                            ),
                            graphics::Color::gray(),
                        );
                    };

                    // Duplicates will draw more lines than necessary
                    for triangle in instance.model.indices.chunks(3) {
                        draw(triangle, 0, 1);
                        draw(triangle, 1, 2);
                        draw(triangle, 2, 0);
                    }
                }
            }
        }
    }

    #[allow(unused_variables)]
    pub fn draw_joint_parent(
        &self,
        entity: entity::Handle,
        draw_cone: bool,
        debug: &mut debug::Handler,
    ) {
        #[cfg(debug_assertions)] {
            let i = entity.get_index() as usize;
            match self.joints.get(&i) {
                Some(joints) => self.draw_parent(i, joints, draw_cone, debug),
                None => panic!(
                    "Softbody instance for entity {} is not a joint parent.",
                    entity,
                ),
            }
        }
    }

    #[allow(unused_variables)]
    pub fn draw_all_joints(&self, debug: &mut debug::Handler) {
        #[cfg(debug_assertions)] {
            // Draw joints for every parent
            for (parent_index, joints) in &self.joints {
                self.draw_parent(*parent_index, joints, true, debug);
            }
        }
    }

    #[allow(unused_variables)]
    fn draw_parent(
        &self,
        index: usize,
        joints: &[Joint],
        draw_cone: bool,
        debug: &mut debug::Handler,
    ) {
        #[cfg(debug_assertions)] {
            let parent = self.instances[index]
                .as_ref().unwrap();

            // Draw all joints for this parent
            for joint in joints {
                let center = parent.center();
                let orientation = parent.frame_orient_conj.conjugate().to_mat();
                let point = parent.extend(joint.offset, orientation, center);
                let joint_orientation = orientation * joint.transform.to_mat();

                if draw_cone {
                    // Compute endpoints
                    // (duplicate code from ReachCone new method)
                    let half_pi = 0.5 * std::f32::consts::PI;
                    let (lower, right, upper, left) = {
                        let x_min = joint.y_limit.min / half_pi;
                        let x_max = joint.y_limit.max / half_pi;
                        let y_min = joint.x_limit.min / half_pi;
                        let y_max = joint.x_limit.max / half_pi;

                        let x_min_inv = 1.0 - x_min.abs();
                        let x_max_inv = 1.0 - x_max.abs();
                        let y_min_inv = 1.0 - y_min.abs();
                        let y_max_inv = 1.0 - y_max.abs();

                        (
                            alg::Vec3::new(0.0, y_min, y_min_inv).norm(),
                            alg::Vec3::new(x_max, 0.0, x_max_inv).norm(),
                            alg::Vec3::new(0.0, y_max, y_max_inv).norm(),
                            alg::Vec3::new(x_min, 0.0, x_min_inv).norm(),
                        )
                    };

                    /* Cone rays */

                    let lower_ray = joint_orientation * lower * 0.5;
                    let right_ray = joint_orientation * right * 0.5;
                    let upper_ray = joint_orientation * upper * 0.5;
                    let left_ray = joint_orientation * left * 0.5;

                    debug.add_ray(point, lower_ray, graphics::Color::green());
                    debug.add_ray(point, right_ray, graphics::Color::red());
                    debug.add_ray(point, upper_ray, graphics::Color::green());
                    debug.add_ray(point, left_ray, graphics::Color::red());

                    /* Edge lines */

                    debug.add_line(
                        alg::Line::new(point + lower_ray, point + right_ray),
                        graphics::Color::yellow(),
                    );

                    debug.add_line(
                        alg::Line::new(point + right_ray, point + upper_ray),
                        graphics::Color::yellow(),
                    );

                    debug.add_line(
                        alg::Line::new(point + upper_ray, point + left_ray),
                        graphics::Color::yellow(),
                    );

                    debug.add_line(
                        alg::Line::new(point + left_ray, point + lower_ray),
                        graphics::Color::yellow(),
                    );

                    /* Child pointer */

                    let child = self.instances[joint.child].as_ref().unwrap();
                    let child_center = child.center();
                    let child_orient = child.frame_orient_conj.conjugate()
                        .to_mat();

                    // Forward pointer
                    debug.add_ray(
                        point,
                        child_orient * alg::Vec3::fwd() * 0.5,
                        graphics::Color::white(),
                    );

                    // Twist joint limits
                    if joint.z_limit.min != 0.0 || joint.z_limit.max != 0.0 {
                        // Twist pointer
                        debug.add_ray(
                            point,
                            child_orient * alg::Vec3::up() * 0.25,
                            graphics::Color::white(),
                        );

                        /* Limits */

                        let min = alg::Vec3::new(
                            joint.z_limit.min.sin(),
                            joint.z_limit.min.cos(),
                            0.0,
                        );

                        let max = alg::Vec3::new(
                            joint.z_limit.max.sin(),
                            joint.z_limit.max.cos(),
                            0.0,
                        );

                        // Reuse computations from joint limit code
                        let joint_inv = joint_orientation.transpose();
                        let local_child = (joint_inv * child_orient).to_quat();

                        let simple = alg::Quat::simple(
                            alg::Vec3::fwd(),
                            local_child * alg::Vec3::fwd(),
                        );

                        let transform = joint_orientation.to_quat() * simple;

                        debug.add_ray(
                            point,
                            transform * min * 0.25,
                            graphics::Color::cyan(),
                        );

                        debug.add_ray(
                            point,
                            transform * max * 0.25,
                            graphics::Color::cyan(),
                        );
                    }
                } else {
                    let fwd = joint_orientation * alg::Vec3::fwd();
                    let up = joint_orientation * alg::Vec3::up();

                    // Draw joint endpoint
                    debug.add_local_axes(point, fwd, up, 1.0, 1.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate rand;
    use components::*;
    use alg::*;

    const DEFAULT_FIXED_DT: f32 = ::FIXED_DT;
    const KE_THRESH: f32 = std::f32::EPSILON * 2.0;
    const KE_MARGIN: f32 = 1e-4;

    struct Empty { }
    impl softbody::Iterate for Empty { }

    struct Context {
        fixed_dt: f32,
        entities: entity::Manager,
        transforms: transform::Manager,
        softbodies: softbody::Manager,
        empty: Empty,
    }

    impl Context {
        fn new(fixed_dt: f32) -> Context {
            Context {
                fixed_dt: fixed_dt,
                entities: entity::Manager::new(1),
                transforms: transform::Manager::new(1),
                softbodies: softbody::Manager::new(fixed_dt, 1, 1, 1),
                empty: Empty { },
            }
        }

        fn particle(fixed_dt: f32) -> (Context, entity::Handle) {
            let mut ctx = Context::new(fixed_dt);
            let e = ctx.entities.add();
            ctx.transforms.register(e);
            ctx.softbodies.register(e);

            ctx.softbodies.build_instance()
                .particles(&[Vec3::zero()])
                .indices(&[0])
                .mass(1.0)
                .for_entity(e);

            (ctx, e)
        }

        fn single(fixed_dt: f32) -> (Context, entity::Handle) {
            let mut ctx = Context::new(fixed_dt);
            let e = ctx.entities.add();
            ctx.transforms.register(e);
            ctx.softbodies.register(e);

            ctx.softbodies.build_instance()
                .make_box_limb(Vec3::one())
                .mass(1.0)
                .for_entity(e);

            (ctx, e)
        }

        fn init_instance<F>(&mut self, e: entity::Handle, f: F)
            where F: Copy, F: FnOnce(Vec3) -> (Vec3, Vec3)
        {
            let instance = self.softbodies.get_mut_instance(e);
            let fixed_dt = self.fixed_dt;
            instance.particles.iter_mut().for_each(
                |p| {
                    let (to, vel) = f(p.position);
                    p.position = to;
                    p.last = to - vel * fixed_dt;
                }
            );
        }

        #[cfg(debug_assertions)]
        fn log_instances(&self) {
            for (i, opt) in self.softbodies.instances.iter().enumerate() {
                let instance = match opt {
                    Some(ref instance) => instance,
                    None => continue,
                };

                println!("instance={}", i);
                instance.debug.log_frame();
            }
        }

        fn cycle(&mut self) {
            self.softbodies.simulate(&mut self.empty, &mut self.transforms);
            #[cfg(debug_assertions)] self.log_instances();
        }

        fn spin_unchecked(&mut self, duration: f32) {
            let count = (duration / self.fixed_dt) as u32;
            for _ in 0..count { self.cycle(); }
        }

        fn spin(&mut self, duration: f32) -> (f32, f32) {
            let count = (duration / self.fixed_dt) as u32;
            let count = if count > 0 { count - 1 } else { count };

            println!("tick={}", 0);
            self.cycle();
            let initial = self.softbodies.energy();
            let mut last = initial;
            println!("\tinitial energy={:.4}J (1 tick warmup)", initial);
            println!("\tfixed_dt={}, count=1+{}", self.fixed_dt, count);

            for i in 0..count {
                let tick = i + 1;

                println!("tick={}", tick);
                self.cycle();
                let ke = self.softbodies.energy();
                println!("\tenergy={:.4}J", ke);

                if last < KE_THRESH {
                    if ke > KE_THRESH {
                        eprintln!(" initial value: {:.2}J", initial);
                        eprintln!("previous value: {}J < {}J", last, KE_THRESH);

                        eprintln!(
                            " trigger value: {}J @ tick {} > {}J",
                            ke,
                            tick,
                            KE_THRESH
                        );

                        assert!(ke < KE_THRESH);
                    }
                // TODO: add warning for strict gt
                } else if ke - last > KE_MARGIN {
                    eprintln!(" initial value: {:.2}J", initial);
                    eprintln!("previous value: {}J > {}J", last, KE_THRESH);

                    eprintln!(
                        " trigger value: {:.2}J @ tick {} > prev.",
                        ke,
                        tick,
                    );

                    assert!(ke - last > KE_MARGIN);
                }

                last = ke;
            }

            (initial, last)
        }

        fn burndown(&mut self, duration: f32) {
            let (initial, end) = self.spin(duration);

            if end > KE_THRESH {
                eprintln!("initial value: {:.2}J", initial);

                eprintln!(
                    "  final value: {}J ({}s) > {}J",
                    end,
                    duration,
                    KE_THRESH
                );

                assert!(end < KE_THRESH);
            }
        }
    }

    fn rands() -> f32 {
        let rand::Closed01(result) = rand::random::<rand::Closed01<f32>>();
        result * 2f32 - 1f32
    }

    #[test]
    fn energy() {
        let (mut ctx, e) = Context::single(1.0);
        ctx.init_instance(e, |pos| (pos, Vec3::up()));
        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.cycle();

        assert_approx_eq!(ctx.softbodies.energy(), 0.5, 1.0);
    }

    #[test]
    fn baseline() {
        let (mut ctx, e) = Context::single(DEFAULT_FIXED_DT);
        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.burndown(4.0);

        let pos = ctx.transforms.get_position(e);
        let orient = ctx.transforms.get_orientation(e);
        assert_approx_eq_vec3!(pos, Vec3::zero(), 1.0);
        assert_approx_eq_quat!(orient, Quat::id(), 1.0);
    }

    #[test]
    fn drag_none() {
        {   /* Pull vertices */

            let (mut ctx, e) = Context::single(1.0);
            ctx.init_instance(e, |pos| (pos, Vec3::up()));
            ctx.softbodies.iterations = 1;
            ctx.softbodies.set_gravity(Vec3::zero());
            ctx.softbodies.set_drag(0.0);
            ctx.spin(2.0);

            let pos = ctx.transforms.get_position(e);
            let orient = ctx.transforms.get_orientation(e);

            assert_approx_eq_vec3!(pos, Vec3::up() * 2.0, 1.0);
            assert_approx_eq_quat!(orient, Quat::id(), 1.0);
        }
        {   /* Add force */

            let (mut ctx, e) = Context::single(1.0);
            ctx.softbodies.iterations = 1;
            ctx.softbodies.set_gravity(Vec3::up());
            ctx.softbodies.set_drag(0.0);
            ctx.spin_unchecked(2.0);

            let pos = ctx.transforms.get_position(e);
            let orient = ctx.transforms.get_orientation(e);

            assert_approx_eq_vec3!(pos, Vec3::up() * 3.0, 1.0);
            assert_approx_eq_quat!(orient, Quat::id(), 1.0);
        }
    }

    #[test]
    fn drag_full() {
        let (mut ctx, e) = Context::single(DEFAULT_FIXED_DT);
        ctx.init_instance(e, |pos| (pos, Vec3::one()));
        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::up() * 128.0);
        ctx.softbodies.set_drag(1.0);
        ctx.burndown(2.0);

        let pos = ctx.transforms.get_position(e);
        let orient = ctx.transforms.get_orientation(e);
        assert_approx_eq_vec3!(pos, Vec3::zero(), 1.0);
        assert_approx_eq_quat!(orient, Quat::id(), 1.0);
    }

    #[test]
    fn shape_scale() {
        for i in 0..256 {
            let scale = i as f32 * 0.5;
            println!("i={}, scale={}", i, scale);

            // Implicitly tests singular shapes
            let (mut ctx, e) = Context::single(1.0 / 30.0);
            {
                let inst = ctx.softbodies.get_mut_instance(e);
                inst.rigidity = 1.0;
                inst.particles.iter_mut()
                    .for_each(|p| p.position = p.position * scale);

                let orient = inst.matched_orient(inst.center());
                assert_approx_eq_quat!(orient, Quat::id(), 1.0);
            }

            ctx.softbodies.iterations = 1;
            ctx.softbodies.set_gravity(Vec3::zero());
            ctx.softbodies.set_drag(0.0);
            ctx.burndown(1.0);
            {
                let orient = ctx.transforms.get_orientation(e);
                assert_approx_eq_quat!(orient, Quat::id(), 1.0);

                let inst = ctx.softbodies.get_mut_instance(e);
                inst.particles.iter()
                    .for_each(
                        |p| {
                            assert_approx_eq!(
                                p.position.mag_squared(),
                                0.75, // Distance from unit cube to center
                                1.0
                            );
                        }
                    );
            }
        }
    }

    #[test]
    fn shape_rot() {
        let substeps = 2;
        let total = 90.0 * substeps as f32 * 2.0;
        let regress_limit = 31;

        for i in 0..(90 * substeps) {
            let rad = std::f32::consts::PI * (i as f32 / total);
            let deg = i as f32 / substeps as f32;
            println!("i={}, a : {} deg, {} rad", i, deg, rad);

            let rot = Quat::axis_angle(Vec3::up(), rad);
            println!("rot={}", rot);

            let (mut ctx, e) = Context::single(1.0 / 30.0);
            {
                let inst = ctx.softbodies.get_mut_instance(e);
                let center = inst.center();
                inst.rotate_around(rot, center);
                inst.rigidity = 1.0;

                let mut orient = inst.matched_orient(inst.center());
                if orient.dot(rot) < 0.0 { orient = orient.neg(); }
                assert_approx_eq_quat!(orient, rot, 512.0); // See EXTRACT_ITER
            }

            ctx.softbodies.iterations = 1;
            ctx.softbodies.set_gravity(Vec3::zero());
            ctx.softbodies.set_drag(0.0);
            ctx.cycle();

            let orient = ctx.transforms.get_orientation(e);
            let actual = orient.abs_angle();
            let err = (actual - 2.0 * rad).abs()
                * 180.0 / std::f32::consts::PI;

            println!(
                "result={:.3}deg err={:.3}deg\n",
                actual * 180.0 / std::f32::consts::PI,
                err
            );

            if err > 1.0 {
                assert_eq!(i, regress_limit);
                let rps = deg / (360.0 * ctx.fixed_dt);
                let rpm = 60.0 * rps;

                println!(
                    "regression limit reached: rpm={:.1} rps={:.1}",
                    rpm,
                    rps
                );

                break;
            }

         /* assert_approx_eq_quat!(orient, new, 1.0); */
            let pos = ctx.transforms.get_position(e);
            assert_approx_eq_vec3!(pos, Vec3::zero(), 4.0);
        }
    }

    #[test]
    fn shape_inv() {
        let (mut ctx, e) = Context::single(1.0 / 30.0);
        ctx.init_instance(
            e,
            |mut pos| {
                pos.y *= -1.0;
                (pos, Vec3::zero())
            }
        );

        {
            let inst = ctx.softbodies.get_mut_instance(e);
            inst.rigidity = 1.0;

            let center = inst.center();
            let transform = inst.matched_transform(center);
            assert!(transform.det() < 0f32);

            let orient = inst.matched_orient(center);
            assert_approx_eq_quat!(orient, Quat::id(), 1.0);
        }

        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.burndown(2.0);

        let inst = ctx.softbodies.get_instance(e);
        let orient = inst.matched_orient(inst.center());
        assert_approx_eq_quat!(orient, Quat::id(), 1.0);
        inst.particles.iter()
            .for_each(
                |p| {
                    assert_approx_eq!(
                        p.position.mag_squared(),
                        0.75, // Distance from unit cube to center
                        1.0
                    );
                }
            );
    }

    #[test]
    fn shape_planar() {
        let (mut ctx, e) = Context::single(1.0 / 30.0);
        ctx.init_instance(
            e,
            |mut pos| {
                pos.y *= 0f32;
                (pos, Vec3::zero())
            }
        );

        {
            let inst = ctx.softbodies.get_mut_instance(e);
            inst.rigidity = 1.0;

            let center = inst.center();
            let transform = inst.matched_transform(center);
            assert!(transform.det() == 0f32);

            let orient = inst.matched_orient(center);
            assert_approx_eq_quat!(orient, Quat::id(), 1.0);
        }

        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.burndown(2.0);

        let inst = ctx.softbodies.get_instance(e);
        let orient = inst.matched_orient(inst.center());
        assert_approx_eq_quat!(orient, Quat::id(), 1.0);
        inst.particles.iter()
            .for_each(
                |p| {
                    assert_approx_eq!(
                        p.position.mag_squared(),
                        0.75, // Distance from unit cube to center
                        1.0
                    );
                }
            );
    }

    #[test]
    fn shape_rand() {
        for i in 0..128 {
            let (mut ctx, e) = Context::single(1.0 / 30.0);
            ctx.init_instance(
                e, |pos| {
                    let off = Vec3::new(rands(), rands(), rands());
                    (pos + off, Vec3::zero())
                }
            );

            {
                let instance = ctx.softbodies.get_mut_instance(e);
                instance.rigidity = 1.0;
            }

            println!("config={}", i + 1);
            ctx.softbodies.iterations = 1;
            ctx.softbodies.set_gravity(Vec3::zero());
            ctx.softbodies.set_drag(0.1); // Speed up burndown
            #[cfg(debug_assertions)] { ctx.softbodies.dump_config(); }
            ctx.burndown(4.0);
            println!("\n= = =\n");
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    fn shape_err_baseline() {
        for i in 0..32 {
            let (mut ctx, e) = Context::single(1.0 / 30.0);
            ctx.init_instance(
                e, |pos| {
                    let off = Vec3::new(rands(), rands(), rands());
                    (pos + off, Vec3::zero())
                }
            );

            {
                let instance = ctx.softbodies.get_mut_instance(e);
                instance.rigidity = 1.0;
            }

            println!("config={}", i + 1);
            ctx.softbodies.iterations = 2; // Add runnoff iteration
            ctx.softbodies.set_gravity(Vec3::zero());
            ctx.softbodies.set_drag(0.0);
            ctx.cycle(); // Should stabilize in one tick

            {
                let instance = ctx.softbodies.get_instance(e);
                // Dependent on EXTRACT_ITER
                assert_approx_eq!(instance.debug.err[1].shape, 0.0, 2.0);
            }

            println!("\n= = =\n");
        }
    }

    #[test]
    fn shape_regression_0() {
        let (mut ctx, e) = Context::single(1.0 / 30.0);

        {
            let instance = ctx.softbodies.get_mut_instance(e);
            instance.rigidity = 1.0;

            instance.particles[0].position = Vec3::new(
                -1.24987769, 0.68691027, -0.01353586,
            );

            instance.particles[1].position = Vec3::new(
                -0.23069978, 0.62659800, 0.45957029,
            );

            instance.particles[2].position = Vec3::new(
                1.32380474, -0.84423178, -0.0874966,
            );

            instance.particles[3].position = Vec3::new(
                0.45426810, 0.32374418, -1.26090717,
            );

            instance.particles[4].position = Vec3::new(
                -1.26433945, 0.39075643, 0.22179824,
            );

            instance.particles[5].position = Vec3::new(
                0.95629561, -0.39130545, 0.17226774,
            );

            instance.particles[6].position = Vec3::new(
                1.45903981, -0.81027216, -0.00294423,
            );

            instance.particles[7].position = Vec3::new(
                -0.99564856, -0.07726038, 0.06232053,
            );
        }

        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.1);
        ctx.burndown(8.0);
    }

    #[test]
    fn plane_particle_baseline()
    {
        let (mut ctx, e) = Context::particle(1.0);
        ctx.init_instance(e, |pos| (pos, -Vec3::up()));
        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.softbodies.set_friction(0.0);
        ctx.softbodies.set_bounce(2.0);

        ctx.softbodies.add_planes(&[Plane::new(Vec3::up(), 0.0)]);

        ctx.spin(1.0);
        let pos = ctx.transforms.get_position(e);
        let vel = {
            let instance = ctx.softbodies.get_instance(e);
            instance.velocity()
        };

        assert_approx_eq_vec3!(pos, Vec3::up(), 1.0);
        assert_approx_eq_vec3!(vel, Vec3::up(), 1.0);
    }

    #[test]
    fn plane_particle_null()
    {
        let (mut ctx, e) = Context::particle(1.0);
        ctx.init_instance(e, |pos| (pos, -Vec3::up()));
        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.softbodies.set_friction(0.0);
        ctx.softbodies.set_bounce(1.0);

        ctx.softbodies.add_planes(&[Plane::new(Vec3::up(), 0.0)]);

        ctx.spin(1.0);
        let pos = ctx.transforms.get_position(e);
        let vel = {
            let instance = ctx.softbodies.get_instance(e);
            instance.velocity()
        };

        assert_approx_eq_vec3!(pos, Vec3::zero(), 1.0);
        assert_approx_eq_vec3!(vel, Vec3::zero(), 1.0);
    }

    #[test]
    fn plane_particle_reflect()
    {
        let (mut ctx, e) = Context::particle(1.0);
        ctx.init_instance(
            e, |pos| (
                pos + Vec3::zero(),
                      Vec3::new(-1.0, -1.0, 0.0)
            )
        );

        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.softbodies.set_friction(0.0);
        ctx.softbodies.set_bounce(2.0);

        ctx.softbodies.add_planes(&[Plane::new(Vec3::up(), 0.0)]);

        ctx.spin(1.0);
        let pos = ctx.transforms.get_position(e);
        let vel = {
            let instance = ctx.softbodies.get_instance(e);
            instance.velocity()
        };

        assert_approx_eq_vec3!(pos, Vec3::new(-1.0, 1.0, 0.0), 1.0);
        assert_approx_eq_vec3!(vel, Vec3::new(-1.0, 1.0, 0.0), 1.0);
    }

    #[test]
    fn plane_particle_reflect_damp()
    {
        let (mut ctx, e) = Context::particle(1.0);
        ctx.init_instance(
            e, |pos| (
                pos + Vec3::zero(),
                      Vec3::new(-1.0, -1.0, 0.0)
            )
        );

        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.softbodies.set_friction(0.0);
        ctx.softbodies.set_bounce(1.5);

        ctx.softbodies.add_planes(&[Plane::new(Vec3::up(), 0.0)]);

        ctx.spin(1.0);
        let pos = ctx.transforms.get_position(e);
        let vel = {
            let instance = ctx.softbodies.get_instance(e);
            instance.velocity()
        };

        assert_approx_eq_vec3!(pos, Vec3::new(-1.0, 0.5, 0.0), 1.0);
        assert_approx_eq_vec3!(vel, Vec3::new(-1.0, 0.5, 0.0), 1.0);
    }

    #[test]
    fn plane_particle_friction_full()
    {
        let (mut ctx, e) = Context::particle(1.0);
        ctx.init_instance(
            e, |pos| (
                pos + Vec3::zero(),
                      Vec3::new(-1.0, -1.0, 0.0)
            )
        );

        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.softbodies.set_friction(1.0);
        ctx.softbodies.set_bounce(2.0);

        ctx.softbodies.add_planes(&[Plane::new(Vec3::up(), 0.0)]);

        ctx.spin(1.0);
        let pos = ctx.transforms.get_position(e);
        let vel = {
            let instance = ctx.softbodies.get_instance(e);
            instance.velocity()
        };

        assert_approx_eq_vec3!(pos, Vec3::new(0.0, 1.0, 0.0), 1.0);
        assert_approx_eq_vec3!(vel, Vec3::new(0.0, 1.0, 0.0), 1.0);
    }

    #[test]
    fn plane_particle_friction_half()
    {
        let (mut ctx, e) = Context::particle(1.0);
        ctx.init_instance(
            e, |pos| (
                pos + Vec3::zero(),
                      Vec3::new(-1.0, -1.0, 0.0)
            )
        );

        ctx.softbodies.iterations = 1;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.softbodies.set_friction(0.5);
        ctx.softbodies.set_bounce(2.0);

        ctx.softbodies.add_planes(&[Plane::new(Vec3::up(), 0.0)]);

        ctx.spin(1.0);
        let pos = ctx.transforms.get_position(e);
        let vel = {
            let instance = ctx.softbodies.get_instance(e);
            instance.velocity()
        };

        assert_approx_eq_vec3!(pos, Vec3::new(-0.5, 1.0, 0.0), 1.0);
        assert_approx_eq_vec3!(vel, Vec3::new(-0.5, 1.0, 0.0), 1.0);
    }

    #[test]
    fn plane_particle_fract()
    {
        for i in 2..9 {
            let v = (i as f32) * 0.1;

            let (mut ctx, e) = Context::particle(1.0);
            ctx.init_instance(e, |_| (Vec3::right(), -Vec3::right() * v));

            ctx.softbodies.iterations = 1;
            ctx.softbodies.set_gravity(Vec3::zero());
            ctx.softbodies.set_drag(0.0);
            ctx.softbodies.set_friction(0.0);
            ctx.softbodies.set_bounce(2.0);

            ctx.softbodies.add_planes(&[Plane::new(Vec3::right(), 0.0)]);

            let steps = 2.0 * 1.0 / v;

            ctx.spin(steps.floor());
            let prev = ctx.transforms.get_position(e);
            ctx.spin(1.0);
            let pos = ctx.transforms.get_position(e);

            let fract = steps % 1.0;
            let interp = prev.lerp(pos, fract).x;

            println!("v={}, steps={}+1", v, steps.floor());
            assert_approx_eq!(interp, 1.0, 400.0);
        }
    }

    #[test]
    fn plane_shape_damp() {
        for i in 0..5 {
            let v = i as f32 * 0.25;

            let (mut ctx, e) = Context::single(1.0);
            ctx.init_instance(e, |pos| (pos + Vec3::up() * 0.5, -Vec3::up() * v));
            ctx.softbodies.add_planes(&[Plane::new(Vec3::up(), 0.0)]);
            ctx.softbodies.iterations = 2;
            ctx.softbodies.set_gravity(Vec3::zero());
            ctx.softbodies.set_drag(0.0);
            ctx.softbodies.set_friction(0.0);
            ctx.softbodies.set_bounce(2.0);

            // The initial velocity is cancelled out for one tick
            // due to the constraint order;
            // afterwards we regain the original energy.
            ctx.spin(2.0);

            let pos = ctx.transforms.get_position(e);
            let vel = {
                let instance = ctx.softbodies.get_instance(e);
                instance.velocity()
            };

            assert_approx_eq_vec3!(pos, Vec3::up() * 0.5, 1.0);
            assert_approx_eq_vec3!(vel, Vec3::up() * v, 1.0);
        }
    }

    #[test]
    fn plane_shape_full() {
        let vy = 1.0 + 1e-4;

        let (mut ctx, e) = Context::single(1.0);
        ctx.init_instance(e, |pos| (pos + Vec3::up() * 0.5, -Vec3::up() * vy));
        ctx.softbodies.add_planes(&[Plane::new(Vec3::up(), 0.0)]);
        ctx.softbodies.iterations = 2;
        ctx.softbodies.set_gravity(Vec3::zero());
        ctx.softbodies.set_drag(0.0);
        ctx.softbodies.set_friction(0.0);
        ctx.softbodies.set_bounce(2.0);

        ctx.spin_unchecked(1.0);
        let pos = ctx.transforms.get_position(e);
        let vel = {
            let instance = ctx.softbodies.get_instance(e);
            instance.velocity()
        };

        // The mesh completely penetrates the plane
        // and a full bounce is observed.
        assert_approx_eq_vec3!(pos, Vec3::up() * (0.5 + 1e-4), 1.0);
        assert_approx_eq_vec3!(vel, Vec3::up() * vy, 1.0);
    }
}
