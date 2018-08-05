extern crate fnv;

use std;
use alg;
use entity;
use render;
use components;
use debug;

#[cfg(debug_assertions)] use graphics;

use ::FIXED_DT; // Import from lib
use components::transform;

/// Default instance mass
pub const INST_DEFAULT_MASS: f32 = 1.0;

/// Default instance rigidity
pub const INST_DEFAULT_RIGID: f32 = 1.0;

/// Default system (softbody manager) bounce
pub const MNGR_DEFAULT_BOUNCE: f32 = 2.0;

/// Default system (softbody manager) friction
pub const MNGR_DEFAULT_FRICTION: f32 = 0.02;

// Constraint solver iterations
const ITERATIONS: usize = 10;

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
const JOINT_CONTAINS_BIAS: f32 = 8.0;

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
    fn iterate(&mut self, fixed_delta: f32, &mut Manager) { }
}

struct Particle {
    position: alg::Vec3,
    last: alg::Vec3,
    displacement: alg::Vec3,
}

impl Particle {
    fn new(position: alg::Vec3) -> Particle {
        Particle {
            position,
            last: position,
            displacement: alg::Vec3::zero(),
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

#[derive(Clone, Copy)]
struct Range {
    min: f32,
    max: f32,
}

#[derive(Clone, Copy)]
struct ReachPlane {
    normal: alg::Vec3,
}

// Specialized plane struct for joint constraints
impl ReachPlane {
    fn new(left: alg::Vec3, right: alg::Vec3) -> ReachPlane {
        ReachPlane {
            normal: left.cross(right),
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
            let x_min = y_limit.min / 90f32.to_radians();
            let x_max = y_limit.max / 90f32.to_radians();
            let y_min = x_limit.min / 90f32.to_radians();
            let y_max = x_limit.max / 90f32.to_radians();

            (
                alg::Vec3::new(0.0, y_min, 1.0 - y_min.abs()).norm(),
                alg::Vec3::new(x_max, 0.0, 1.0 - x_max.abs()).norm(),
                alg::Vec3::new(0.0, y_max, 1.0 - y_max.abs()).norm(),
                alg::Vec3::new(x_min, 0.0, 1.0 - x_min.abs()).norm(),
            )
        };

        // Build planes
        let lower_left = ReachPlane::new(left, lower);
        let lower_right = ReachPlane::new(lower, right);
        let upper_right = ReachPlane::new(right, upper);
        let upper_left = ReachPlane::new(upper, left);

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
    fwd: alg::Vec3,
    up: alg::Vec3,
    offset: alg::Vec3,
    x_limit: Option<Range>,
    y_limit: Option<Range>,
    z_limit: Option<Range>,
}

impl<'a> JointBuilder<'a> {
    pub fn new(manager: &'a mut Manager) -> JointBuilder {
        JointBuilder {
            manager,
            parent: None,
            fwd: alg::Vec3::fwd(),
            up: alg::Vec3::up(),
            offset: alg::Vec3::zero(),
            x_limit: None,
            y_limit: None,
            z_limit: None,
        }
    }

    pub fn with_parent(
        &mut self,
        parent: entity::Handle,
    ) -> &'a mut JointBuilder {
        self.parent = Some(parent);
        self
    }

    /// Joint x-axis limit range, in degrees
    pub fn x(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        let (min, max) = (min.to_radians(), max.to_radians());
        self.x_limit = Some(Range { min, max });
        self
    }

    /// Joint y-axis limit range, in degrees
    pub fn y(&mut self, min: f32, max: f32) -> &'a mut JointBuilder {
        let (min, max) = (min.to_radians(), max.to_radians());
        self.y_limit = Some(Range { min, max });
        self
    }

    /// Joint z-axis limit range, in degrees
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

    /* Joint transform */

    pub fn fwd(&mut self, fwd: alg::Vec3) -> &'a mut JointBuilder {
        self.fwd = fwd;
        self
    }

    pub fn up(&mut self, up: alg::Vec3) -> &'a mut JointBuilder {
        self.up = up;
        self
    }

    pub fn offset(&mut self, offset: alg::Vec3) -> &'a mut JointBuilder {
        self.offset = offset;
        self
    }

    /// Finalize
    pub fn for_child(&mut self, child: entity::Handle) {
        let parent = if let Some(parent) = self.parent {
            parent
        } else {
            panic!("No parent specified to joint builder");
        };

        debug_assert!(parent != child);
        debug_assert!(self.x_limit.is_some());
        debug_assert!(self.y_limit.is_some());
        debug_assert!(self.z_limit.is_some());

        let transform = alg::Quat::from_vecs(self.fwd, self.up);

        self.manager.add_joint(
            parent,
            child,
            transform,
            self.offset,
            (
                self.x_limit.unwrap(),
                self.y_limit.unwrap(),
                self.z_limit.unwrap()
            ),
        );
    }
}

pub struct Instance {
    particles: Vec<Particle>,
    rods: Vec<Rod>,
    match_shape: bool,

    force: alg::Vec3,
    accel_dt: alg::Vec3, // Cached value, dependent on force

    /* Updated per-frame */

    frame_position: alg::Vec3,
    frame_orientation: alg::Quat,

    /* "Constants" */

    mass: f32,
    inv_pt_mass: f32, // Inverse mass per particle
    perfect_model: Vec<alg::Vec3>, // Vertices reference
    perfect_com: alg::Vec3, // Center of mass reference
    model: Option<Vec<alg::Vec3>>, // Optional override
    triangles: Vec<usize>, // Indices reference, for normals
    normals: Vec<alg::Vec3>, // Normals reference

    // Range 0 - 0.5; "Rigid" = 0.5
    // Lower values produce springier meshes
    // A value of zero nullifies all rods in the instance
    rigidity: f32,
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
        initial_accel: alg::Vec3,
    ) -> Instance {
        debug_assert!(mass > 0.0);
        debug_assert!(rigidity > 0.0 && rigidity <= 0.5);

        /* Initialize particles and base comparison model */

        let (particles, perfect_model) = {
            let mut particles = Vec::with_capacity(points.len());
            let mut perfect_model = Vec::with_capacity(points.len());

            for point in points {
                particles.push(Particle::new(*point));
                perfect_model.push(*point);
            }

            (particles, perfect_model)
        };

        let center = |model: &[alg::Vec3]| model.iter().fold(
            alg::Vec3::zero(),
            |sum, position| sum + *position
        ) / model.len() as f32;

        let perfect_com = center(&perfect_model);

        #[cfg(debug_assertions)] {
            if let Some(model) = model_override {
                assert_eq!(center(model), perfect_com);
            }
        }

        // Compute base comparison normals for instance
        debug_assert!(indices.len() % 3 == 0);
        let normals = Instance::compute_normals(&particles, &indices);

        // Initialize rods
        let mut rods = Vec::with_capacity(bindings.len());
        for binding in bindings {
            rods.push(Rod::new(binding.0, binding.1, &particles));
        }

        debug_assert!(points.len() == particles.len());
        debug_assert!(particles.len() == perfect_model.len());

        Instance {
            particles,
            rods,
            match_shape,

            force: alg::Vec3::zero(),
            accel_dt: initial_accel * FIXED_DT * FIXED_DT,

            frame_position: alg::Vec3::zero(),
            frame_orientation: alg::Quat::id(),

            mass,
            inv_pt_mass: 1.0 / (mass / points.len() as f32),
            rigidity,
            perfect_model,
            perfect_com,
            model: model_override.map(|model| model.to_vec()),
            triangles: indices.to_vec(),
            normals,
        }
    }

    /* Calculate normals for implicit softbody mesh,
     * useful for blending rendered mesh normals
     * and/or determining softbody topology.
     */
    fn compute_normals(
        particles: &[Particle],
        triangles: &[usize],
    ) -> Vec<alg::Vec3> {
        let mut result = vec![alg::Vec3::zero(); particles.len()];

        for indices in triangles.chunks(3) {
            let (i, j, k) = (indices[0], indices[1], indices[2]);

            let normal = alg::Vec3::normal(
                particles[i].position,
                particles[j].position,
                particles[k].position,
            );

            // Sum normal contributions
            result[i] = result[i] + normal;
            result[j] = result[j] + normal;
            result[k] = result[k] + normal;
        }

        // Rescale
        result.iter().map(|raw| raw.norm()).collect()
    }

    // Get offset from center for specific particle
    fn offset(&self, index: usize) -> alg::Vec3 {
        // Compare current transform against model reference
        self.frame_orientation.conjugate() * (
            self.particles[index].position - self.frame_position
        ) - self.model.as_ref().unwrap_or(&self.perfect_model)[index]
    }

    #[inline]
    // Must be called when gravity or force changes
    fn update_cache(&mut self, gravity: alg::Vec3) {
        self.accel_dt = (self.force * self.inv_pt_mass + gravity)
            * FIXED_DT * FIXED_DT;
    }

    /* General instance methods */

    pub fn center(&self) -> alg::Vec3 {
        self.particles.iter().fold(
            alg::Vec3::zero(),
            |sum, particle| sum + particle.position
        ) / self.particles.len() as f32
    }

    pub fn velocity(&self) -> alg::Vec3 {
        self.particles.iter().fold(
            alg::Vec3::zero(),
            |sum, particle| sum + particle.displacement,
        ) / self.particles.len() as f32
    }

    /* Limb methods */

    #[inline]
    fn start(&self) -> alg::Vec3 {
        (     self.particles[0].position
            + self.particles[1].position
            + self.particles[2].position
            + self.particles[3].position
        ) * 0.25
    }

    #[inline]
    fn end(&self) -> alg::Vec3 {
        (     self.particles[4].position
            + self.particles[5].position
            + self.particles[6].position
            + self.particles[7].position
        ) * 0.25
    }

    #[inline]
    fn top(&self) -> alg::Vec3 {
        (     self.particles[0].position
            + self.particles[1].position
            + self.particles[4].position
            + self.particles[5].position
        ) * 0.25
    }

    #[inline]
    fn bot(&self) -> alg::Vec3 {
        (     self.particles[2].position
            + self.particles[3].position
            + self.particles[6].position
            + self.particles[7].position
        ) * 0.25
    }

    // Orientation relies on forward vector calculation
    #[inline]
    fn fwd(&self) -> alg::Vec3 {
        (self.end() - self.start()).norm()
    }

    #[inline]
    fn up_est(&self) -> alg::Vec3 {
        (self.top() - self.bot()).norm()
    }

    #[inline]
    fn extend(&self, offset: alg::Vec3) -> alg::Vec3 {
        self.center() + self.orientation() * offset
    }

    // Get limb orientation as matrix
    fn orientation(&self) -> alg::Mat3 {
        // Build orthogonal rotation matrix
        let fwd = self.fwd();
        let up = self.up_est(); // Approximation
        let right = up.cross(fwd); // Resistant to x-axis deformity
        let up = fwd.cross(right); // Recreate up vector

        alg::Mat3::axes(right, up, fwd)
    }

    // Determine instance orientation using least squares fit
    fn matched_orientation(&self, center: alg::Vec3) -> alg::Mat3 {
        let mut transform = alg::Mat3::zero();

        // Sum multiplication of actual and model particle positions
        for i in 0..self.particles.len() {
            let actual = self.particles[i].position - center;
            let model = self.perfect_model[i] - self.perfect_com;
            transform = transform + (actual * model);
        }

        // Compute rotation component using polar decomposition
        let s = (transform.transpose() * transform).sqrt();
        transform * s.inverse()
    }

    #[inline]
    #[allow(dead_code)]
    fn rotate_start(&mut self, rotation: alg::Quat) {
        let point = self.start();
        self.rotate_around(rotation, point);
    }

    #[inline]
    fn rotate_end(&mut self, rotation: alg::Quat) {
        let point = self.end();
        self.rotate_around(rotation, point);
    }

    #[inline]
    #[allow(dead_code)]
    fn transform_inner(
        &mut self,
        rotation: alg::Quat,
        translation: alg::Vec3,
    ) {
        let point = self.center();

        // Center axis of rotation
        for i in 0..8 {
            self.particles[i].position = self.particles[i].position - point;
        }

        // Translate
        for i in 0..8 {
            self.particles[i].position = self.particles[i].position
                + translation;
        }

        // Rotate
        for i in 0..8 {
            self.particles[i].position = rotation
                * self.particles[i].position;
        }

        // Move back to world space
        for i in 0..8 {
            self.particles[i].position = self.particles[i].position + point;
        }
    }

    fn transform_outer(
        &mut self,
        rotation: alg::Quat,
        translation: alg::Vec3,
    ) {
        // Translate
        for i in 0..8 {
            self.particles[i].position = self.particles[i].position
                + translation;
        }

        let point = self.center();

        // Center axis of rotation
        for i in 0..8 {
            self.particles[i].position = self.particles[i].position - point;
        }

        // Rotate
        for i in 0..8 {
            self.particles[i].position = rotation
                * self.particles[i].position;
        }

        // Move back to world space
        for i in 0..8 {
            self.particles[i].position = self.particles[i].position + point;
        }
    }

    #[inline]
    fn rotate_around(&mut self, rotation: alg::Quat, point: alg::Vec3) {
        // Rotate
        for i in 0..8 {
            self.particles[i].position = rotation
                * (self.particles[i].position - point) // Center rotation axis
                + point; // Move back to world space
        }
    }

    #[inline]
    fn translate(&mut self, offset: alg::Vec3) {
        for i in 0..8 {
            let new = self.particles[i].position + offset;
            self.particles[i].position = new;
        }
    }
}

/// Builder pattern for softbody instances
pub struct InstanceBuilder<'a> {
    manager: &'a mut Manager,
    scale: Option<alg::Vec3>, // For optional limb creation
    mass: f32,
    rigidity: f32,
    particles: Option<&'a [alg::Vec3]>,
    indices: Option<&'a [usize]>,
    bindings: Option<&'a [(usize, usize)]>,
    match_shape: bool,
}

impl<'a> InstanceBuilder<'a> {
    // Initialize with manager
    pub fn new(manager: &mut Manager) -> InstanceBuilder {
        InstanceBuilder {
            manager,
            scale: None,
            mass: INST_DEFAULT_MASS,
            rigidity: INST_DEFAULT_RIGID,
            particles: None,
            indices: None,
            bindings: None,
            match_shape: false,
        }
    }

    /// Override general instance creation with limb preset.
    /// Takes in the limb scale as an argument.
    pub fn make_limb(&mut self, scale: alg::Vec3) -> &mut InstanceBuilder<'a> {
        self.scale = Some(scale);
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

    pub fn match_shape(&mut self) -> &mut InstanceBuilder<'a> {
        self.match_shape = true;
        self
    }

    /// Finalize
    pub fn for_entity(&mut self, entity: entity::Handle) {
        let rigidity = self.rigidity * 0.5; // Scale rigidity properly
        let initial_accel = self.manager.gravity; // Initialize with gravity

        /* Limb instance */

        let instance = if let Some(scale) = self.scale {
            let scale = scale * 0.5;

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
                /* Override scaled input with unit cube.
                 * Enables offsets to work properly with non-matching mesh.
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
                rigidity,
                initial_accel,
            )
        }

        /* Generic instance */

        else {
            debug_assert!(self.particles.is_some());
            debug_assert!(self.indices.is_some());

            let bindings = if let Some(rods) = self.bindings
                { rods } else { &[] };

            Instance::new(
                self.particles.unwrap(),
                self.indices.unwrap(),
                None, // No model override
                bindings,
                self.match_shape,
                self.mass,
                rigidity,
                initial_accel,
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
    joints: fnv::FnvHashMap<usize, Vec<Joint>>,
    planes: Vec<alg::Plane>,
    gravity: alg::Vec3,
    bounce: f32,
    friction: f32,
    count: usize,
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
            joints: joint_map,
            planes: Vec::with_capacity(plane_hint),
            gravity: alg::Vec3::new(0., -9.8, 0.), // Default gravity
            bounce: MNGR_DEFAULT_BOUNCE,
            friction: MNGR_DEFAULT_FRICTION,
            count: 0,
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

    pub fn set_force(&mut self, entity: entity::Handle, force: alg::Vec3) {
        let instance = get_mut_instance!(self, entity);
        instance.force = force;
        instance.update_cache(self.gravity);
    }

    pub fn get_particle(
        &self,
        entity: entity::Handle,
        index: usize,
    ) -> alg::Vec3 {
        let instance = get_instance!(self, entity);
        debug_assert!(index < instance.particles.len());
        instance.particles[index].position
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
            for i in 0..instance.particles.len() {
                offsets[i] = render::PaddedVec3::new(instance.offset(i));
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
            let new = Instance::compute_normals(
                &instance.particles,
                &instance.triangles,
            );

            // Compute offsets
            for i in 0..new.len() {
                offsets[i] = render::PaddedVec3::new(
                    new[i] - instance.normals[i]
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

            let rotation = parent.orientation().to_quat() * transform;
            let translation = parent.extend(offset) + rotation * child.end();

            // Align child with parent and joint transform
            child.transform_outer(rotation, translation);

            // Reset child position for integrator
            for particle in &mut child.particles {
                particle.last = particle.position;
            }
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
                instance.update_cache(self.gravity);
            }
        }
    }

    /// Set gravity for all instances. \
    /// May not immediately affect all instances.
    pub fn set_gravity_raw(&mut self, gravity: alg::Vec3) {
        self.gravity = gravity;
    }

    /// Range 0 - inf; "Realistic" = 2.0 \
    /// Values < 2 become force zones, values > 2 add impossible force. \
    /// A value of zero nullifies all collisions.
    pub fn set_bounce(&mut self, bounce: f32) {
        self.bounce = bounce;
    }

    /// Range 0 - 1; 0 = no planar friction
    pub fn set_friction(&mut self, friction: f32) {
        self.friction = friction;
    }

    pub(crate) fn simulate<T>(
        &mut self,
        game: &mut T,
        transforms: &mut transform::Manager
    ) where T: Iterate {
        // Update instance particles
        for i in 0..self.instances.len() {
            let mut instance = match self.instances[i] {
                Some(ref mut instance) => instance,
                None => continue,
            };

            // Position Verlet
            for particle in &mut instance.particles {
                let next_position = particle.position * 2.
                    - particle.last
                    + instance.accel_dt;

                particle.displacement = (next_position - particle.last) / 2.0;
                particle.last = particle.position;
                particle.position = next_position;
            }

            // Plane friction
            for plane in &self.planes {
                for particle in &mut instance.particles {
                    let distance = plane.dist(particle.position);

                    if distance > 0. {
                        continue;
                    }

                    let direction = particle.displacement.norm();
                    let tangent = direction
                        .cross(plane.normal)
                        .cross(plane.normal);

                    let factor = tangent.dot(direction);
                    let projected = tangent
                        * particle.displacement.mag() * factor;

                    particle.position = particle.position
                        - projected * self.friction;
                }
            }
        }

        // Solve constraints
        for _ in 0..ITERATIONS {
            for i in 0..self.instances.len() {
                let mut instance = match self.instances[i] {
                    Some(ref mut instance) => instance,
                    None => continue,
                };

                // Plane collision
                for plane in &self.planes {
                    for particle in &mut instance.particles {
                        let distance = plane.dist(particle.position);

                        if distance > 0. {
                            continue;
                        }

                        particle.position = particle.position
                            - plane.normal * self.bounce * distance;
                    }
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
                    let orientation = instance.matched_orientation(center);

                    for (particle, model_point) in instance.particles.iter_mut()
                        .zip(instance.perfect_model.iter())
                    {
                        let target = orientation
                            * (*model_point - instance.perfect_com)
                            + center;

                        let offset = target - particle.position;

                        particle.position = particle.position
                            + offset * instance.rigidity;
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
        for _ in 0..ITERATIONS {
            // External constraints
            game.iterate(FIXED_DT, self);

            // Joint constraints
            self.solve_joints();
        }

        // Finalize instances
        for i in 0..self.instances.len() {
            let mut instance = match self.instances[i] {
                Some(ref mut instance) => instance,
                None => continue,
            };

            // Compute average position and best fit orientation
            let center = instance.center();
            let orientation = instance.matched_orientation(center).to_quat();

            // Update instance position and orientation
            instance.frame_position = center;
            instance.frame_orientation = orientation;

            // Update transform
            debug_validate_entity!(transforms, self.handles[i].unwrap());
            transforms.set_raw(i, center, orientation, alg::Vec3::one());
        }
    }

    #[inline]
    fn solve_joints(&mut self) {
        for (parent_index, joints) in &self.joints {
            /* Unsafely acquire mutable references to vector elements.
             * Unfortunately, the only safe Rust alternative (split_at_mut())
             * is slower.
             */

            let mut parent = unsafe {
                let ptr = self.instances.as_mut_ptr()
                    .offset(*parent_index as isize);

                (*ptr).as_mut().unwrap()
            };

            let mut children = Vec::with_capacity(joints.len());

            for joint in joints {
                let mut child = unsafe {
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

                let parent_end = parent.extend(joints[i].offset);
                let parent_start = parent.extend(-joints[i].offset);

                // Find midpoint for initial correction
                let midpoint = children[i].start().lerp(parent_end, weight);

                /* Rotate child towards midpoint */

                let child_correction = alg::Quat::from_to(
                    children[i].fwd(),
                    (children[i].end() - midpoint).norm(),
                );

                children[i].rotate_end(child_correction);

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

                let offset = (children[i].start() - parent_end)
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
        // If all limits are equal, then the joint is unlimited
        debug_assert!(!(
               joint.x_limit.min == joint.x_limit.max
            && joint.y_limit.min == joint.y_limit.max
            && joint.x_limit.min == joint.y_limit.min
        ));

        let parent_orient = parent.orientation();
        let child_orient = child.orientation();
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

        // Rebind (limit) twist
        let (axis, angle) = twist.to_axis_angle();
        let twist = alg::Quat::axis_angle(
            axis,
            angle.max(joint.z_limit.min).min(joint.z_limit.max),
        );

        // Calculate mass imbalance
        let weight = 1. / (child.mass / parent.mass + 1.);

        /* Correct child */

        let point = child.start();

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
        let point = parent.extend(-joint.offset);

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
            // (Vec3::fwd() + local_child_fwd - alg::Vec3::fwd()).norm()
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

    #[allow(unused_variables)]
    pub fn draw_debug(
        &self,
        entity: entity::Handle,
        debug: &mut debug::Handler,
    ) {
        #[cfg(debug_assertions)] {
            let i = entity.get_index() as usize;
            self.draw_instance_debug(i, debug);
        }
    }

    #[allow(unused_variables)]
    pub fn draw_all_debug(&self, debug: &mut debug::Handler) {
        #[cfg(debug_assertions)] {
            for i in 0..self.instances.len() {
                self.draw_instance_debug(i, debug);
            }
        }
    }

    #[allow(unused_variables)]
    fn draw_instance_debug(&self, index: usize, debug: &mut debug::Handler) {
        #[cfg(debug_assertions)] {
            debug_assert!(index < self.instances.len());

            if let Some(ref instance) = self.instances[index] {
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

                    for triangle in instance.triangles.chunks(3) {
                        draw(triangle, 0, 1);
                        draw(triangle, 1, 2);
                        draw(triangle, 2, 0);
                    }
                }
            }

            // Draw joint endpoints
            for (parent_index, joints) in &self.joints {
                if *parent_index == index {
                    let parent = self.instances[*parent_index]
                        .as_ref().unwrap();

                    for joint in joints {
                        let joint_orientation = parent.orientation()
                            * joint.transform.conjugate().to_mat();

                        debug.add_local_axes(
                            parent.extend(joint.offset),
                            joint_orientation * alg::Vec3::fwd(),
                            joint_orientation * alg::Vec3::up(),
                            1.0,
                            1.0,
                        );
                    }
                }
            }
        }
    }
}
