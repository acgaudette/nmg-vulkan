#![allow(dead_code)] // Library

use std;

const JACOBI_ITERATIONS: usize = 16;
const JACOBI_SKIP_SCALE: f32 = 10.0;
const JACOBI_SKIP_ITERATIONS: usize = 4;

// For kicks
fn inverse_sqrt(x: f32) -> f32 {
    let half = x * 0.5;

    let cast: u32 = unsafe {
        std::mem::transmute(x)
    };

    let guess = 0x5f3759df - (cast >> 1);
    let guess = f32::from_bits(guess);

    let iteration = guess * (1.5 - half * guess * guess);
    iteration * (1.5 - half * iteration * iteration)
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    #[inline]
    pub fn up() -> Vec2 {
        Vec2 {
            x: 0.0,
            y: 1.0,
        }
    }

    #[inline]
    pub fn right() -> Vec2 {
        Vec2 {
            x: 1.0,
            y: 0.0,
        }
    }

    #[inline]
    pub fn zero() -> Vec2 {
        Vec2 {
            x: 0.0,
            y: 0.0,
        }
    }

    #[inline]
    pub fn one() -> Vec2 {
        Vec2::new(1., 1.)
    }

    #[inline]
    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn norm(self) -> Vec2 {
        let inverse_len = inverse_sqrt(self.mag_squared());

        Vec2 {
            x: self.x * inverse_len,
            y: self.y * inverse_len,
        }
    }

    pub fn mag_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;

    fn add(self, other: Vec2) -> Vec2 {
        Vec2::new(
            self.x + other.x,
            self.y + other.y,
        )
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;

    fn sub(self, other: Vec2) -> Vec2 {
        Vec2::new(
            self.x - other.x,
            self.y - other.y,
        )
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, scalar: f32) -> Vec2 {
        Vec2::new(
            self.x * scalar,
            self.y * scalar,
        )
    }
}

impl std::fmt::Display for Vec2 {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            out,
            "( {}, {} )",
            self.x, self.y,
        )
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    #[inline]
    pub fn right() -> Vec3 {
        Vec3::new(1., 0., 0.)
    }

    #[inline]
    pub fn up() -> Vec3 {
        Vec3::new(0., 1., 0.)
    }

    #[inline]
    pub fn fwd() -> Vec3 {
        Vec3::new(0., 0., 1.)
    }

    #[inline]
    pub fn zero() -> Vec3 {
        Vec3::new(0., 0., 0.,)
    }

    #[inline]
    pub fn one() -> Vec3 {
        Vec3::new(1., 1., 1.)
    }

    #[inline]
    // Triangle normal (CW)
    pub fn normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        (b - a).cross(c - a).norm()
    }

    pub fn norm(self) -> Vec3 {
        let inverse_len = inverse_sqrt(self.mag_squared());

        Vec3::new(
            self.x * inverse_len,
            self.y * inverse_len,
            self.z * inverse_len,
        )
    }

    pub fn mag_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn mag(self) -> f32 {
        self.mag_squared().sqrt()
    }

    pub fn dist_squared(self, other: Vec3) -> f32 {
        (self - other).mag_squared()
    }

    pub fn dist(self, other: Vec3) -> f32 {
        self.dist_squared(other).sqrt()
    }

    #[inline]
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn lerp(self, other: Vec3, t: f32) -> Vec3 {
        self * (1. - t) + other * t
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, scalar: f32) -> Vec3 {
        Vec3::new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, scalar: f32) -> Vec3 {
        Vec3::new(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
        )
    }
}

// Multiply column vector (LHS) by row vector (RHS) to produce a matrix
impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Mat3;

    fn mul(self, other: Vec3) -> Mat3 {
        Mat3::new(
            self.x * other.x, self.x * other.y, self.x * other.z,
            self.y * other.x, self.y * other.y, self.y * other.z,
            self.z * other.x, self.z * other.y, self.z * other.z,
        )
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            out,
            "( {}, {}, {} )",
            self.x, self.y, self.z,
        )
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Mat3 {
    /* GLSL/SPIR-V expects matrices in column-major order
     * Here I format methods transposed for conventional readability:
     * [ x0 ] [ x1 ] [ x2 ]
     * [ y0 ] [ y1 ] [ y2 ]
     * [ z0 ] [ z1 ] [ z2 ]
     */

    pub x0: f32, pub y0: f32, pub z0: f32,
    pub x1: f32, pub y1: f32, pub z1: f32,
    pub x2: f32, pub y2: f32, pub z2: f32,
}

impl Mat3 {
    pub fn new(
        x0 : f32, x1: f32, x2: f32,
        y0 : f32, y1: f32, y2: f32,
        z0 : f32, z1: f32, z2: f32,
    ) -> Mat3 {
        Mat3 {
            x0, x1, x2,
            y0, y1, y2,
            z0, z1, z2,
        }
    }

    pub fn get(self, row: usize, column: usize) -> f32 {
        match (row, column) {
            (0, 0) => self.x0,
            (0, 1) => self.x1,
            (0, 2) => self.x2,
            (1, 0) => self.y0,
            (1, 1) => self.y1,
            (1, 2) => self.y2,
            (2, 0) => self.z0,
            (2, 1) => self.z1,
            (2, 2) => self.z2,
            _ => panic!("Mat3 index ({}, {}) out of bounds", row, column)
        }
    }

    pub fn set(&mut self, row: usize, column: usize, value: f32) {
        match (row, column) {
            (0, 0) => self.x0 = value,
            (0, 1) => self.x1 = value,
            (0, 2) => self.x2 = value,
            (1, 0) => self.y0 = value,
            (1, 1) => self.y1 = value,
            (1, 2) => self.y2 = value,
            (2, 0) => self.z0 = value,
            (2, 1) => self.z1 = value,
            (2, 2) => self.z2 = value,
            _ => panic!("Mat3 index ({}, {}) out of bounds", row, column)
        }
    }

    #[inline]
    pub fn id() -> Mat3 {
        Mat3::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn zero() -> Mat3 {
        Mat3::new(
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        )
    }

    #[inline]
    pub fn new_diagonal(x0: f32, y1: f32, z2: f32) -> Mat3 {
        Mat3::new(
            x0, 0.0, 0.0,
            0.0, y1, 0.0,
            0.0, 0.0, z2,
        )
    }

    #[inline]
    pub fn diagonal(self) -> Mat3 {
        Mat3::new_diagonal(self.x0, self.y1, self.z2)
    }

    pub fn is_diagonal(self) -> bool {
           self.x1 == 0.0 && self.x2 == 0.0
        && self.y0 == 0.0 && self.y2 == 0.0
        && self.z0 == 0.0 && self.z1 == 0.0
    }

    pub fn transpose(self) -> Mat3 {
        Mat3::new(
            self.x0, self.y0, self.z0,
            self.x1, self.y1, self.z1,
            self.x2, self.y2, self.z2,
        )
    }

    pub fn inverse(self) -> Mat3 {
        let reciprocal = 1. / self.det();

        Mat3::new(
            // Row 0: x0, x1, x2
            ((self.y1 * self.z2) - (self.z1 * self.y2)) * reciprocal,
            ((self.x2 * self.z1) - (self.x1 * self.z2)) * reciprocal,
            ((self.x1 * self.y2) - (self.x2 * self.y1)) * reciprocal,

            // Row 1: y0, y1, y2
            ((self.y2 * self.z0) - (self.y0 * self.z2)) * reciprocal,
            ((self.x0 * self.z2) - (self.x2 * self.z0)) * reciprocal,
            ((self.y0 * self.x2) - (self.x0 * self.y2)) * reciprocal,

            // Row 2: z0, z1, z2
            ((self.y0 * self.z1) - (self.z0 * self.y1)) * reciprocal,
            ((self.z0 * self.x1) - (self.x0 * self.z1)) * reciprocal,
            ((self.x0 * self.y1) - (self.y0 * self.x1)) * reciprocal,
        )
    }

    pub fn det(self) -> f32 {
          self.x0 * ((self.y1 * self.z2) - (self.z1 * self.y2))
        - self.x1 * ((self.y0 * self.z2) - (self.y2 * self.z0))
        + self.x2 * ((self.y0 * self.z1) - (self.y1 * self.z0))
    }

    pub fn trace(self) -> f32 {
        self.x0 + self.y1 + self.z2
    }

    pub fn axes(right: Vec3, up: Vec3, fwd: Vec3) -> Mat3 {
        Mat3::new(
            right.x, up.x, fwd.x,
            right.y, up.y, fwd.y,
            right.z, up.z, fwd.z,
        )
    }

    pub fn inverse_axes(right: Vec3, up: Vec3, fwd: Vec3) -> Mat3 {
        Mat3::new(
            right.x, right.y, right.z,
               up.x,    up.y,    up.z,
              fwd.x,   fwd.y,   fwd.z,
        )
    }

    pub fn rotation_x(rad: f32) -> Mat3 {
        Mat3::new(
            1.0,       0.0,        0.0,
            0.0, rad.cos(), -rad.sin(),
            0.0, rad.sin(),  rad.cos(),
        )
    }

    pub fn rotation_y(rad: f32) -> Mat3 {
        Mat3::new(
             rad.cos(), 0.0, rad.sin(),
                   0.0, 1.0,       0.0,
            -rad.sin(), 0.0, rad.cos(),
        )
    }

    pub fn rotation_z(rad: f32) -> Mat3 {
        Mat3::new(
             rad.cos(), rad.sin(), 0.0,
            -rad.sin(), rad.cos(), 0.0,
                  0.0,        0.0, 1.0,
        )
    }

    pub fn rotation(x: f32, y: f32, z: f32) -> Mat3 {
        Mat3::rotation_x(x) * Mat3::rotation_y(y) * Mat3::rotation_z(z)
    }

    pub fn to_quat(self) -> Quat {
        let trace = self.trace();

        if trace > 0.0 {
            let s = 2.0 * (1.0 + trace).sqrt();

            Quat::new(
                (self.z1 - self.y2) / s,
                (self.x2 - self.z0) / s,
                (self.y0 - self.x1) / s,
                0.25 * s,
            )
        } else if (self.x0 > self.y1) && (self.x0 > self.z2) {
            let s = 2.0 * (1.0 + self.x0 - self.y1 - self.z2).sqrt();

            Quat::new(
                s * 0.25,
                (self.x1 + self.y0) / s,
                (self.x2 + self.z0) / s,
                (self.z1 - self.y2) / s,
            )
        } else if self.y1 > self.z2 {
            let s = 2.0 * (1.0 + self.y1 - self.x0 - self.z2).sqrt();

            Quat::new(
                (self.x1 + self.y0) / s,
                0.25 * s,
                (self.y2 + self.z1) / s,
                (self.x2 - self.z0) / s,
            )
        } else {
            let s = 2.0 * (1.0 + self.z2 - self.x0 - self.y1).sqrt();

            Quat::new(
                (self.x2 + self.z0) / s,
                (self.y2 + self.z1) / s,
                0.25 * s,
                (self.y0 - self.x1) / s,
            )
        }
    }

    pub fn diagonal_sqrt(self) -> Mat3 {
        debug_assert!(self.x0 >= 0.0);
        debug_assert!(self.y1 >= 0.0);
        debug_assert!(self.z2 >= 0.0);

        Mat3::new_diagonal(
            self.x0.sqrt(),
            self.y1.sqrt(),
            self.z2.sqrt(),
        )
    }

    pub fn sqrt(self) -> Mat3 {
        // Check for early exit
        if self.is_diagonal() { self.diagonal_sqrt() }

        else {
            let (vectors, values) = self.jacobi(); // Diagonalize
            let d = values.diagonal_sqrt(); // Get square root

            // Diagonalizing matrix is orthogonal; can use transpose
            vectors * d * vectors.transpose()
        }
    }

    /* Jacobi eigenvalue algorithm
     * Input: real symmetric matrix
     * Returns (eigenvectors, diagonal eigenvalues)
     */

    pub fn jacobi(self) -> (Mat3, Mat3) {
        let mut vectors = Mat3::id(); // Eigenvectors
        // Initialize eigenvalues with diagonal
        let mut values = [self.x0, self.y1, self.z2];

        let mut b = values.clone();
        let mut accumulator = [0.0; 3];
        let mut input = self.clone();

        let rotate = |
            matrix: &mut Mat3,
            g_cell: (usize, usize),
            h_cell: (usize, usize),
            s: f32, tau: f32,
        | {
            let g = matrix.get(g_cell.0, g_cell.1);
            let h = matrix.get(h_cell.0, h_cell.1);

            matrix.set(g_cell.0, g_cell.1, g - s * (h + g * tau));
            matrix.set(h_cell.0, h_cell.1, h + s * (g - h * tau));
        };

        for iteration in 0..JACOBI_ITERATIONS {
            // Iterate through off-diagonal
            for i in 0..3 {
                for j in (i + 1)..3 {
                    /* Skip rotation if the current cell is close to zero */

                    let scaled = JACOBI_SKIP_SCALE * input.get(i, j).abs();
                    let i_abs = values[i].abs();
                    let j_abs = values[j].abs();

                    if iteration >= JACOBI_SKIP_ITERATIONS
                        && scaled + i_abs == i_abs
                        && scaled + j_abs == j_abs
                    {
                        input.set(i, j, 0.0);
                    }

                    /* Perform rotation */

                    else {
                        let h = values[j] - values[i];
                        let h_abs = h.abs();

                        let t = if scaled + h_abs == h_abs {
                            input.get(i, j) / if h == 0.0 {
                                std::f32::EPSILON
                            } else { h }
                        } else {
                            let theta = 0.5 * h / input.get(i, j);

                            let t = 1.0 / (
                                theta.abs() + (1.0 + theta * theta).sqrt()
                            );

                            let t = if t.is_nan() { 0.0 } else { t };
                            if theta < 0.0 { -t } else { t }
                        };

                        let c = 1.0 / (1.0 + t * t).sqrt();
                        let s = t * c;
                        let tau = s / (1.0 + c);

                        /* Accumulate */

                        let h = t * input.get(i, j);

                        accumulator[i] -= h;
                        accumulator[j] += h;
                        values[i] -= h;
                        values[j] += h;

                        input.set(i, j, 0.0);

                        /* Rotate */

                        for k in 0..i {
                            rotate(&mut input, (k, i), (k, j), s, tau);
                        }

                        for k in (i + 1)..j {
                            rotate(&mut input, (i, k), (k, j), s, tau);
                        }

                        for k in (j + 1)..3 {
                            rotate(&mut input, (i, k), (j, k), s, tau);
                        }

                        for k in 0..3 {
                            rotate(&mut vectors, (k, i), (k, j), s, tau);
                        }
                    }
                }
            }

            // Update values and reset accumulator
            for i in 0..3 {
                b[i] += accumulator[i];
                values[i] = b[i];
                accumulator[i] = 0.0;
            }
        }

        /* Sort in ascending order via network */

        {
            let mut compare_swap = |i, j| {
                if values[i] > values[j] {
                    values.swap(i, j);

                    // Swap eigenvector columns
                    for k in 0..3 {
                        let ki = vectors.get(k, i);
                        let kj = vectors.get(k, j);
                        vectors.set(k, j, ki);
                        vectors.set(k, i, kj);
                    }
                }
            };

            compare_swap(0, 1);
            compare_swap(0, 2);
            compare_swap(1, 2);
        }

        debug_assert!(values[0] <= values[1]);
        debug_assert!(values[1] <= values[2]);

        let values = Mat3::new_diagonal(
            // Clamp result in case it's close to zero
            values[0].max(0.0),
            values[1].max(0.0),
            values[2].max(0.0),
        );

        (vectors, values)
    }

    pub fn to_cardan(self) -> (f32, f32, f32) {
        let cy = (
            self.x0 * self.x0 + self.x1 * self.x1
        ).sqrt();

        if cy < 16. * std::f32::EPSILON { // Singular matrix
            (
               -(-self.z1).atan2(self.y1),
               -(-self.x2).atan2(cy),
                0.0, // Fix for gimbal lock
            )
        } else {
            (
               -( self.y2).atan2(self.z2),
               -(-self.x2).atan2(cy),
                ( self.x1).atan2(self.x0),
            )
        }
    }

    pub fn to_cardan_safe(self) -> (f32, f32, f32) {
        let cy = (
            self.x0 * self.x0 + self.x1 * self.x1
        ).sqrt();

        let ax = -(self.y2).atan2(self.z2);
        let cx = ax.cos();
        let sx = ax.sin();

        (
            ax,
            -(-self.x2).atan2(cy),
            (sx * self.z0 - cx * self.y0).atan2(cx * self.y1 - sx * self.z1),
        )
    }
}

impl std::ops::Mul for Mat3 {
    type Output = Mat3;

    fn mul(self, m: Mat3) -> Mat3 {
        let x0 = self.x0 * m.x0 + self.x1 * m.y0 + self.x2 * m.z0;
        let x1 = self.x0 * m.x1 + self.x1 * m.y1 + self.x2 * m.z1;
        let x2 = self.x0 * m.x2 + self.x1 * m.y2 + self.x2 * m.z2;

        let y0 = self.y0 * m.x0 + self.y1 * m.y0 + self.y2 * m.z0;
        let y1 = self.y0 * m.x1 + self.y1 * m.y1 + self.y2 * m.z1;
        let y2 = self.y0 * m.x2 + self.y1 * m.y2 + self.y2 * m.z2;

        let z0 = self.z0 * m.x0 + self.z1 * m.y0 + self.z2 * m.z0;
        let z1 = self.z0 * m.x1 + self.z1 * m.y1 + self.z2 * m.z1;
        let z2 = self.z0 * m.x2 + self.z1 * m.y2 + self.z2 * m.z2;

        Mat3::new(
            x0, x1, x2,
            y0, y1, y2,
            z0, z1, z2,
        )
    }
}

impl std::ops::Mul<Mat4> for Mat3 {
    type Output = Mat4;

    fn mul(self, other: Mat4) -> Mat4 {
        let (a, b) = (self, other);

        let x0 = a.x0 * b.x0 + a.x1 * b.y0 + a.x2 * b.z0;
        let x1 = a.x0 * b.x1 + a.x1 * b.y1 + a.x2 * b.z1;
        let x2 = a.x0 * b.x2 + a.x1 * b.y2 + a.x2 * b.z2;
        let x3 = a.x0 * b.x3 + a.x1 * b.y3 + a.x2 * b.z3;

        let y0 = a.y0 * b.x0 + a.y1 * b.y0 + a.y2 * b.z0;
        let y1 = a.y0 * b.x1 + a.y1 * b.y1 + a.y2 * b.z1;
        let y2 = a.y0 * b.x2 + a.y1 * b.y2 + a.y2 * b.z2;
        let y3 = a.y0 * b.x3 + a.y1 * b.y3 + a.y2 * b.z3;

        let z0 = a.z0 * b.x0 + a.z1 * b.y0 + a.z2 * b.z0;
        let z1 = a.z0 * b.x1 + a.z1 * b.y1 + a.z2 * b.z1;
        let z2 = a.z0 * b.x2 + a.z1 * b.y2 + a.z2 * b.z2;
        let z3 = a.z0 * b.x3 + a.z1 * b.y3 + a.z2 * b.z3;

        let w0 = b.w0;
        let w1 = b.w1;
        let w2 = b.w2;
        let w3 = b.w3;

        Mat4::new(
            x0, x1, x2, x3,
            y0, y1, y2, y3,
            z0, z1, z2, z3,
            w0, w1, w2, w3,
        )
    }
}

impl std::ops::Mul<Vec3> for Mat3 {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3::new(
            self.x0 * vec.x + self.x1 * vec.y + self.x2 * vec.z,
            self.y0 * vec.x + self.y1 * vec.y + self.y2 * vec.z,
            self.z0 * vec.x + self.z1 * vec.y + self.z2 * vec.z,
        )
    }
}

impl std::ops::Add for Mat3 {
    type Output = Mat3;

    fn add(self, other: Mat3) -> Mat3 {
        Mat3::new(
            self.x0 + other.x0, self.x1 + other.x1, self.x2 + other.x2,
            self.y0 + other.y0, self.y1 + other.y1, self.y2 + other.y2,
            self.z0 + other.z0, self.z1 + other.z1, self.z2 + other.z2,
        )
    }
}

impl std::fmt::Display for Mat3 {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            out,
            "[ {}, {}, {} ]\n[ {}, {}, {} ]\n[ {}, {}, {} ]",
            self.x0, self.x1, self.x2,
            self.y0, self.y1, self.y2,
            self.z0, self.z1, self.z2,
        )
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Mat4 {

    /* GLSL/SPIR-V expects matrices in column-major order
     * Here I format methods transposed for conventional readability:
     * [ x0 ] [ x1 ] [ x2 ] [ x3 ]
     * [ y0 ] [ y1 ] [ y2 ] [ y3 ]
     * [ z0 ] [ z1 ] [ z2 ] [ z3 ]
     * [ w0 ] [ w1 ] [ w2 ] [ w3 ]
     */

    pub x0: f32, pub y0: f32, pub z0: f32, pub w0: f32,
    pub x1: f32, pub y1: f32, pub z1: f32, pub w1: f32,
    pub x2: f32, pub y2: f32, pub z2: f32, pub w2: f32,
    pub x3: f32, pub y3: f32, pub z3: f32, pub w3: f32,
}

impl Mat4 {
    pub fn new(
        x0: f32, x1: f32, x2: f32, x3: f32,
        y0: f32, y1: f32, y2: f32, y3: f32,
        z0: f32, z1: f32, z2: f32, z3: f32,
        w0: f32, w1: f32, w2: f32, w3: f32,
    ) -> Mat4 {
        Mat4 {
            x0, y0, z0, w0,
            x1, y1, z1, w1,
            x2, y2, z2, w2,
            x3, y3, z3, w3,
        }
    }

    #[inline]
    pub fn id() -> Mat4 {
        Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub fn transpose(self) -> Mat4 {
        Mat4::new(
            self.x0, self.y0, self.z0, self.w0,
            self.x1, self.y1, self.z1, self.w1,
            self.x2, self.y2, self.z2, self.w2,
            self.x3, self.y3, self.z3, self.w3,
        )
    }

    pub fn to_mat3(self) -> Mat3 {
        Mat3::new(
            self.x0, self.x1, self.x2,
            self.y0, self.y1, self.y2,
            self.z0, self.z1, self.z2,
        )
    }

    pub fn translation(x: f32, y: f32, z: f32) -> Mat4 {
        Mat4::new(
            1.0, 0.0, 0.0,   x,
            0.0, 1.0, 0.0,   y,
            0.0, 0.0, 1.0,   z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub fn translation_vec(translation: Vec3) -> Mat4 {
        Mat4::translation(translation.x, translation.y, translation.z)
    }

    pub fn scale(x: f32, y: f32, z: f32) -> Mat4 {
        Mat4::new(
              x, 0.0, 0.0, 0.0,
            0.0,   y, 0.0, 0.0,
            0.0, 0.0,   z, 0.0,
            0.0, 0.0, 0.0, 1.0
        )
    }

    pub fn scale_vec(scale: Vec3) -> Mat4 {
        Mat4::scale(scale.x, scale.y, scale.z)
    }

    pub fn transform(translation: Vec3, rotation: Quat, scale: Vec3) -> Mat4 {
        Mat4::translation_vec(translation)
            * (rotation.to_mat() * Mat4::scale_vec(scale))
    }

    pub fn to_scale(self) -> Vec3 {
        Vec3::new(
            Vec3::new(self.x0, self.y0, self.z0).mag(),
            Vec3::new(self.x1, self.y1, self.z1).mag(),
            Vec3::new(self.x2, self.y2, self.z2).mag(),
        )
    }

    pub fn to_rotation(self) -> Mat3 {
        self.to_rotation_raw(self.to_scale())
    }

    pub fn to_rotation_raw(self, scale: Vec3) -> Mat3 {
        Mat3::new(
            self.x0 / scale.x, self.x1 / scale.y, self.x2 / scale.z,
            self.y0 / scale.x, self.y1 / scale.y, self.y2 / scale.z,
            self.z0 / scale.x, self.z1 / scale.y, self.z2 / scale.z,
        )
    }

    // Returns view matrix (inverted)
    pub fn look_at_view(position: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let fwd = (target - position).norm();
        let right = up.cross(fwd).norm();
        let up = fwd.cross(right);

        // Transpose orthogonal matrix to get inverse
        let inverse_rotation = Mat3::inverse_axes(right, up, fwd);

        // Reverse position input
        let inverse_position = Mat4::translation(
            -position.x,
            -position.y,
            -position.z,
        );

        inverse_rotation * inverse_position
    }

    // Input: vertical field of view, screen aspect ratio, near and far planes
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        // Perspective scaling (rectilinear)
        let y_scale = 1. / (0.5 * fov).to_radians().tan();
        let x_scale = y_scale / aspect;

        // Fit into Vulkan clip space (0-1)
        let z_scale = 1. / (far - near);
        let z_offset = -near / (far - near);

        Mat4::new(
            x_scale,      0.0,     0.0,      0.0,
                0.0, -y_scale,     0.0,      0.0, // Flip for Vulkan
                0.0,      0.0, z_scale, z_offset,
                0.0,      0.0,     1.0,      0.0, // Left-handed (scaling factor)
        )
    }
}

impl std::ops::Mul for Mat4 {
    type Output = Mat4;

    fn mul(self, other: Mat4) -> Mat4 {
        let (a, b) = (self, other);

        let x0 = a.x0 * b.x0 + a.x1 * b.y0 + a.x2 * b.z0 + a.x3 * b.w0;
        let x1 = a.x0 * b.x1 + a.x1 * b.y1 + a.x2 * b.z1 + a.x3 * b.w1;
        let x2 = a.x0 * b.x2 + a.x1 * b.y2 + a.x2 * b.z2 + a.x3 * b.w2;
        let x3 = a.x0 * b.x3 + a.x1 * b.y3 + a.x2 * b.z3 + a.x3 * b.w3;

        let y0 = a.y0 * b.x0 + a.y1 * b.y0 + a.y2 * b.z0 + a.y3 * b.w0;
        let y1 = a.y0 * b.x1 + a.y1 * b.y1 + a.y2 * b.z1 + a.y3 * b.w1;
        let y2 = a.y0 * b.x2 + a.y1 * b.y2 + a.y2 * b.z2 + a.y3 * b.w2;
        let y3 = a.y0 * b.x3 + a.y1 * b.y3 + a.y2 * b.z3 + a.y3 * b.w3;

        let z0 = a.z0 * b.x0 + a.z1 * b.y0 + a.z2 * b.z0 + a.z3 * b.w0;
        let z1 = a.z0 * b.x1 + a.z1 * b.y1 + a.z2 * b.z1 + a.z3 * b.w1;
        let z2 = a.z0 * b.x2 + a.z1 * b.y2 + a.z2 * b.z2 + a.z3 * b.w2;
        let z3 = a.z0 * b.x3 + a.z1 * b.y3 + a.z2 * b.z3 + a.z3 * b.w3;

        let w0 = a.w0 * b.x0 + a.w1 * b.y0 + a.w2 * b.z0 + a.w3 * b.w0;
        let w1 = a.w0 * b.x1 + a.w1 * b.y1 + a.w2 * b.z1 + a.w3 * b.w1;
        let w2 = a.w0 * b.x2 + a.w1 * b.y2 + a.w2 * b.z2 + a.w3 * b.w2;
        let w3 = a.w0 * b.x3 + a.w1 * b.y3 + a.w2 * b.z3 + a.w3 * b.w3;

        Mat4::new(
            x0, x1, x2, x3,
            y0, y1, y2, y3,
            z0, z1, z2, z3,
            w0, w1, w2, w3,
        )
    }
}

impl std::ops::Mul<Vec3> for Mat4 {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        // Assume scaling factor of one (translation)
        Vec3::new(
            self.x0 * vec.x + self.x1 * vec.y + self.x2 * vec.z + (self.x3),
            self.y0 * vec.x + self.y1 * vec.y + self.y2 * vec.z + (self.y3),
            self.z0 * vec.x + self.z1 * vec.y + self.z2 * vec.z + (self.z3),
        )
    }
}

impl std::fmt::Display for Mat4 {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            out,
            "[ {}, {}, {}, {} ]\n[ {}, {}, {}, {} ]\n\
            [ {}, {}, {}, {} ]\n[ {}, {}, {}, {} ]",
            self.x0, self.x1, self.x2, self.x3,
            self.y0, self.y1, self.y2, self.y3,
            self.z0, self.z1, self.z2, self.z3,
            self.w0, self.w1, self.w2, self.w3,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quat {
        Quat { x, y, z, w }
    }

    #[inline]
    pub fn id() -> Quat {
        Quat {
            x: 0.,
            y: 0.,
            z: 0.,
            w: 1.,
        }
    }

    pub fn from_vecs(fwd: Vec3, up: Vec3) -> Quat {
        let right = up.cross(fwd);
        Mat3::axes(right, up, fwd).to_quat()
    }

    pub fn axis_angle(axis: Vec3, angle: f32) -> Quat {
        Quat::axis_angle_raw(axis.norm(), angle) // Normalize first
    }

    pub fn axis_angle_raw(axis: Vec3, angle: f32) -> Quat {
        let half = 0.5 * angle;
        let half_sin = half.sin();
        let half_cos = half.cos();

        Quat {
            x: axis.x * half_sin,
            y: axis.y * half_sin,
            z: axis.z * half_sin,
            w: half_cos,
        }
    }

    pub fn look_at(position: Vec3, target: Vec3, up: Vec3) -> Quat {
        let fwd = (target - position).norm();
        let right = up.cross(fwd).norm();
        let up = fwd.cross(right);
        Mat3::axes(right, up, fwd).to_quat()
    }

    pub fn from_to(from: Vec3, to: Vec3) -> Quat {
        let cross = from.cross(to);
        let dot = from.dot(to);

        if dot > 1.0 - std::f32::EPSILON {
            Quat::id()
        } else if dot < -1.0 + std::f32::EPSILON {
            Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            }
        } else {
            Quat {
                x: cross.x,
                y: cross.y,
                z: cross.z,
                w: (from.mag_squared() * to.mag_squared()).sqrt() + dot,
            }
        }
    }

    // Hestenes "simple" rotation (does not preserve Z twist)
    pub fn simple(from: Vec3, to: Vec3) -> Quat {
        let axis = from.cross(to);
        let dot = from.dot(to);

        if dot < -1.0 + std::f32::EPSILON {
            Quat {
                x: 0.0,
                y: 1.0,
                z: 0.0,
                w: 0.0,
            }
        } else {
            Quat {
                x: axis.x,
                y: axis.y,
                z: axis.z,
                w: (dot + 1.0),
            }.norm()
        }
    }

    #[inline]
    pub fn dot(self, other: Quat) -> f32 {
        self.x * other.x
            + self.y * other.y
            + self.z * other.z
            + self.w * other.w
    }

    pub fn norm(self) -> Quat {
        let inverse_len = inverse_sqrt(self.mag_squared());

        Quat {
            x: self.x * inverse_len,
            y: self.y * inverse_len,
            z: self.z * inverse_len,
            w: self.w * inverse_len,
        }
    }

    pub fn mag_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    pub fn mag(self) -> f32 {
        self.mag_squared().sqrt()
    }

    pub fn conjugate(self) -> Quat {
        Quat {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w:  self.w,
        }
    }

    // Normalizes the quaternion (self) in the process
    // Conversion satisfies the Hamilton convention
    pub fn to_mat(self) -> Mat3 {
        let xx = self.x * self.x;
        let yy = self.y * self.y;
        let zz = self.z * self.z;
        let ww = self.w * self.w;

        let inverse = 1.0 / (xx + yy + zz + ww);

        /* Fill out diagonal */

        let x0 = inverse * ( xx - yy - zz + ww);
        let y1 = inverse * (-xx + yy - zz + ww);
        let z2 = inverse * (-xx - yy + zz + ww);

        /* Move across the matrix */

        let (xy, zw) = (self.x * self.y, self.z * self.w);
        let y0 = 2.0 * inverse * (xy + zw);
        let x1 = 2.0 * inverse * (xy - zw);

        let (xz, yw) = (self.x * self.z, self.y * self.w);
        let z0 = 2.0 * inverse * (xz - yw);
        let x2 = 2.0 * inverse * (xz + yw);

        let (yz, xw) = (self.y * self.z, self.x * self.w);
        let z1 = 2.0 * inverse * (yz + xw);
        let y2 = 2.0 * inverse * (yz - xw);

        Mat3 {
            x0, x1, x2,
            y0, y1, y2,
            z0, z1, z2,
        }
    }

    pub fn to_axis_angle(self) -> (Vec3, f32) {
        let this = if self.w > 1.0 { self.norm() } else { self };
        this.to_axis_angle_raw()
    }

    pub fn to_axis_angle_raw(self) -> (Vec3, f32) {
        let inverse = inverse_sqrt(1.0 - self.w * self.w);

        if 1.0 / inverse < std::f32::EPSILON {
            (
                Vec3::new(
                    self.x,
                    self.y,
                    self.z,
                ),
                0.0,
            )
        } else {
            (
                Vec3::new(
                    self.x * inverse,
                    self.y * inverse,
                    self.z * inverse,
                ),
                self.angle(),
            )
        }
    }

    #[inline]
    pub fn abs_angle(self) -> f32 {
        self.w.abs().acos() * 2.0
    }

    #[inline]
    pub fn angle(self) -> f32 {
        self.w.acos() * 2.0
    }

    pub fn pow(self, t: f32) -> Quat {
        let mag = Vec3::new(self.x, self.y, self.z).mag();

        let scale = if mag > std::f32::EPSILON {
            mag.atan2(self.w) * t
        } else { 0.0 };

        let x = self.x * scale;
        let y = self.y * scale;
        let z = self.z * scale;
        let w = self.mag_squared().ln() * 0.5 * t;

        let mag = Vec3::new(x, y, z).mag();
        let wexp = w.exp();

        let scale = if mag >= std::f32::EPSILON {
            wexp * mag.sin() / mag
        } else { 0.0 };

        Quat {
            x: x * scale,
            y: y * scale,
            z: z * scale,
            w: wexp * mag.cos(),
        }
    }

    /// Linearly interpolate from self to target and normalize
    pub fn nlerp(self, target: Quat, t: f32) -> Quat {
        (self + (target - self) * t).norm()
    }
}

impl std::cmp::PartialEq for Quat {
    fn eq(&self, other: &Quat) -> bool {
        let equal = {
               self.x == other.x
            && self.y == other.y
            && self.z == other.z
            && self.w == other.w
        };

        if !equal {
               self.x == -other.x
            && self.y == -other.y
            && self.z == -other.z
            && self.w == -other.w
        } else { true }
    }
}

impl std::ops::Mul<Vec3> for Quat {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        self.to_mat() * vec
    }
}

impl std::ops::Mul for Quat {
    type Output = Quat;

    fn mul(self, other: Quat) -> Quat {
        let x = self.w * other.x
            + self.x * other.w
            + self.y * other.z
            - self.z * other.y;

        let y = self.w * other.y
            + self.y * other.w
            + self.z * other.x
            - self.x * other.z;

        let z = self.w * other.z
            + self.z * other.w
            + self.x * other.y
            - self.y * other.x;

        let w = self.w * other.w
            - self.x * other.x
            - self.y * other.y
            - self.z * other.z;

        Quat { x, y, z, w, }
    }
}

impl std::ops::Mul<f32> for Quat {
    type Output = Quat;

    fn mul(self, scalar: f32) -> Quat {
        Quat {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }
}

impl std::ops::Add for Quat {
    type Output = Quat;

    fn add(self, other: Quat) -> Quat {
        Quat {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl std::ops::Sub for Quat {
    type Output = Quat;

    fn sub(self, other: Quat) -> Quat {
        Quat {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl std::fmt::Display for Quat {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(out, "( {}, {}, {}, {} )", self.x, self.y, self.z, self.w)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub offset: f32,
}

impl Plane {
    pub fn new(normal: Vec3, offset: f32) -> Plane {
        Plane {
            normal: normal.norm(),
            offset,
        }
    }

    pub fn new_raw(normal: Vec3, offset: f32) -> Plane {
        Plane { normal, offset }
    }

    #[inline]
    pub fn contains(self, point: Vec3) -> bool {
        self.normal.dot(point) > 0.0
    }

    #[inline]
    pub fn intersects(self, start: Vec3, ray: Vec3) -> bool {
        self.normal.dot(start + ray) < 0.0
    }

    #[inline]
    pub fn dist(self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.offset
    }

    #[inline]
    pub fn reflect(self, vec: Vec3) -> Vec3 {
        // Does not flip the sign of the result
        self.normal * vec.dot(self.normal) * 2. - vec
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Line {
    pub start: Vec3,
    pub end: Vec3,
}

impl Line {
    pub fn new(start: Vec3, end: Vec3) -> Line {
        Line {
            start,
            end,
        }
    }
}

#[cfg(test)]
mod tests {
    use alg::*;

    /* Vec3 */

    #[test]
    fn norm_vec() {
        // Baseline
        let error = (Vec3::up().norm().mag() - Vec3::up().mag()).abs();

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);

        let vec = Vec3::new(-1., 3., 5.);
        let error = (vec.norm().mag() - 1.).abs();

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);
    }

    #[test]
    fn cross_vec() {
        assert!(Vec3::right().cross(Vec3::up()) == Vec3::fwd());
    }

    /* Mat3 */

    #[test]
    fn mat3_diagonal_sqrt() {
        let mat = Mat3::new_diagonal(4.0, 4.0, 4.0);
        let compare = Mat3::new_diagonal(2.0, 2.0, 2.0);

        let error = mat3_error(mat.diagonal_sqrt(), compare);
        eprintln!("Error: {}", error);
        assert!(error < 0.0001);
    }

    #[test]
    fn mat3_sqrt() {
        let mat = Mat3::new(
            1., 0., 1.,
            0., 1., 0.,
            1., 0., 1.,
        );

        assert!(mat.transpose() == mat);

        let sqrt = mat.sqrt();
        let error = mat3_error(mat, sqrt * sqrt);

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);
    }

    #[test]
    fn invert_mat3() {
        let mat = Mat3::new(
            1.0,  7.0,  3.0,
            7.0,  4.0, -5.0,
            3.0, -5.0,  6.0,
        );

        let id = mat * mat.inverse();
        let error = mat3_error(Mat3::new_diagonal(1.0, 1.0, 1.0), id);

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);
    }

    /* Mat4 */

    #[test]
    fn mul_mat4() {
        let translation = Mat4::translation(1.0, 2.0, 3.0);

        assert!(translation * Mat4::id() == translation);
        assert!(Mat4::id() * translation == translation);
    }

    #[test]
    fn mul_mat4_vec() {
        let vec = Vec3::new(9., -4., 0.);
        let scale = Mat4::scale(-1., 3., 2.);

        assert!(Mat4::id() * vec == vec);
        assert!(scale * vec == Vec3::new(-9., -12., 0.));

        let mat = Mat4::new(
            1., 1., 1., 0.,
            0., 1., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
        );

        assert!(mat * vec == Vec3::new(5., -4., 0.,));

        let translation = Mat4::translation(2., -7., 0.5);
        assert!(translation * Vec3::zero() == Vec3::new(2., -7., 0.5));
    }

    /* Quaternion */

    #[test]
    fn hamilton() {
        let i = Quat::new(1.0, 0.0, 0.0, 0.0);
        let j = Quat::new(0.0, 1.0, 0.0, 0.0);
        let k = (i * j).z;

        assert_eq!(k, 1.0); // -1 would indicate JPL

        let half_sqrt = 0.5f32.sqrt();
        let quat = Quat::new(0.0, 0.0, half_sqrt, half_sqrt);

        let compare = Mat3::new(
            0.0, -1.0, 0.0,
            1.0,  0.0, 0.0,
            0.0,  0.0, 1.0,
        );

        assert_eq!(compare.to_quat(), quat);
        assert_eq!(quat.to_mat(), compare);
    }

    #[test]
    fn axis_angle() {
        let rotation = Mat3::rotation_y(45f32.to_radians()).to_quat();
        let (axis, angle) = rotation.to_axis_angle();

        let error = vec3_error(axis, Vec3::up());
        eprintln!("Axis Error: {}", error);
        assert!(error < 0.0001);

        let error = (angle.to_degrees() - 45.0).abs();
        eprintln!("Angle Error: {}", error);
        assert!(error < 0.001);
    }

    #[test]
    fn convert_axis_angle() {
        let rotation = Quat::axis_angle(
            Vec3::one().norm(),
            120f32.to_radians(),
        );

        let (axis, angle) = rotation.to_axis_angle();

        let error = vec3_error(axis, Vec3::one().norm());
        eprintln!("Axis Error: {}", error);
        assert!(error < 0.0001);

        let error = (angle.to_degrees() - 120.0).abs();
        eprintln!("Angle Error: {}", error);
        assert!(error < 0.0001);
    }

    #[test]
    fn convert_quat_mat3() {
        let half_sqrt = 0.5f32.sqrt();
        let quat = Quat::new(
            0.0,
            0.0,
            half_sqrt,
            half_sqrt,
        );

        let mat = Mat3::new(
            0.0, -1.0, 0.0,
            1.0,  0.0, 0.0,
            0.0,  0.0, 1.0,
        );

        assert_eq!(quat, mat.to_quat());
        assert_eq!(quat, quat.to_mat().to_quat());

        assert_eq!(mat, quat.to_mat());
        assert_eq!(mat, mat.to_quat().to_mat());
    }

    #[test]
    fn mul_quat_vec() {
        let quat = Quat::axis_angle(Vec3::up(), 7.1);
        let mat = Mat3::rotation_y(7.1);
        let vec = Vec3::new(1., 2., 3.);

        let error = vec3_error(quat.to_mat() * vec , mat * vec);
        eprintln!("Error: {}", error);
        assert!(error < 0.0001);
    }

    #[test]
    fn mul_quat() {
        assert!(Quat::id() * Quat::id() == Quat::id());

        let m1 = Mat3::rotation_x(4.0);
        let m2 = Mat3::rotation_y(1.0);
        let m3 = Mat3::rotation_z(-7.0);

        let q1 = m1.to_quat();
        let q2 = m2.to_quat();
        let q3 = m3.to_quat();

        let error = quat_error(
            (m1 * m2 * m3).to_quat(),
            q1 * q2 * q3,
        );

        assert!(error < 0.0001);
    }

    #[test]
    fn invert_quat() {
        assert!(Quat::id() * Quat::id().conjugate() == Quat::id());

        let q1 = Quat::axis_angle(Vec3::one(), 3.5);

        let error = quat_error(
            q1 * q1.conjugate(),
            Quat::id(),
        );

        assert!(error < 0.0001);

        let q2 = Quat::new(-1.0, -2.0, -3.0, -4.0).norm();
        let diff = q2 * q1.conjugate();

        let error = quat_error(
            diff * q1,
            q2,
        );

        assert!(error < 0.0001);
    }

    #[test]
    fn pow_quat() {
        let error = quat_error(
            Quat::id().pow(0.1),
            Quat::id(),
        );

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);

        let q1 = Quat::new(-4.0, -3.0, -2.0, -1.0).norm();

        let error = quat_error(
            q1.pow(0.5) * q1.pow(0.5),
            q1,
        );

        eprintln!("Error: {}", error);
        assert!(error < 0.1); // TODO
    }

    #[test]
    fn quat_eq() {
        let q1 = Quat::new(-1.0, -2.0, -3.0, -4.0).norm();

        assert_eq!(q1, q1);

        let q2 = Quat::new(1.0, 2.0, 3.0, 4.0).norm();

        assert_eq!(q2, q2);
        assert_eq!(q1, q2);

        let q3 = Quat::new(4.0, 3.0, 2.0, 1.0).norm();

        assert_ne!(q1, q3);
        assert_ne!(q2, q3);
    }

    #[test]
    fn norm_quat() {
        // Baseline
        let error = (
            Quat::id().norm().mag() - Quat::id().mag()
        ).abs();

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);

        let quat = Quat::new(-1., 3., 5., 0.);
        let error = (quat.norm().mag() - 1.).abs();

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);
    }

    /* Utility */

    fn mat4_error(a: Mat4, b: Mat4) -> f32 {
        let mut total = 0f32;

        {
            let mut error = |x: f32, y: f32| total += (x - y).abs();

            error(a.x0, b.x0);
            error(a.y0, b.y0);
            error(a.z0, b.z0);
            error(a.w0, b.w0);

            error(a.x1, b.x1);
            error(a.y1, b.y1);
            error(a.z1, b.z1);
            error(a.w1, b.w1);

            error(a.x2, b.x2);
            error(a.y2, b.y2);
            error(a.z2, b.z2);
            error(a.w2, b.w2);

            error(a.x3, b.x3);
            error(a.y3, b.y3);
            error(a.z3, b.z3);
            error(a.w3, b.w3);
        }

        total
    }

    fn mat3_error(a: Mat3, b: Mat3) -> f32 {
        let mut total = 0f32;

        {
            let mut error = |x: f32, y: f32| total += (x - y).abs();

            error(a.x0, b.x0);
            error(a.y0, b.y0);
            error(a.z0, b.z0);

            error(a.x1, b.x1);
            error(a.y1, b.y1);
            error(a.z1, b.z1);

            error(a.x2, b.x2);
            error(a.y2, b.y2);
            error(a.z2, b.z2);
        }

        total
    }

    fn vec3_error(a: Vec3, b: Vec3) -> f32 {
        let mut total = 0f32;

        {
            let mut error = |x: f32, y: f32| total += (x - y).abs();

            error(a.x, b.x);
            error(a.y, b.y);
            error(a.z, b.z);
        }

        total
    }

    fn quat_error(a: Quat, b: Quat) -> f32 {
        let mut total = 0f32;

        {
            // Two-norm
            let mut error = |x: f32, y: f32| total += (x - y) * (x - y);

            error(a.x, b.x);
            error(a.y, b.y);
            error(a.z, b.z);
            error(a.w, b.w);
        }

        total.sqrt()
    }
}
