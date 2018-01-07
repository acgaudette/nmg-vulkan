#![allow(dead_code)] // Library

use std;

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

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }
}

#[derive(Clone, Copy, PartialEq)]
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

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
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

impl std::fmt::Display for Vec3 {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            out,
            "( {}, {}, {} )",
            self.x, self.y, self.z,
        )
    }
}

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat {

    /*
     * GLSL expects matrices in column-major order
     * Calculations below are formatted in row-major order
     */

    pub x0: f32, pub y0: f32, pub z0: f32, pub w0: f32,
    pub x1: f32, pub y1: f32, pub z1: f32, pub w1: f32,
    pub x2: f32, pub y2: f32, pub z2: f32, pub w2: f32,
    pub x3: f32, pub y3: f32, pub z3: f32, pub w3: f32,
}

impl Mat {
    pub fn new(
        x0: f32, x1: f32, x2: f32, x3: f32,
        y0: f32, y1: f32, y2: f32, y3: f32,
        z0: f32, z1: f32, z2: f32, z3: f32,
        w0: f32, w1: f32, w2: f32, w3: f32,
    ) -> Mat {
        Mat {
            x0, x1, x2, x3,
            y0, y1, y2, y3,
            z0, z1, z2, z3,
            w0, w1, w2, w3,
        }
    }

    #[inline]
    pub fn identity() -> Mat {
        Mat::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub fn translation(x: f32, y: f32, z: f32) -> Mat {
        Mat::new(
            1.0, 0.0, 0.0,   x,
            0.0, 1.0, 0.0,   y,
            0.0, 0.0, 1.0,   z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub fn translation_vec(translation: Vec3) -> Mat {
        Mat::translation(translation.x, translation.y, translation.z)
    }

    pub fn rotation_x(rad: f32) -> Mat {
        Mat::new(
            1.0,       0.0,        0.0, 0.0,
            0.0, rad.cos(), -rad.sin(), 0.0,
            0.0, rad.sin(),  rad.cos(), 0.0,
            0.0,       0.0,        0.0, 1.0,
        )
    }

    pub fn rotation_y(rad: f32) -> Mat {
        Mat::new(
             rad.cos(), 0.0, rad.sin(), 0.0,
                   0.0, 1.0,       0.0, 0.0,
            -rad.sin(), 0.0, rad.cos(), 0.0,
                   0.0, 0.0,       0.0, 1.0,
        )
    }

    pub fn rotation_z(rad: f32) -> Mat {
        Mat::new(
             rad.cos(), rad.sin(), 0.0, 0.0,
            -rad.sin(), rad.cos(), 0.0, 0.0,
                  0.0,        0.0, 1.0, 0.0,
                  0.0,        0.0, 0.0, 1.0,
        )
    }

    pub fn rotation(x: f32, y: f32, z: f32) -> Mat {
        Mat::rotation_x(x) * Mat::rotation_y(y) * Mat::rotation_z(z)
    }

    pub fn scale(x: f32, y: f32, z: f32) -> Mat {
        Mat::new(
              x, 0.0, 0.0, 0.0,
            0.0,   y, 0.0, 0.0,
            0.0, 0.0,   z, 0.0,
            0.0, 0.0, 0.0, 1.0
        )
    }

    pub fn scale_vec(scale: Vec3) -> Mat {
        Mat::scale(scale.x, scale.y, scale.z)
    }

    // Returns view matrix (inverted)
    pub fn look_at_view(position: Vec3, target: Vec3, up: Vec3) -> Mat {
        let forward = (target - position).norm();
        let right = up.cross(forward).norm();
        let up = forward.cross(right);

        // Transpose orthogonal matrix to get inverse
        let inverse_rotation = Mat::new(
              right.x,   right.y,   right.z, 0.0,
                 up.x,      up.y,      up.z, 0.0,
            forward.x, forward.y, forward.z, 0.0,
                  0.0,       0.0,       0.0, 1.0,
        );

        // Reverse position input
        let inverse_position = Mat::translation(
            -position.x,
            -position.y,
            -position.z,
        );

        inverse_rotation * inverse_position
    }

    // Input: vertical field of view, screen aspect ratio, near and far planes
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Mat {
        // Perspective scaling (rectilinear)
        let y_scale = 1. / (0.5 * fov).to_radians().tan();
        let x_scale = y_scale / aspect;

        // Fit into Vulkan clip space (0-1)
        let z_scale = 1. / (far - near);
        let z_offset = -near / (far - near);

        Mat::new(
            x_scale,      0.0,     0.0,      0.0,
                0.0, -y_scale,     0.0,      0.0, // Flip for Vulkan
                0.0,      0.0, z_scale, z_offset,
                0.0,      0.0,     1.0,      0.0, // Left-handed (scaling factor)
        )
    }

    fn transpose(self) -> Mat {
        Mat::new(
            self.x0, self.y0, self.z0, self.w0,
            self.x1, self.y1, self.z1, self.w1,
            self.x2, self.y2, self.z2, self.w2,
            self.x3, self.y3, self.z3, self.w3,
        )
    }
}

impl std::ops::Mul for Mat {
    type Output = Mat;

    // Naive matrix multiply
    fn mul(self, m: Mat) -> Mat {
        let x0 = self.x0 * m.x0 + self.x1 * m.y0 + self.x2 * m.z0 + self.x3 * m.w0;
        let x1 = self.x0 * m.x1 + self.x1 * m.y1 + self.x2 * m.z1 + self.x3 * m.w1;
        let x2 = self.x0 * m.x2 + self.x1 * m.y2 + self.x2 * m.z2 + self.x3 * m.w2;
        let x3 = self.x0 * m.x3 + self.x1 * m.y3 + self.x2 * m.z3 + self.x3 * m.w3;

        let y0 = self.y0 * m.x0 + self.y1 * m.y0 + self.y2 * m.z0 + self.y3 * m.w0;
        let y1 = self.y0 * m.x1 + self.y1 * m.y1 + self.y2 * m.z1 + self.y3 * m.w1;
        let y2 = self.y0 * m.x2 + self.y1 * m.y2 + self.y2 * m.z2 + self.y3 * m.w2;
        let y3 = self.y0 * m.x3 + self.y1 * m.y3 + self.y2 * m.z3 + self.y3 * m.w3;

        let z0 = self.z0 * m.x0 + self.z1 * m.y0 + self.z2 * m.z0 + self.z3 * m.w0;
        let z1 = self.z0 * m.x1 + self.z1 * m.y1 + self.z2 * m.z1 + self.z3 * m.w1;
        let z2 = self.z0 * m.x2 + self.z1 * m.y2 + self.z2 * m.z2 + self.z3 * m.w2;
        let z3 = self.z0 * m.x3 + self.z1 * m.y3 + self.z2 * m.z3 + self.z3 * m.w3;

        let w0 = self.w0 * m.x0 + self.w1 * m.y0 + self.w2 * m.z0 + self.w3 * m.w0;
        let w1 = self.w0 * m.x1 + self.w1 * m.y1 + self.w2 * m.z1 + self.w3 * m.w1;
        let w2 = self.w0 * m.x2 + self.w1 * m.y2 + self.w2 * m.z2 + self.w3 * m.w2;
        let w3 = self.w0 * m.x3 + self.w1 * m.y3 + self.w2 * m.z3 + self.w3 * m.w3;

        Mat::new(
            x0, x1, x2, x3,
            y0, y1, y2, y3,
            z0, z1, z2, z3,
            w0, w1, w2, w3,
        )
    }
}

impl std::ops::Mul<Vec3> for Mat {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3::new(
            self.x0 * vec.x + self.x1 * vec.y + self.x2 * vec.z,
            self.y0 * vec.x + self.y1 * vec.y + self.y2 * vec.z,
            self.z0 * vec.x + self.z1 * vec.y + self.z2 * vec.z,
        )
    }
}

impl std::fmt::Display for Mat {
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

#[derive(Clone, Copy)]
pub struct Quat {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Quat {
    pub fn new(vec: Vec3) -> Quat {
        Quat {
            x: vec.x,
            y: vec.y,
            z: vec.z,
            w: 0.,
        }
    }

    #[inline]
    pub fn identity() -> Quat {
        Quat {
            x: 0.,
            y: 0.,
            z: 0.,
            w: 1.,
        }
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

    pub fn to_mat(self) -> Mat {
        let this = self.norm();

        let x0 = 1. - 2. * this.y * this.y - 2. * this.z * this.z;
        let y0 = 2. * this.x * this.y - 2. * this.z * this.w;
        let z0 = 2. * this.x * this.z + 2. * this.y * this.w;

        let x1 = 2. * this.x * this.y + 2. * this.z * this.w;
        let y1 = 1. - 2. * this.x * this.x - 2. * this.z * this.z;
        let z1 = 2. * this.y * this.z - 2. * this.x * this.w;

        let x2 = 2. * this.x * this.z - 2. * this.y * this.w;
        let y2 = 2. * this.y * this.z + 2. * this.x * this.w;
        let z2 = 1. - 2. * this.x * this.x - 2. * this.y * this.y;

        Mat::new(
            x0, y0, z0, 0.,
            x1, y1, z1, 0.,
            x2, y2, z2, 0.,
            0., 0., 0., 1.,
        )
    }
}

impl std::ops::Mul<Vec3> for Quat {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        self.to_mat() * vec
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
            z: self.z + other.y,
            w: self.w + other.w,
        }
    }
}

#[cfg(test)]
mod tests {
    use alg::*;

    #[test]
    fn mul_mat() {
        let translation = Mat::translation(1.0, 2.0, 3.0);

        assert!(translation * Mat::identity() == translation);
        assert!(Mat::identity() * translation == translation);
    }

    #[test]
    fn mul_vec() {
        let vec = Vec3::new(9., -4., 0.);
        let scale = Mat::scale(-1., 3., 2.);

        assert!(Mat::identity() * vec == vec);
        assert!(scale * vec == Vec3::new(-9., -12., 0.));

        let mat = Mat::new(
            1., 1., 1., 0.,
            0., 1., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
        );

        assert!(mat * vec == Vec3::new(5., -4., 0.,));
    }

    #[test]
    fn convert_quat() {
        //
    }

    #[test]
    fn mul_quat() {
        //
    }

    #[test]
    fn norm_quat() {
        // Baseline
        let error = (
            Quat::identity().norm().mag() - Quat::identity().mag()
        ).abs();

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);

        let quat = Quat::new(Vec3::new(-1., 3., 5.));
        let error = (quat.norm().mag() - 1.).abs();

        eprintln!("Error: {}", error);
        assert!(error < 0.0001);
    }

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
}
