use std;

fn inverse_sqrt(x: f32) -> f32 {
    let cast: u32 = unsafe {
        std::mem::transmute(x)
    };

    let guess = 0x5f3759df - (cast >> 1);
    let guess: f32 = unsafe {
        std::mem::transmute(guess)
    };

    guess * (1.5 - (x * 0.5) * guess * guess)
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

#[derive(Clone, Copy)]
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
}

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat {

    /*
     * GLSL expects matrices in column-major order
     * Calculations below are formatted in row-major order
     * (for readability)
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
            0.0, 1.0, 0.0,  -y, // Left-handed (correct Vulkan axis)
            0.0, 0.0, 1.0,   z,
            0.0, 0.0, 0.0, 1.0,
        )
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
            rad.cos(), -rad.sin(), 0.0, 0.0,
            rad.sin(),  rad.cos(), 0.0, 0.0,
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

    // Input: vertical field of view, screen aspect ratio, near and far planes
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Mat {
        // Perspective scaling (rectilinear)
        let y_scale = 1. / (0.5 * fov).to_radians().tan();
        let x_scale = y_scale / aspect;

        // Fit into Vulkan clip space (0-1)
        let z_scale = 1. / (far - near);
        let z_offset = -near / (far - near);

        Mat::new(
            x_scale,     0.0,     0.0,      0.0,
                0.0, y_scale,     0.0,      0.0,
                0.0,     0.0, z_scale, z_offset,
                0.0,     0.0,     1.0,      0.0, // Left-handed (scaling factor)
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

#[cfg(test)]
mod tests {
    use alg::*;

    #[test]
    fn mult_mat() {
        let translation = Mat::translation(1.0, 2.0, 3.0);

        assert!(translation * Mat::identity() == translation);
        assert!(Mat::identity() * translation == translation);
    }
}
