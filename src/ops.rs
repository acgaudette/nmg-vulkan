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

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Mat {
    pub x0: f32, pub x1: f32, pub x2: f32, pub x3: f32,
    pub y0: f32, pub y1: f32, pub y2: f32, pub y3: f32,
    pub z0: f32, pub z1: f32, pub z2: f32, pub z3: f32,
    pub w0: f32, pub w1: f32, pub w2: f32, pub w3: f32,
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
}
