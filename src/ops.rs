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
