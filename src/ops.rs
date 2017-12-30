#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Mat {
    x0: f32, x1: f32, x2: f32, x3: f32,
    y0: f32, y1: f32, y2: f32, y3: f32,
    z0: f32, z1: f32, z2: f32, z3: f32,
    w0: f32, w1: f32, w2: f32, w3: f32,
}
