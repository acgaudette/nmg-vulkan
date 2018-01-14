#![allow(dead_code)] // Library

use std;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Color {
        Color {
            r,
            g,
            b,
        }
    }

    #[inline]
    pub fn red() -> Color {
        Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
        }
    }

    #[inline]
    pub fn green() -> Color {
        Color {
            r: 0.0,
            g: 1.0,
            b: 0.0,
        }
    }

    #[inline]
    pub fn blue() -> Color {
        Color {
            r: 0.0,
            g: 0.0,
            b: 1.0,
        }
    }

    #[inline]
    pub fn yellow() -> Color {
        Color {
            r: 1.0,
            g: 1.0,
            b: 0.0,
        }
    }

    #[inline]
    pub fn cyan() -> Color {
        Color {
            r: 0.0,
            g: 1.0,
            b: 1.0,
        }
    }

    #[inline]
    pub fn magenta() -> Color {
        Color {
            r: 1.0,
            g: 0.0,
            b: 1.0,
        }
    }

    #[inline]
    pub fn white() -> Color {
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        }
    }

    #[inline]
    pub fn orange() -> Color {
        Color {
            r: 1.0,
            g: 0.5,
            b: 0.0,
        }
    }

    #[inline]
    pub fn gray() -> Color {
        Color {
            r: 0.5,
            g: 0.5,
            b: 0.5,
        }
    }

    #[inline]
    pub fn lerp(a: Color, b: Color, t: f32) -> Color {
        a * (1.0 - t) + b * t
    }
}

impl std::ops::Add for Color {
    type Output = Color;

    fn add(self, other: Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl std::ops::Sub for Color {
    type Output = Color;

    fn sub(self, other: Color) -> Color {
        Color {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
        }
    }
}

impl std::ops::Mul<f32> for Color {
    type Output = Color;

    fn mul(self, scalar: f32) -> Color {
        Color {
            r: self.r * scalar,
            g: self.g * scalar,
            b: self.b * scalar,
        }
    }
}
