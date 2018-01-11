#![allow(dead_code)] // Library

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
            r: 1.,
            g: 0.,
            b: 0.,
        }
    }

    #[inline]
    pub fn green() -> Color {
        Color {
            r: 0.,
            g: 1.,
            b: 0.,
        }
    }

    #[inline]
    pub fn blue() -> Color {
        Color {
            r: 0.,
            g: 0.,
            b: 1.,
        }
    }
}
