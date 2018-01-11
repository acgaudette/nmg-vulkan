#![allow(dead_code)] // Library

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
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
    fn red() -> Color {
        Color {
            r: 1.,
            g: 0.,
            b: 0.,
        }
    }

    #[inline]
    fn green() -> Color {
        Color {
            r: 0.,
            g: 1.,
            b: 0.,
        }
    }

    #[inline]
    fn blue() -> Color {
        Color {
            r: 0.,
            g: 0.,
            b: 1.,
        }
    }
}
