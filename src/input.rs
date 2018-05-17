extern crate voodoo_winit as vdw;

use alg;

pub struct Manager {
    pub key_pressed_map: [(bool, bool); 12],
    pub cursor_coords: alg::Vec2,
    pub mouse_delta: alg::Vec2,
}

#[derive(Clone, Copy, Debug, Eq)]
pub enum Key {
    W = 0,
    A = 1,
    S = 2,
    D = 3,
    Up = 4,
    Down = 5,
    Left = 6,
    Right = 7,
    Space = 8,
    Enter = 9,
    LCtrl = 10,
    LShift = 11,
}

impl PartialEq for Key {
    fn eq(&self, other: &Key) -> bool {
        self == other
    }
}

impl Manager {
    pub fn new() -> Manager {
        Manager {
            key_pressed_map: [(false, false); 12],
            cursor_coords: alg::Vec2::zero(),
            mouse_delta: alg::Vec2::zero(),
        }
    }

    pub fn key_held(&self, key: Key) -> bool {
        self.key_pressed_map[key as usize].1
    }

    pub fn key_pressed(&self, key: Key) -> bool {
        let vals = self.key_pressed_map[key as usize];
        !vals.0 && vals.1
    }

    pub fn key_released(&self, key: Key) -> bool {
        let vals = self.key_pressed_map[key as usize];
        vals.0 && !vals.1
    }

    pub fn cursor_coords(&self) -> alg::Vec2 {
        self.cursor_coords
    }

    pub fn mouse_delta(&self) -> alg::Vec2 {
        self.mouse_delta
    }
}
