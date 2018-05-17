extern crate voodoo_winit as vdw;

use alg;

pub struct Manager {
    key_map: [KeyState; 12],
    cursor_coords: alg::Vec2,
    mouse_delta: alg::Vec2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyState {
    pub was_pressed: bool,
    pub pressed: bool,
}

impl Default for KeyState {
    fn default() -> KeyState {
        KeyState {
            was_pressed: false,
            pressed: false,
        }
    }
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
            key_map: [KeyState::default(); 12],
            cursor_coords: alg::Vec2::zero(),
            mouse_delta: alg::Vec2::zero(),
        }
    }

    /* Key state */

    pub fn key_held(&self, key: Key) -> bool {
        self.key_map[key as usize].pressed
    }

    pub fn key_pressed(&self, key: Key) -> bool {
        let key_state = self.key_map[key as usize];
        !key_state.was_pressed && key_state.pressed
    }

    pub fn key_released(&self, key: Key) -> bool {
        let key_state = self.key_map[key as usize];
        key_state.was_pressed && !key_state.pressed
    }

    /* Mouse */

    pub fn cursor_coords(&self) -> alg::Vec2 {
        self.cursor_coords
    }

    pub fn mouse_delta(&self) -> alg::Vec2 {
        self.mouse_delta
    }
}
