extern crate voodoo_winit as vdw;

use alg;

pub const KEY_COUNT: usize = 16;

pub struct Manager {
    key_map: [KeyState; KEY_COUNT],
    pub cursor_coords: alg::Vec2,
    pub mouse_delta: alg::Vec2,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    W,
    A,
    S,
    D,
    H,
    J,
    K,
    L,
    Up,
    Down,
    Left,
    Right,
    Space,
    Enter,
    LCtrl,
    LShift,
}


impl Manager {
    pub fn new() -> Manager {
        Manager {
            key_map: [KeyState::default(); KEY_COUNT],
            cursor_coords: alg::Vec2::zero(),
            mouse_delta: alg::Vec2::zero(),
        }
    }

    /* Key states */

    pub fn increment_key_states(&mut self) {
        for key_state in &mut self.key_map {
            key_state.was_pressed = key_state.pressed;
        }
    }

    pub fn set_key_pressed(&mut self, key: usize, pressed: bool) {
        self.key_map[key].pressed = pressed;
    }

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
}
