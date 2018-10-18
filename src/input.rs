extern crate voodoo_winit as vdw;

use alg;

pub const KEY_COUNT: usize = 20;

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
    Z,
    X,
    C,
    V,
    Up,
    Down,
    Left,
    Right,
    Space,
    Enter,
    LCtrl,
    LShift,
}

impl Key {
    pub fn from_usize(input: usize) -> Option<Key> {
        match input {
            0 => Some(Key::W),
            1 => Some(Key::A),
            2 => Some(Key::S),
            3 => Some(Key::D),
            4 => Some(Key::H),
            5 => Some(Key::J),
            6 => Some(Key::K),
            7 => Some(Key::L),
            8 => Some(Key::Z),
            9 => Some(Key::X),
            10 => Some(Key::C),
            11 => Some(Key::V),
            12 => Some(Key::Up),
            13 => Some(Key::Down),
            14 => Some(Key::Left),
            15 => Some(Key::Right),
            16 => Some(Key::Space),
            17 => Some(Key::Enter),
            18 => Some(Key::LCtrl),
            19 => Some(Key::LShift),
            _ => None,
        }
    }
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

    pub(crate) fn increment_key_states(&mut self) {
        for key_state in &mut self.key_map {
            key_state.was_pressed = key_state.pressed;
        }
    }

    pub(crate) fn set_key_pressed(&mut self, key: usize, pressed: bool) {
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

    // Expose key pressed array, created on call
    pub fn pressed(&self) -> [bool; KEY_COUNT] {
        let mut keys = [false; KEY_COUNT];

        for i in 0..self.key_map.len() {
            keys[i] = self.key_map[i].pressed;
        }

        keys
    }
}
