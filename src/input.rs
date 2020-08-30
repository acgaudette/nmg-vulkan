extern crate voodoo_winit as vdw;

use alg;

pub const KEY_COUNT: usize = 23;

pub struct Manager {
    key_map: [KeyState; KEY_COUNT],
    pub rumbles_lo: Vec<f32>,
    pub rumbles_hi: Vec<f32>,

    pub cursor_coords: alg::Vec2,
    pub mouse_delta: alg::Vec2,

    pub joy_l: alg::Vec2,
    pub joy_r: alg::Vec2,
    pub trig_l: f32,
    pub trig_r: f32,
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
    Q,
    E,
    M,
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
            12 => Some(Key::Q),
            13 => Some(Key::E),
            14 => Some(Key::E),
            15 => Some(Key::Up),
            16 => Some(Key::Down),
            17 => Some(Key::Left),
            18 => Some(Key::Right),
            19 => Some(Key::Space),
            20 => Some(Key::Enter),
            21 => Some(Key::LCtrl),
            22 => Some(Key::LShift),
            _ => None,
        }
    }
}

impl Manager {
    pub fn new() -> Manager {
        Manager {
            key_map: [KeyState::default(); KEY_COUNT],
            rumbles_lo: Vec::with_capacity(4),
            rumbles_hi: Vec::with_capacity(4),
            cursor_coords: alg::Vec2::zero(),
            mouse_delta: alg::Vec2::zero(),
            joy_l: alg::Vec2::zero(),
            joy_r: alg::Vec2::zero(),
            trig_l: 0.0,
            trig_r: 0.0,
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

    /// Check if key was held this frame
    pub fn key_held(&self, key: Key) -> bool {
        self.key_map[key as usize].pressed
    }

    /// Check if key was pressed down this frame
    pub fn key_pressed(&self, key: Key) -> bool {
        let key_state = self.key_map[key as usize];
        !key_state.was_pressed && key_state.pressed
    }

    /// Check if key was released this frame
    pub fn key_released(&self, key: Key) -> bool {
        let key_state = self.key_map[key as usize];
        key_state.was_pressed && !key_state.pressed
    }

    /// Expose key pressed array, created on call
    pub fn pressed(&self) -> [bool; KEY_COUNT] {
        let mut keys = [false; KEY_COUNT];

        for i in 0..self.key_map.len() {
            keys[i] = self.key_map[i].pressed;
        }

        keys
    }

    /* Gamepad */

    pub fn mix_rumble(&self) -> (f32, f32) {
        (
            self.rumbles_lo.iter().sum::<f32>() / self.rumbles_lo.len() as f32,
            self.rumbles_hi.iter().sum::<f32>() / self.rumbles_hi.len() as f32,
        )
    }

    pub fn add_rumble_lo(&mut self, amt: f32) {
        debug_assert!(amt >= 0.0);
        debug_assert!(amt <= 1.0);
        self.rumbles_lo.push(amt);
    }

    pub fn add_rumble_hi(&mut self, amt: f32) {
        debug_assert!(amt >= 0.0);
        debug_assert!(amt <= 1.0);
        self.rumbles_hi.push(amt);
    }
}
