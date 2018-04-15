extern crate voodoo_winit as vdw;

pub struct Manager {
    pub key_pressed_map: [(bool, bool); 12],
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
        }
    }

    pub fn get_key(&self, key: Key) -> bool {
        self.key_pressed_map[key as usize].1
    }

    pub fn get_key_down(&self, key: Key) -> bool {
        let vals = self.key_pressed_map[key as usize];
        !vals.0 && vals.1
    }

    pub fn get_key_up(&self, key: Key) -> bool {
        let vals = self.key_pressed_map[key as usize];
        vals.0 && !vals.1
    }
}
