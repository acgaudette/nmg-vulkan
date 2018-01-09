use ini;

lazy_static! {
    pub static ref ENGINE_CONFIG: ini::Ini = {
        load_config("config.ini")
    };
}

pub fn load_config(filename: &str) -> ini::Ini {
    let conf = ini::Ini::load_from_file(filename).unwrap();
    conf
}
