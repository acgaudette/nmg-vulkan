use ini::Ini;

lazy_static! {
    pub static ref UNIVERSAL_CONFIG: Ini = {
        load_config("config.ini")
    };
}

pub fn load_config(filename: &str) -> Ini {
    let conf = Ini::load_from_file(filename).unwrap();
    conf
}