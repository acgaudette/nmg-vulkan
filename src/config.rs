use ini;
use std::string::String;

lazy_static! {
    pub static ref ENGINE_CONFIG: ini::Ini = {
        load_config("config.ini")
    };
}

pub fn load_config(filename: &str) -> ini::Ini {
    let conf_opt = ini::Ini::load_from_file(filename);
    assert!(conf_opt.is_ok());
    conf_opt.unwrap()
}

pub fn load_section_setting(section: &str, setting: &str) -> String {
    let section_opt = ENGINE_CONFIG.section(Some(section));
    assert!(section_opt.is_some());
    let settings = section_opt.unwrap();
    
    let setting_opt = settings.get(setting);
    assert!(setting_opt.is_some());
    setting_opt.unwrap().clone()
}