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
    let settings = ENGINE_CONFIG.section(Some(section))
        .unwrap_or_else(
        || panic!(
            "Failed to load section {}",
            section,
        )
    );

    settings.get(setting)
        .unwrap_or_else(
        || panic!(
            "Failed to load setting {}",
            setting,
        )
    ).clone()
}
