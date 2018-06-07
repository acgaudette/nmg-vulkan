extern crate ini;

use std;
use std::error::Error;

lazy_static! {
    pub static ref ENGINE_CONFIG: ini::Ini =
        load_config("config.ini");
}

pub fn load_config(filename: &str) -> ini::Ini {
    match ini::Ini::load_from_file(filename) {
        Ok(result) => result,
        Err(e) => panic!("{}", e.description()),
    }
}

pub fn load_section_setting<T: std::str::FromStr> (
    config: &ini::Ini,
    section: &str,
    setting: &str,
) -> T
where <T as std::str::FromStr>::Err: std::error::Error {
    let settings = config.section(Some(section))
        .unwrap_or_else(
        || panic!(
            "Failed to load section \"{}\"",
            section,
        )
    );

    let raw = settings.get(setting)
        .unwrap_or_else(
        || panic!(
            "Failed to load setting \"{}\" in section \"{}\"",
            setting,
            section,
        )
    );

    match raw.parse::<T>() {
        Ok(result) => result,
        Err(e) => panic!("{}", e.description()),
    }
}

pub fn load_section<'a>(
    config: &'a ini::Ini,
    section: &str,
) -> &'a ini::ini::Properties {
    config.section(Some(section))
        .unwrap_or_else(
            || panic!(
                "Failed to load section \"{}\"",
                section,
            )
        )
}
