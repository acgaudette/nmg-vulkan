#![allow(dead_code)] // Library

extern crate image;
extern crate fnv;

use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;

// AngelCode .fnt format structs and classes
#[derive(Copy)]
pub struct Bmchar {
    pub id: i32,
	pub x: f32,
    pub y: f32,
	pub width: f32,
	pub height: f32,
	pub xoffset: f32,
	pub yoffset: f32,
	pub xadvance: f32,
	pub page: i32,
}

impl Clone for Bmchar {
    fn clone(&self) -> Bmchar { *self }
}

pub struct Font {
    pub pixels: Vec<u8>,
    pub common_font_data: CommonFont,
}

pub struct CommonFont {
    pub line_height: f32,
    pub base_width: f32,
    pub uv_width: f32,
    pub uv_height: f32,
    pub char_map: fnv::FnvHashMap<i32, Bmchar>,
}

fn get_value_from_pair<'a> (pair: &'a str) -> &'a str {
    let val:Vec<&str> = pair.split('=').collect();
    val[1]
}

fn parse_bmchar<'a>(full_str: &'a str) -> Bmchar {
    let mut iter = full_str.split_whitespace();
    iter.next(); // Skip
    let id = get_value_from_pair(iter.next().unwrap()).parse::<i32>().unwrap();
    let x = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
    let y = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
    let width = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
    let height = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
    let xoffset = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
    let yoffset = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
    let xadvance = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
    let page = get_value_from_pair(iter.next().unwrap()).parse::<i32>().unwrap();
    //let chnl = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
    Bmchar {
        id,
        x,
        y,
        width,
        height,
        xoffset,
        yoffset,
        xadvance,
        page,
    }
}

impl Font {
    pub fn new(
        font_data_path: &str
    ) -> Font {
        let file_handle = File::open(font_data_path).unwrap_or_else(
            |err| panic!(
                "Could not load font data file: \"{}\"", err
        ));
        let mut pixels = Vec::new();

        let file = BufReader::new(&file_handle);
        
        let mut char_map = fnv::FnvHashMap::with_capacity_and_hasher(
                255,
                Default::default(),
            );
            
        let mut uv_width = 0f32;
        let mut uv_height = 0f32;
        let mut base_width = 0f32;
        let mut line_height = 0f32;

        for line in file.lines() {
            let line_string = line.unwrap();
            let clone = line_string.clone();
            let mut iter = clone.split_whitespace();
            match iter.next() {
                Some("info") => {
                    // Currently do not care
                    //face="Cambria Math" size=27 bold=0 italic=0 charset="" unicode=0 stretchH=100 smooth=1 aa=1 padding=1,1,1,1 spacing=-2,-2
                    continue;
                },
                Some("common") => {
                    let clone2 = line_string.clone();
                    let mut iter2 = clone2.split_whitespace();
                    iter2.next();
                    line_height = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
                    base_width = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
                    uv_width = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
                    uv_height = get_value_from_pair(iter.next().unwrap()).parse::<f32>().unwrap();
                    continue;
                },
                Some("page") => {
                    iter.next(); //skip
                    let path = get_value_from_pair(iter.next().unwrap()).replace("\"", "");
                    let borrow = image::open(path).unwrap().as_rgba8().unwrap().clone();
                    pixels = borrow.into_raw();
                },
                Some("char") => {
                    let bmchar_instance = parse_bmchar(&line_string.clone());
                    char_map.insert(bmchar_instance.id, bmchar_instance);
                },
                _ => {
                    continue;
                },
            }
        }
        
        Font {
            pixels,
            common_font_data: CommonFont {
                line_height,
                base_width,
                uv_width,
                uv_height,
                char_map,
            }
        }
    }
}
