#![allow(dead_code)] // Library

extern crate png;
extern crate fnv;

use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;

macro_rules! get_float_from_pair {
    ($( $x:expr ),* ) => {
        {$(
            get_value_from_pair(
                $x.next().expect("Found None value instead of Some")
            ).parse::<f32>().unwrap()
        )*}
    };
}

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

pub struct Data {
    pub pixels: Vec<u8>,
    pub common_font_data: CommonFont,
}

pub struct CommonFont {
    pub line_height: f32,
    pub base_height: f32,
    pub uv_width: f32,
    pub uv_height: f32,
    pub char_map: fnv::FnvHashMap<i32, Bmchar>,
    pub font_face: String,
    pub size: f32,
}

fn get_value_from_pair<'a> (pair: &'a str) -> &'a str {
    let val:Vec<&str> = pair.split('=').collect();
    val[1]
}

fn parse_bmchar<'a>(full_str: &'a str) -> Bmchar {
    let mut iter = full_str.split_whitespace();
    iter.next(); // Skip

    let id = get_float_from_pair!(iter) as i32;
    let x = get_float_from_pair!(iter);
    let y = get_float_from_pair!(iter);
    let width = get_float_from_pair!(iter);
    let height = get_float_from_pair!(iter);
    let xoffset = get_float_from_pair!(iter);
    let yoffset = get_float_from_pair!(iter);
    let xadvance = get_float_from_pair!(iter);
    let page = get_float_from_pair!(iter) as i32;
    //let chnl = get_float_from_pair!(iter);

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

impl Data {
    pub fn new(
        font_data_path: &str
    ) -> Data {
        let file_handle = File::open(font_data_path).unwrap_or_else(
            |err| panic!("Could not load font data file: \"{}\"", err)
        );

        let mut pixels = Vec::new();
        let file = BufReader::new(&file_handle);
        let mut char_map = fnv::FnvHashMap::with_capacity_and_hasher(
            255,
            Default::default(),
        );

        let mut uv_width = 0f32;
        let mut uv_height = 0f32;
        let mut base_height = 0f32;
        let mut line_height = 0f32;
        let mut font_face = "".into();
        let mut size = 0f32;
        let mut font_texture_name: String;

        for line in file.lines() {
            let line_string = line.unwrap_or_else(
                |err| panic!("Could not unwrap line: \"{}\"", err)
            );

            let clone = line_string.clone();
            let mut iter = clone.split_whitespace();

            match iter.next() {
                Some("info") => {
                    let pair = iter.next().expect("Could not find font face");
                    font_face = get_value_from_pair(pair).replace("\"", "");
                    println!("Reading font \"{}\"", font_face);
                    size = get_float_from_pair!(iter);
                    continue;
                },

                Some("common") => {
                    line_height = get_float_from_pair!(iter);
                    base_height = get_float_from_pair!(iter);
                    uv_width = get_float_from_pair!(iter);
                    uv_height = get_float_from_pair!(iter);
                    continue;
                },

                Some("page") => {
                    iter.next(); // Skip

                    let pair = iter.next().expect(
                        "Found None value instead of Some"
                    );

                    let mut path_vec: Vec<&str> = font_data_path.split("/").collect();
                    font_texture_name = get_value_from_pair(pair).replace("\"", "");

                    let last_idx = path_vec.len() - 1;
                    path_vec[last_idx] = &font_texture_name;

                    let path = path_vec.join("/");

                    let decoder = png::Decoder::new(
                        File::open(path).unwrap_or_else(
                            |err| panic!("Could not find file: \"{}\"", err)
                        )
                    );

                    let (info, mut reader) = decoder.read_info()
                        .unwrap_or_else(
                            |err| panic!("Could not decode path: \"{}\"", err)
                        );

                    let mut buf = vec![0; info.buffer_size()];
                    reader.next_frame(&mut buf).unwrap_or_else(
                        |err| panic!("Could not read frame: \"{}\"", err)
                    );

                    pixels = buf.clone();
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

        Data {
            pixels,
            common_font_data: CommonFont {
                line_height,
                base_height,
                uv_width,
                uv_height,
                char_map,
                font_face,
                size,
            }
        }
    }
}
