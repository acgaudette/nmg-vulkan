use alg;
use render;
use config;

pub struct QuadProperties {
    left_x: f32,
    right_x: f32,
    top_y: f32,
    bot_y: f32,
    u_start: f32,
    u_end: f32,
    v_start: f32,
    v_end: f32,
}

#[derive(Clone, PartialEq)]
pub enum TextScale {
    Pixel,
    Aspect,
}

#[derive(Clone)]
pub enum TextAlign { Left, Center, Right }

#[derive(Clone)]
pub struct Text {
    pub text: String,
    pub position: alg::Vec3,
    pub align: TextAlign,
    pub scale: TextScale,
    pub scale_factor: f32,
    pub raw_width: f32,
    pub raw_height: f32,
    pub anchor_x: f32,
    pub anchor_y: f32,
    pub is_2d: bool,
}

impl Text {
    pub fn empty_2d_instance() -> Text {
        Text {
            text: "".to_string(),
            position: alg::Vec3::zero(),
            align: TextAlign::Center,
            scale: TextScale::Pixel,
            scale_factor: 1f32,
            raw_width: 0f32,
            raw_height: 0f32,
            anchor_x: 0f32,
            anchor_y: 0f32,
            is_2d: true,
        }
    }

    pub fn empty_3d_instance() -> Text {
        Text {
            text: "".to_string(),
            position: alg::Vec3::zero(),
            align: TextAlign::Right,
            scale: TextScale::Aspect,
            scale_factor: 1f32,
            raw_width: 0f32,
            raw_height: 0f32,
            anchor_x: 0f32,
            anchor_y: 0f32,
            is_2d: false,
        }
    }

    pub fn set_str(&mut self, str: &str) {
        self.text = str.to_string();
        let common_data = &config::FONT_DATA.common_font_data;
        let mut width = 0f32;
        let mut height = common_data.line_height;
        let mut width_counter = 0f32;
        for c in str.chars() {
            if c == '\n' {
                height += common_data.line_height;
                if width_counter > width {
                    width = width_counter;
                }
                width_counter = 0.0;
                continue;
            }
    
            if c == '\t' {
                width_counter += common_data.size;
                continue;
            }
    
            // Alias for character data from character map
            let char_data = &common_data.char_map[&(c as i32)];
            width_counter += char_data.xoffset + char_data.width;
        }
        if width_counter > width {
            width = width_counter;
        }
        self.raw_width = width;
        self.raw_height = height;
    }
}

// label text viewable on screen at all times
pub fn get_2d_char_scale(
    text_instance: &Text,
    framebuffer_height: u32,
) -> f32 {
    let common_data = &config::FONT_DATA.common_font_data;
    // Double for 2D to scale into (-1, 1) NDC
    return 2.0 * match text_instance.scale {
        TextScale::Pixel => {
            text_instance.scale_factor / framebuffer_height as f32
        },
        TextScale::Aspect => {
            text_instance.scale_factor / common_data.line_height
        },
    };
}

// 3D text dependent on position
pub fn get_3d_char_scale(
    text_instance: &Text,
) -> f32{
    let common_data = &config::FONT_DATA.common_font_data;
    return text_instance.scale_factor / common_data.line_height;
}

// Shared method for preparing and rendering text instances, whether 3d or not
pub fn prepare_text<T>(
    text_instance: &Text,
    idx_ptr: *mut *mut u32,
    idx_offset: &mut &mut u32,
    text_instances: &mut Vec<render::TextInstance>,
    char_scale: f32,
)-> Vec<QuadProperties> {
    let font = &config::FONT_DATA;
    let common_data = &font.common_font_data;
    let uv_width = common_data.uv_width;
    let uv_height = common_data.uv_height;

    // Starting positions for current text instance being rendered
    let mut cursor_x = 0.0; // NDC
    let mut cursor_y = 0.0;
    let perspective_scale = if text_instance.is_2d { 1.0 } else { -1.0 };
    let mut num_letters = 0;

    let mut quads: Vec<QuadProperties> = vec![];

    // Render quads for each individual character
    for c in text_instance.text.chars() {
        if c == '\n' {
            cursor_y += common_data.line_height * char_scale
                * perspective_scale;
            cursor_x = 0.0;
            continue;
        }

        if c == '\t' {
            cursor_x += common_data.size * char_scale;
            continue;
        }

        // Alias for character data from character map
        let char_data = &common_data.char_map[&(c as i32)];

        // UV coordinates
        let us = char_data.x / uv_width;
        let ue = (char_data.x + char_data.width) / uv_width;
        let vs = char_data.y / uv_height;
        let ve = (char_data.y + char_data.height) / uv_height;

        // Flip vertical UV coordinates
        let (u_start, u_end, v_start, v_end) = (
            us as f32,
            ue as f32,
            vs as f32,
            ve as f32,
        );

        // Calculate data to be sent to GPU for the 
        // positions of the character quad
        let draw_x = cursor_x + ((char_data.xoffset -
            (text_instance.raw_width * text_instance.anchor_x)) * char_scale);
        let left_x = draw_x;
        let right_x = draw_x + char_data.width * char_scale;

        // Note: the Y offset will center the characters within the line,
        // making them appear as if they are rotating around a distant point.
        let draw_y = cursor_y + ((char_data.yoffset -
            (text_instance.raw_height * text_instance.anchor_y)) *
                char_scale * perspective_scale);
        let top_y = draw_y;
        let bot_y = draw_y + char_data.height * char_scale
            * perspective_scale;
        quads.push(
            QuadProperties {
                left_x,
                right_x,
                top_y,
                bot_y,
                u_start,
                u_end,
                v_start,
                v_end,
            }
        );
        /*
            top_left, top_right, bottom_right,
            top_left, bottom_right, bottom_left,
        */
        for index in [
            0, 1, 3,
            0, 3, 2,
        ].iter() {
            unsafe {
                (**idx_ptr) = *index as u32 + **idx_offset as u32;
                (*idx_ptr) = (*idx_ptr).offset(1);
            }
        }

        cursor_x += char_data.xadvance * char_scale;

        num_letters += 1;
        **idx_offset = **idx_offset + 4;
    }

    let mut text_instance = render::TextInstance {
        index_count: num_letters as u32 * 6 as u32,
        index_offset: 0,
        vertex_count: num_letters as usize * 4 as usize,
        vertex_offset: 0,
    };

    if text_instances.len() > 0 {
        let last_instance = text_instances.last().unwrap();

        text_instance.index_offset = last_instance.index_offset as u32
            + last_instance.index_count;
        text_instance.vertex_offset = last_instance.vertex_count as i32
            + last_instance.vertex_offset as i32;
    }

    text_instances.push(text_instance);
    return quads;
}

pub fn write_vertices_2d<T>(
    quads_props: Vec<QuadProperties>,
    vertex_ptr: *mut *mut T,
) {
    for QuadProperties {
        left_x,
        right_x,
        top_y,
        bot_y,
        u_start,
        u_end,
        v_start,
        v_end,
    } in quads_props.iter() {
        let (top_left, bottom_right, bottom_left, top_right) =
            (render::FontVertex_2d::new_raw( // Top left
                    *left_x,
                    *top_y,
                    *u_start,
                    *v_start,
                ),
            render::FontVertex_2d::new_raw( // Bottom right
                    *right_x,
                    *bot_y,
                    *u_end,
                    *v_end,
                ),
            render::FontVertex_2d::new_raw( // Bottom left
                    *left_x,
                    *bot_y,
                    *u_start,
                    *v_end,
                ),
            render::FontVertex_2d::new_raw( // Top right
                    *right_x,
                    *top_y,
                    *u_end,
                    *v_start,
                ));
        for vertex in [
            top_left, top_right, bottom_left, bottom_right,
        ].iter() {
            unsafe {
                let vp = vertex_ptr as *mut *mut render::FontVertex_2d;
                (**vp) = *vertex;
                (*vertex_ptr) = (*vp).offset(1) as *mut T;
            }
        }
    }
}

pub fn write_vertices_3d<T>(
    quads_props: Vec<QuadProperties>,
    vertex_ptr: *mut *mut T,
) {
    for QuadProperties {
        left_x,
        right_x,
        top_y,
        bot_y,
        u_start,
        u_end,
        v_start,
        v_end,
    } in quads_props.iter() {
        let (top_left, bottom_right, bottom_left, top_right) =
            (render::FontVertex_3d::new_raw(
                    *left_x,
                    *top_y,
                    0.0,
                    *u_start,
                    *v_start,
                ),
            render::FontVertex_3d::new_raw(
                    *right_x,
                    *bot_y,
                    0.0,
                    *u_end,
                    *v_end,
                ),
            render::FontVertex_3d::new_raw(
                    *left_x,
                    *bot_y,
                    0.0,
                    *u_start,
                    *v_end,
                ),
            render::FontVertex_3d::new_raw(
                    *right_x,
                    *top_y,
                    0.0,
                    *u_end,
                    *v_start,
                ));
        for vertex in [
            top_left, top_right, bottom_left, bottom_right,
        ].iter() {
            unsafe {
                let vp = vertex_ptr as *mut *mut render::FontVertex_3d;
                (**vp) = *vertex;
                (*vertex_ptr) = (*vp).offset(1) as *mut T;
            }
        }
    }
}