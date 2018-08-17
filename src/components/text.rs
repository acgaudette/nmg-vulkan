extern crate fnv;

use render;
use entity;

use font::*;

/**
 * Shared method for preparing and rendering text isntances, whether 3d or not
 */
pub fn prepare_bitmap_text(
    instances: &mut fnv::FnvHashMap<entity::Handle, render::Text>,
    font_data: &Font,
    ptr: *mut *mut render::FontData,
    framebuffer_width:  u32,
    framebuffer_height: u32,
    num_letters: *mut u64,
) {
    let common_data = &font_data.common_font_data;
    let fb_w = framebuffer_width as f32;
    let fb_h = framebuffer_height as f32;
    let uv_width = common_data.uv_width;
    let uv_height = common_data.uv_height;
    
    //Iterating through text instances for rendering
    for (_, text_instance) in (*instances).iter() {
        let x = text_instance.position.x;
        let y = text_instance.position.y;
        let z = text_instance.position.z;

        /*
          Determines scaling for text depending on type of text
          e.g. 3D text dependent on position versus label text
          viewable on screen at all times
         */
        let mut char_w = text_instance.scale_factor;
        let mut char_h = text_instance.scale_factor;

        if text_instance.scale == render::TextScale::ScreenSpace {
            char_w /= fb_w;
            char_h /= fb_h;
        } else if text_instance.scale == render::TextScale::ScaleFactor {
            if text_instance.is_2d {
                char_w /= fb_w;
                char_h /= fb_h;
            } else {
                char_h /= common_data.line_height;
            }
        }

        /*
          Determines appropriate horizontal and vertical divisors dependent
          on text type
         */
        let xo_divisor =
            if text_instance.scale == render::TextScale::ScreenSpace {
                fb_w
            } else {
                if text_instance.is_2d {
                    fb_w
                } else {
                    common_data.base_width
                }
            };

        let yo_divisor = 
            if text_instance.scale == render::TextScale::ScreenSpace {
                fb_h
            } else {
                if text_instance.is_2d {
                    fb_h
                } else {
                    common_data.line_height
                }
            };

        // Starting positions for current text instance being rendered
        let mut curr_line_start_x = x;
        let mut curr_line_start_y = y + common_data.line_height / yo_divisor;

        // Rendering quads for each individual character
        for c in text_instance.text.chars() {
            // Aliasing for character data from character map
            let char_data = &common_data.char_map[&(c as i32)];
            // UV coordinates
            let us = char_data.x / uv_width;
            let ue = (char_data.x + char_data.width) / uv_width;
            let vs = char_data.y / uv_height;
            let ve = (char_data.y + char_data.height) / uv_height;

            // Flipping vertical UV coordinates
            let (u_start, u_end, v_start, v_end) = {
                (us, ue, ve, vs)
            };

            // Applying transformations based on scale factor and divisors
            let xoffset =
                (char_data.xoffset / xo_divisor) * text_instance.scale_factor;
            curr_line_start_x += xoffset;

            let yoffset =
                (char_data.yoffset / yo_divisor) * text_instance.scale_factor;
            let curr_line_start_y_box = curr_line_start_y - yoffset;

            let mut height_of_char = char_data.height / yo_divisor;
            let normalize_glyphs = 
                text_instance.scale == render::TextScale::ScreenSpace ||
                !text_instance.is_2d;

            // Sends data to the GPU for the positions of the character quad
            unsafe {
                let mut char_data_width = text_instance.scale_factor * 
                    (if normalize_glyphs { 1f32 }
                     else { char_data.width as f32 });

                let mut height_scale = if normalize_glyphs { 1f32 }
                     else { char_data.height as f32 };

                let mut curr_x_advance = char_data.xadvance * char_w * char_data_width;
                let mut _x = curr_line_start_x;
                let mut _y = curr_line_start_y_box - (height_scale * char_h + height_of_char);
                let mut _s = u_start as f32;
                let mut _t = v_start as f32;

                let top_left = render::FontData::new_raw(
                    _x,
                    _y,
                    z,
                    _s,
                    _t,
                    );
                (**ptr) = top_left;
                (*ptr) = (*ptr).offset(1);

                if !text_instance.is_2d {
                    curr_x_advance /= common_data.base_width;
                }
                _x = curr_line_start_x + curr_x_advance;
                _y = curr_line_start_y_box;
                _s = u_end;
                _t = v_end;

                let bottom_right = render::FontData::new_raw(
                    _x,
                    _y,
                    z,
                    _s,
                    _t,
                    );
                (**ptr) = bottom_right;
                (*ptr) = (*ptr).offset(1);

                // Bottom left
                _x = curr_line_start_x;
                _y = curr_line_start_y_box;
                _s = u_start;
                _t = v_end;
                send_point_to_buffer(
                    ptr,
                    _x,
                    _y,
                    z,
                    _s,
                    _t,
                    );

                (**ptr) = top_left;
                (*ptr) = (*ptr).offset(1);

                _x = curr_line_start_x + curr_x_advance;
                _y = curr_line_start_y_box - (height_scale * char_h + height_of_char);
                _s = u_end;
                _t = v_start;
                
                send_point_to_buffer(
                    ptr,
                    _x,
                    _y,
                    z,
                    _s,
                    _t,
                );

                (**ptr) = bottom_right;
                (*ptr) = (*ptr).offset(1);

                curr_line_start_x += curr_x_advance;
                (*num_letters) += 1;
            }
        }
    }
}

// Helper function for reducing amount of code written; may consider rewriting
unsafe fn send_point_to_buffer(
    mapped: *mut *mut render::FontData,
    x: f32,
    y: f32,
    z: f32,
    s: f32,
    t: f32,
) {
    (**mapped) = render::FontData::new_raw(
        x,
        y,
        z,
        s,
        t,
        );
    (*mapped) = (*mapped).offset(1);
}
