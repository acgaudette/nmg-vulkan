extern crate fnv;

use render;
use entity;

use font::*;

// Shared method for preparing and rendering text instances, whether 3d or not
pub fn prepare_text(
    text_instance: &render::Text,
    font: &Font,
    ptr: *mut *mut render::FontData,
    framebuffer_width:  u32,
    framebuffer_height: u32,
    num_letters: *mut u64,
) {
    let common_data = &font.common_font_data;
    let fb_w = framebuffer_width as f32;
    let fb_h = framebuffer_height as f32;
    let aspect_ratio = fb_w / fb_h;
    let uv_width = common_data.uv_width;
    let uv_height = common_data.uv_height;

    let x = text_instance.position.x;
    let y = text_instance.position.y;
    let z = text_instance.position.z;

    /*
      Determines scaling for text depending on type of text
      e.g. 3D text dependent on position versus label text
      viewable on screen at all times
     */

    let (char_width_scale, char_height_scale) = {
        if text_instance.is_2d {
            if text_instance.scale == render::TextScale::PixelScale {
                (text_instance.scale_factor / fb_w,
                 text_instance.scale_factor / fb_h)
            } else {
                (text_instance.scale_factor / (aspect_ratio * common_data.base_width),
                 text_instance.scale_factor / common_data.base_width)
            }
        } else {
            (text_instance.scale_factor / common_data.base_width,
             text_instance.scale_factor / common_data.base_width)
        }
    };

    // Starting positions for current text instance being rendered
    let mut curr_line_start_x = x;
    let mut curr_line_start_y = y + common_data.line_height * char_height_scale;

    // Render quads for each individual character
    for c in text_instance.text.chars() {
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
            ve as f32,
            vs as f32,
        );

        // Apply transformations based on scale factor and divisors
        let xoffset =
            char_data.xoffset * char_width_scale * text_instance.scale_factor;
        curr_line_start_x += xoffset;

        let yoffset =
            char_data.yoffset * char_height_scale * text_instance.scale_factor;

        // Send data to the GPU for the positions of the character quad
        let curr_x_advance = char_data.xadvance * char_width_scale;
        let left_x = curr_line_start_x;
        let right_x = curr_line_start_x + curr_x_advance;
        let bottom_y = curr_line_start_y - yoffset;
        let top_y = bottom_y - (char_height_scale * char_data.height);

        let top_left = render::FontData::new_raw(
            left_x,
            top_y,
            z,
            u_start,
            v_start,
        );

        let bottom_right = render::FontData::new_raw(
            right_x,
            bottom_y,
            z,
            u_end,
            v_end,
            );

        let bottom_left = render::FontData::new_raw(
            left_x,
            bottom_y,
            z,
            u_start,
            v_end,
        );

        let top_right = render::FontData::new_raw(
            right_x,
            top_y,
            z,
            u_end,
            v_start,
        );

        for vertex in [
            top_left, top_right, bottom_right,
            top_left, bottom_right, bottom_left,
        ].iter() {
            unsafe {
                (**ptr) = *vertex;
                (*ptr) = (*ptr).offset(1);
            }
        }

        curr_line_start_x += curr_x_advance;
        unsafe { (*num_letters) += 1; }
    }
}
