use render;
use font;

// Shared method for preparing and rendering text instances, whether 3d or not
pub fn prepare_text<T>(
    text_instance: &render::Text,
    font: &font::Data,
    vertex_ptr: *mut *mut T,
    idx_ptr: *mut *mut u32,
    idx_offset: &mut &mut u32,
    framebuffer_width:  u32,
    framebuffer_height: u32,
    text_instances: &mut Vec<render::TextInstance>,
) {
    let common_data = &font.common_font_data;
    let fb_w = framebuffer_width as f32;
    let fb_h = framebuffer_height as f32;
    let aspect_ratio = fb_w / fb_h;
    let uv_width = common_data.uv_width;
    let uv_height = common_data.uv_height;

    let x = 0f32;
    let y = 0f32;
    let z = 0f32;

    /*
      Determines scaling for text depending on type of text
      e.g. 3D text dependent on position versus label text
      viewable on screen at all times
     */
    let (char_width_scale, char_height_scale) = {
        if text_instance.is_2d {
            match text_instance.scale {
                render::TextScale::Pixel => {
                    let height = text_instance.scale_factor
                        / framebuffer_height as f32;
                    (height / aspect_ratio, height)
                },
                render::TextScale::Aspect => {
                    let height = text_instance.scale_factor
                        / common_data.line_height;
                    (height / aspect_ratio, height)
                },
            }
        } else {
            let char_measure = text_instance.scale_factor / common_data.base_width;
            (char_measure, char_measure)
        }
    };

    // Starting positions for current text instance being rendered
    let mut cursor_x = x; // NDC
    let x_starting = x;
    let mut cursor_y = y;

    let mut num_letters = 0;
    // Render quads for each individual character
    for c in text_instance.text.chars() {
        if c == '\n' {
            curr_line_start_y -= common_data.line_height * char_height_scale;
            curr_line_start_x = x_starting;
            continue;
        }

        if c == '\t' {
            // TODO: Replace with scalable constant
            curr_line_start_x += 10.;
            continue;
        }

        // Alias for character data from character map
        let char_data = &common_data.char_map[&(c as i32)];

        // UV coordinates
        let us = char_data.x / uv_width;
        let ue = (char_data.x + char_data.width) / uv_width;
        let vs = (char_data.y + char_data.height) / uv_height;
        let ve = char_data.y / uv_height;

        // Flip vertical UV coordinates
        let (u_start, u_end, v_start, v_end) = (
            us as f32,
            ue as f32,
            ve as f32,
            vs as f32,
        );

        // Send data to the GPU for the positions of the character quad
        let draw_x = cursor_x + char_data.xoffset * char_width_scale;
        let draw_y = cursor_y + char_data.yoffset * char_height_scale;

        let left_x = draw_x;
        let right_x = draw_x + char_data.width * char_width_scale;
        let top_y = draw_y;
        let bot_y = draw_y + char_data.height * char_height_scale;

        if text_instance.is_2d {
            let (top_left, bottom_right, bottom_left, top_right,) =
                (render::FontVertex_2d::new_raw(
                        left_x,
                        top_y,
                        u_start,
                        v_start,
                    ),
                render::FontVertex_2d::new_raw(
                        right_x,
                        bottom_y,
                        u_end,
                        v_end,
                    ),
                render::FontVertex_2d::new_raw(
                        left_x,
                        bottom_y,
                        u_start,
                        v_end,
                    ),
                render::FontVertex_2d::new_raw(
                        right_x,
                        top_y,
                        u_end,
                        v_start,
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
        } else {
            let (top_left, bottom_right, bottom_left, top_right,) =
                (render::FontVertex_3d::new_raw(
                    left_x,
                    top_y,
                    z,
                    u_start,
                    v_start,
                    ),
                render::FontVertex_3d::new_raw(
                        right_x,
                        bottom_y,
                        z,
                        u_end,
                        v_end,
                    ),
                render::FontVertex_3d::new_raw(
                        left_x,
                        bottom_y,
                        z,
                        u_start,
                        v_end,
                    ),
                render::FontVertex_3d::new_raw(
                        right_x,
                        top_y,
                        z,
                        u_end,
                        v_start,
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
        };

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

        curr_line_start_x += curr_x_advance;
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
}
