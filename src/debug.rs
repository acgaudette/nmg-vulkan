use alg;
use render;
use graphics;

pub struct Handler {
    #[cfg(debug_assertions)]
    pub lines: Vec<render::DebugLine>,
}

impl Handler {
    pub fn new() -> Handler {
        #[cfg(debug_assertions)] {
            Handler {
                lines: Vec::new(),
            }
        }

        #[cfg(not(debug_assertions))] { Handler { } }
    }

    #[allow(unused_variables)]
    pub fn add_line(
        &mut self,
        line: alg::Line,
        color: graphics::Color,
    ) {
        #[cfg(debug_assertions)] {
            self.lines.push(render::DebugLine::new(line, color));
        }
    }

    #[allow(unused_variables)]
    pub fn add_cross(
        &mut self,
        center: alg::Vec3,
        size: f32,
        color: graphics::Color,
    ) {
        #[cfg(debug_assertions)] {
            let scale = 0.5 * size;

            let first = alg::Line::new(
                center + alg::Vec3::new( scale, 0.,  scale),
                center + alg::Vec3::new(-scale, 0., -scale),
            );

            let second = alg::Line::new(
                center + alg::Vec3::new( scale, 0., -scale),
                center + alg::Vec3::new(-scale, 0.,  scale),
            );

            self.lines.push(render::DebugLine::new(first, color));
            self.lines.push(render::DebugLine::new(second, color));
        }
    }

    pub fn clear_lines(&mut self) {
        #[cfg(debug_assertions)] {
            self.lines.clear();
        }
    }
}

