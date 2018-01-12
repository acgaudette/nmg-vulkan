use alg;
use graphics;

#[cfg(debug_assertions)]
use render;

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
    pub fn add_ray(
        &mut self,
        start: alg::Vec3,
        ray: alg::Vec3,
        color: graphics::Color,
    ) {
        #[cfg(debug_assertions)] {
            self.lines.push(
                render::DebugLine::new(
                    alg::Line::new(start, start + ray),
                    color,
                )
            );
        }
    }

    #[allow(unused_variables)]
    pub fn add_axes(
        &mut self,
        center: alg::Vec3,
        size: f32,
    ) {
        #[cfg(debug_assertions)] {
            let scale = 0.5 * size;

            self.lines.push(
                render::DebugLine::new(
                    alg::Line::new(
                        center,
                        center + alg::Vec3::right() * scale,
                    ),
                    graphics::Color::red(),
                )
            );
            self.lines.push(
                render::DebugLine::new(
                    alg::Line::new(
                        center,
                        center + alg::Vec3::up() * scale,
                    ),
                    graphics::Color::green(),
                )
            );
            self.lines.push(
                render::DebugLine::new(
                    alg::Line::new(
                        center,
                        center + alg::Vec3::fwd() * scale,
                    ),
                    graphics::Color::cyan(),
                )
            );
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

