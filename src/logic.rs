use render;

pub fn init() -> Vec<render::ModelData> {
    /* Demo model data */

    let pyramid = render::ModelData::new(
        vec![
            render::Vertex::new( 0.0,  0.5, 0.5, 1., 0., 0.),
            render::Vertex::new( 0.5, -0.5, 0.0, 1., 0., 0.),
            render::Vertex::new(-0.5, -0.5, 0.0, 1., 0., 0.),

            render::Vertex::new( 0.0,  0.5, 0.5, 0., 1., 0.),
            render::Vertex::new( 0.5, -0.5, 1.0, 0., 1., 0.),
            render::Vertex::new( 0.5, -0.5, 0.0, 0., 1., 0.),

            render::Vertex::new( 0.0,  0.5, 0.5, 0., 0., 1.),
            render::Vertex::new(-0.5, -0.5, 1.0, 0., 0., 1.),
            render::Vertex::new( 0.5, -0.5, 1.0, 0., 0., 1.),

            render::Vertex::new( 0.0,  0.5, 0.5, 1., 1., 0.),
            render::Vertex::new(-0.5, -0.5, 0.0, 1., 1., 0.),
            render::Vertex::new(-0.5, -0.5, 1.0, 1., 1., 0.),

            render::Vertex::new( 0.5, -0.5, 0.0, 1., 0., 0.),
            render::Vertex::new(-0.5, -0.5, 0.0, 1., 0., 0.),
            render::Vertex::new(-0.5, -0.5, 1.0, 0., 0., 1.),

            render::Vertex::new(-0.5, -0.5, 1.0, 0., 0., 1.),
            render::Vertex::new( 0.5, -0.5, 1.0, 0., 1., 0.),
            render::Vertex::new( 0.5, -0.5, 0.0, 1., 0., 0.),
        ], vec![
            0u32, 1u32, 2u32,
            0u32, 4u32, 1u32,
            0u32, 7u32, 4u32,
            0u32, 2u32, 7u32,
            1u32, 2u32, 7u32,
            7u32, 4u32, 1u32,
        ],
    );

    vec![pyramid.clone(), pyramid.clone(), pyramid]
}

pub fn update() {
    //
}
