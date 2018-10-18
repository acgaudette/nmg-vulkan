extern crate tobj;

use std;
use render;

/// Load obj meshes from path to vector of `render::ModelData`
pub fn load_obj(filename: &str) -> Vec<render::ModelData> {
    let tobj_models = tobj::load_obj(&std::path::Path::new(filename));
    let (models, _) = tobj_models.unwrap_or_else(
        |err| panic!(
            "Could not load obj file: \"{}\"", err
        )
    );

    let mut result = Vec::new();

    for model in models {
        let positions = &model.mesh.positions;
        let count = positions.len();

        let (normals, has_normals) = if model.mesh.normals.is_empty() {
            let mut normals = Vec::with_capacity(count);

            for _ in 0..count {
                normals.push(0f32)
            }

            (normals, false)
        } else {
            (model.mesh.normals, true)
        };

        let mut vertices = Vec::with_capacity(count);
        for v in 0..count / 3 {
            let v3 = v * 3;

            vertices.push(
                render::Vertex::new_raw(
                    positions[v3], positions[v3 + 1], positions[v3 + 2],
                      normals[v3],   normals[v3 + 1],   normals[v3 + 2],
                    1.0, 1.0, 1.0, // White
                    0.0, 0.0,
                )
            );
        }

        result.push(
            if has_normals {
                render::ModelData::new(
                    vertices,
                    model.mesh.indices,
                )
            } else {
                render::ModelData::new_with_normals(
                    vertices,
                    model.mesh.indices,
                    render::NormalMode::Smooth,
                )
            }
        );
    }

    result
}
