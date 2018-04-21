extern crate tobj;

use render;

use std::path::Path;

pub fn load_obj(filename: &str) -> Vec<render::ModelData> {

    let tobj_models = tobj::load_obj(&Path::new(filename));
    assert!(tobj_models.is_ok());
    let (models, materials) = tobj_models.unwrap();
    
    let mut return_models = Vec::new();

    for m in models {
        let mesh = &m.mesh;
        let positions = &mesh.positions;
        let mut normals = mesh.normals.clone();
        let indices = mesh.indices.clone();
        let mut vertices = Vec::new();
        let mut has_normals = true;
        
        
        if normals.len() == 0 {
            for _ in 0..positions.len() {
                normals.push(0.);
            }
            hasNormals = false;
        }

        for v in 0..positions.len() / 3 {
            let v3 = v * 3;

            vertices.push(
                render::Vertex::new_raw(
                    positions[v3], positions[v3+1], positions[v3+2],
                    normals[v3], normals[v3+1], normals[v3+2],
                    1., 1., 1.
                )
            );
        }

        return_models.push(
            if has_normals {
                render::ModelData::new(
                    vertices,
                    indices,
                )
            } else {
                render::ModelData::new_with_normals(
                    vertices,
                    indices,
                    render::NormalMode::Smooth,
                )
            }
        );
    }
    return_models
}
