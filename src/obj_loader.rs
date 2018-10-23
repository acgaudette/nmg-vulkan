extern crate tobj;

use std;
use alg;
use graphics;
use render;

/// Load obj meshes from path to vector of `render::ModelData` \
/// If the file contains normal data, it will be used.
/// Otherwise, normals are computed using `render::NormalMode::Smooth`. \
pub fn load_obj(
    filename: &str,
    color: graphics::Color,
) -> Vec<render::ModelData> {
    let tobj_models = tobj::load_obj(&std::path::Path::new(filename));
    let (models, _) = tobj_models.unwrap_or_else(
        |err| panic!(
            "Could not load obj file: \"{}\"", err
        )
    );

    // TODO: UVs
    let mut result = Vec::new();

    for model in models {
        if model.mesh.normals.is_empty() {
            let vertices: Vec<render::Vertex> = model.mesh.positions
                .chunks_exact(3)
                .map(|chunk| alg::Vec3::new(chunk[0], chunk[1], chunk[2]))
                .map(|position| render::Vertex {
                    position, color, .. Default::default()
                }).collect();

            // Cache buffer lengths for printing
            let (vertices_len, indices_len) = (
                vertices.len(),
                model.mesh.indices.len(),
            );

            // Compute normals and add submesh
            result.push(
                render::ModelData::new_with_normals(
                    vertices,
                    model.mesh.indices,
                    render::NormalMode::Smooth,
                )
            );

            println!(
                "\tLoaded submesh with {} verts, {} indices (computed normals)",
                vertices_len,
                indices_len,
            );
        } else {
            let vertices: Vec<render::Vertex> = model.mesh.positions
                .chunks_exact(3)
                .zip(model.mesh.normals.chunks_exact(3))
                .map(|data| {
                    (
                        alg::Vec3::new(data.0[0], data.0[1], data.0[2]),
                        alg::Vec3::new(data.1[0], data.1[1], data.1[2]),
                    )
                }).map(|(position, normal)| render::Vertex {
                    position, normal, color, .. Default::default()
                }).collect();

            // Cache buffer lengths for printing
            let (vertices_len, indices_len) = (
                vertices.len(),
                model.mesh.indices.len(),
            );

            // Add submesh
            result.push(render::ModelData::new(vertices, model.mesh.indices));

            println!(
                "\tLoaded submesh with {} verts, {} indices, and normals",
                vertices_len,
                indices_len,
            );
        }
    }

    println!(
        "Loaded model from \"{}\" with {} submesh(es)",
        filename,
        result.len(),
    );

    result
}
