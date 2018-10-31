extern crate tobj;

use std;
use alg;
use graphics;
use render;

/// Load obj meshes from path to vector of `render::ModelData` \
/// If the file contains normal data, it will be used.
/// Otherwise, normals are computed using `render::NormalMode::Smooth`. \
/// All submeshes will have all vertex colors set to `color`.
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

    let mut result = Vec::new();

    for model in models {
        // Cache buffer lengths for printing
        let (positions_len, indices_len, uvs_string) = (
            model.mesh.positions.len() / 3,
            model.mesh.indices.len(),
            if model.mesh.texcoords.is_empty() { "no uvs" } else { "uvs" },
        );

        // Compute normals and add submesh
        if model.mesh.normals.is_empty() {
            let vertices = if model.mesh.texcoords.is_empty() {
                model.mesh.positions.chunks_exact(3).map(
                    |chunk| alg::Vec3::new(chunk[0], chunk[1], chunk[2])
                ).map(|position| render::Vertex {
                    position, color, .. Default::default()
                }).collect()
            } else {
                model.mesh.positions.chunks_exact(3)
                    .zip(model.mesh.texcoords.chunks_exact(2))
                    .map(|data| {
                        (
                            alg::Vec3::new(data.0[0], data.0[1], data.0[2]),
                            alg::Vec2::new(data.1[0], data.1[1]),
                        )
                    }).map(|(position, uv)| render::Vertex {
                        position, color, uv, .. Default::default()
                    }).collect()
            };

            result.push(
                render::ModelData::new_with_normals(
                    &model.name,
                    vertices,
                    model.mesh.indices,
                    render::NormalMode::Smooth,
                )
            );

            println!(
                "\tLoaded submesh \"{}\" with \
                {} verts, {} indices, {} (computed normals)",
                model.name, positions_len, indices_len, uvs_string,
            );
        }

        // Add submesh with existing normals
        else {
            let vertices = if model.mesh.texcoords.is_empty() {
                model.mesh.positions.chunks_exact(3)
                    .zip(model.mesh.normals.chunks_exact(3))
                    .map(|data| {
                        (
                            alg::Vec3::new(data.0[0], data.0[1], data.0[2]),
                            alg::Vec3::new(data.1[0], data.1[1], data.1[2]),
                        )
                    }).map(
                        |(position, normal)| render::Vertex {
                            position,
                            normal,
                            color,
                            .. Default::default()
                        }
                    ).collect()
            } else {
                model.mesh.positions.chunks_exact(3)
                    .zip(model.mesh.normals.chunks_exact(3))
                    .zip(model.mesh.texcoords.chunks_exact(2))
                    .map(|data| {
                        (
                            alg::Vec3::new(
                                (data.0).0[0],
                                (data.0).0[1],
                                (data.0).0[2],
                            ),
                            alg::Vec3::new(
                                (data.0).1[0],
                                (data.0).1[1],
                                (data.0).1[2],
                            ),
                            alg::Vec2::new(data.1[0], data.1[1]),
                        )
                    }).map(
                        |(position, normal, uv)| render::Vertex {
                            position,
                            normal,
                            uv,
                            color,
                            .. Default::default()
                        }
                    ).collect()
            };

            result.push(
                render::ModelData::new(
                    &model.name,
                    vertices,
                    model.mesh.indices,
                )
            );

            println!(
                "\tLoaded submesh \"{}\" with \
                {} verts, {} indices, {}, and normals",
                model.name, positions_len, indices_len, uvs_string,
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
