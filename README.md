# The `nmg` Rust x Vulkan Engine

For the upcoming game, nmg ([original prototype here][3])

[![Build Status][1]][2]

`nmg-vulkan`
is a game engine written from scratch in Rust,
specialized for two things:
1. Retro PSX-inspired graphics
2. High-performance mech physics

![](media/pingpong.gif)
![](media/softy.gif)
![](media/pyramid.gif)
![](media/walking.gif)

## "Features"
- Modular ECS architecture
- Softbody physics engine
- Low-level Vulkan backend
- Bathtub brew math module
- Cross-platform library crate
- Mimimal dependency count
- Could be data oriented

## Notes
- Currently requires Rust nightly

## Acknowledgements
- [cogciprocate/voodoo][4]

Papers implemented
- [Meshless Deformations Based On Shape Matching (Muller et al. 2016)][5]
- [A Robust Method to Extract the Rotational Part of Deformations (Muller et al. '05)][6]
- [Inverse Kinematics with Quaternion Joint Limits (JBlow '02)][7]

[1]: https://travis-ci.org/acgaudette/nmg-vulkan.svg?branch=master
[2]: https://travis-ci.org/acgaudette/nmg-vulkan
[3]: https://youtu.be/dD4nkrqb9RY
[4]: https://github.com/cogciprocate/voodoo
[5]: https://www.cs.drexel.edu/~david/Classes/Papers/MeshlessDeformations_SIG05.pdf
[6]: https://animation.rwth-aachen.de/media/papers/2016-MIG-StableRotation.pdf
[7]: http://number-none.com/product/IK%20with%20Quaternion%20Joint%20Limits/index.html
