# The `nmg` Rust x Vulkan Engine

For the upcoming game, `nmg` ([watch original prototype here!][3])

[![Build Status][1]][2]

`nmg-vulkan`
is a data-oriented game engine written from scratch in Rust,
specialized for two things:
1. Retro PSX-inspired graphics
2. High-performance mech physics

![](media/pingpong.gif)
![](media/softy.gif)
![](media/pyramid.gif)
![](media/walking.gif)

## Features
- Low-level Vulkan backend
- Softbody physics engine
- Optimized ECS architecture
- Homebrew math module
- Library crate

Currently requires Rust nightly.

## Acknowledgements
- [cogciprocate/voodoo][4]
- [tomaka/winit][5]

[1]: https://travis-ci.org/acgaudette/nmg-vulkan.svg?branch=master
[2]: https://travis-ci.org/acgaudette/nmg-vulkan
[3]: https://youtu.be/dD4nkrqb9RY
[4]: https://github.com/cogciprocate/voodoo
[5]: https://github.com/tomaka/winit
