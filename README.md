# Towards Real-Time Path Tracing

## Environment

- cuda/11.7.0-7sdye3
- Nvidia V100 GPU

## Getting Started

In one of the above directories, run the following commands:

```bash
mkdir build
cd build
cmake ..
make # if Linux or Mac
cmake --build . # if Windows
# by default it will build both cpu_path_tracer and gpu_patch_tracer, you can specify the target you want

# [Attention] run the program from root, for example
./build/cpu_path_tracer/Debug/cpu_path_tracer # default SPP = 32, MaxDepth = 50
# or
./build/cpu_path_tracer/Debug/cpu_path_tracer <SPP>
# or
./build/cpu_path_tracer/Debug/cpu_path_tracer <SPP> <MaxDepth>
```

Now, view the output image in the `~/out` directory.

## Features

TODO

## Roadmap

- [x] CPU Path Tracer
- [x] Anti-Aliasing by Jittered Sampling
- [ ] Construct GPU-friendly BVH
- [ ] Self-defined material description file

### GPU Parallelism

- Naive GPU Path Tracer
  - [ ] Thread-per-pixel parallelism (# of threads = # of pixels)
- GPU Path Tracer
  - [ ] Thread-per-path parallelism (# of threads = # of paths = # of pixels * SPP)
- Work-Efficient GPU Path Tracer
  - [ ] Thread-per-ray parallelism (fixed # of threads)

### Rendering

- [x] Naive Diffuse Surfaces
- [ ] Lambert Material
- [ ] Metal Material
- [ ] Dielectric Material

### Acceleration Structures

- [x] Naive BVH
- [ ] SAH BVH

## Future Work

- [ ] Building BVH on GPU (LBVH <https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/>)
