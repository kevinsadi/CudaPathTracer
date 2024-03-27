# Towards Real-Time Path Tracing

## Environment

- cuda/11.7.0-7sdye3
- Nvidia V100 GPU

## Getting Started

- CPU Tracer

```bash
cd cpu_tracer
```

- GPU Tracer

```bash
cd gpu_tracer
```

In one of the above directories, run the following commands:

```bash
mkdir build
cd build
cmake ..
make
./RayTracing # default SPP = 32, MaxDepth = 50
# or
./RayTracing <SPP>
# or
./RayTracing <SPP> <MaxDepth>
```

Now, view the output image in the `out` directory.

## Features

TODO

## Roadmap

- [x] CPU Path Tracer
- [x] Anti-Aliasing by Jittered Sampling
- [ ] Construct GPU-friendly BVH

### GPU Parallelism

- Naive GPU Path Tracer
  - [ ] Thread-per-pixel parallelism (# of threads = # of pixels)
- GPU Path Tracer
  - [ ] Thread-per-path parallelism (# of threads = # of paths = # of pixels * SPP)
- Work-Efficient GPU Path Tracer
  - [ ] Thread-per-ray parallelism (fixed # of threads)

### Rendering

- [x] Diffuse Surfaces

### Acceleration Structures

- [x] Naive BVH
- [ ] SAH BVH

## Future Work

- [ ] Building BVH on GPU (LBVH <https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/>)
