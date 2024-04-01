# Towards Real-Time Path Tracing

## Environment

- cuda/11.7.0-7sdye3
- Nvidia V100 GPU

## Getting Started

### Linux or Mac

```bash
./scripts/setup.sh

./scripts/run.sh cpu_path_tracer 
# or
./scripts/run.sh cpu_path_tracer <SPP>
# or
./scripts/run.sh cpu_path_tracer <SPP> <MaxDepth>
```

### Windows

```bash
./scripts/setup.bat

./scripts/run.bat cpu_path_tracer 
# or
./scripts/run.bat cpu_path_tracer <SPP>
# or
./scripts/run.bat cpu_path_tracer <SPP> <MaxDepth>
```

Now, view the output image in the `out` directory at root.

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
