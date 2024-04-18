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

### Parameters
```bash
# cpu path tracer, spp=4096, traceDepth=20, openMP threads=256
build/cpu_path_tracer/cpu_path_tracer 4096 20 256

# cuda path tracer, spp=4096, traceDepth=20, cuda threads=256, Singlekernel Mode
build/gpu_path_tracer/gpu_path_tracer 4096 20 256 1
# cuda path tracer, spp=4096, traceDepth=20, cuda threads=256, StreamCompaction Mode
build/gpu_path_tracer/gpu_path_tracer 4096 20 256 2
```

## Features

TODO

## Roadmap

- [x] CPU Path Tracer
- [x] Anti-Aliasing by Jittered Sampling
- [ ] Construct GPU-friendly BVH
- [ ] Self-defined material description file

### GPU Parallelism

- Naive GPU Path Tracer
  - [x] Thread-per-pixel parallelism (# of threads = # of pixels)
- GPU Path Tracer
  - [x] Thread-per-path parallelism (# of threads = # of paths = # of pixels * SPP)
- Work-Efficient GPU Path Tracer
  - [x] Thread-per-ray parallelism (fixed # of threads)

### Rendering

- [x] Naive Diffuse Surfaces
- [x] Lambert Material
- [x] Metal Material
- [ ] Dielectric Material

### Acceleration Structures

- [x] Naive BVH
- [ ] SAH BVH

## Future Work

- [ ] Building BVH on GPU (LBVH <https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/>)
