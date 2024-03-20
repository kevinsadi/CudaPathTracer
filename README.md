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
./RayTracing
```

Now, view the output image in the `out` directory.

## Features

TODO

## Roadmap

### Parallelism

- [x] CPU Path Tracer
- Naive GPU Path Tracer
  - [ ] Building BVH on GPU
  - [ ] Thread-per-pixel parallelism (# of threads = # of pixels)
- GPU Path Tracer
  - [ ] Thread-per-path parallelism (# of threads = # of paths = # of pixels * SPP)
- Work-Efficient GPU Path Tracer
  - [ ] Thread-per-ray parallelism (fixed # of threads)

### Rendering

- [x] Diffuse Surfaces

TODO
