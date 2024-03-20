# Towards Real-Time Path Tracing

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
  - [ ] Thread-per-pixel parallelism
- GPU Path Tracer
  - [ ] Thread-per-path parallelism
- Work-Efficient GPU Path Tracer
  - [ ] Thread-per-ray parallelism

### Rendering

- [x] Diffuse Surfaces

TODO
