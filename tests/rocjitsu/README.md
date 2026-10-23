# GPU-less AMD smoke test

This test runs Gunrock HIP binaries on a CPU-emulated AMD GPU. Mirage loads
rocjitsu (ROCm Just-in-Time Suite), which intercepts KFD/DRM calls and
executes AMD GPU ISA on the CPU. rocjitsu is not HIPRTC or another HIP
runtime-compilation API.

No `/dev/kfd`, `/dev/dri`, physical GPU, or privileged container is required.

## Coverage

The smoke test builds only the `bfs` target for `gfx942` and `gfx950`, then
validates a five-vertex graph against Gunrock's CPU reference. This covers
CMake's AMD backend, HIP compilation and linking, device discovery, streams
and events, device memory, Thrust/rocThrust, atomics, advance/filter kernels,
and host-device copies.

rocjitsu has no GPU timing model, so this test does not run benchmarks or
check timing. It also does not cover occupancy, cooperative launch, virtual
memory, sparse libraries, asynchronous execution, or multi-GPU behavior.
Those paths still require testing on AMD hardware.

## Run locally

Prerequisites are x86-64 Linux, Docker, roughly 20 GB of free disk, and enough
RAM for one compiler/emulator process. The base image is about 5.4 GiB
compressed and roughly 15 GB unpacked.

```bash
docker build -t gunrock-rocjitsu docker/rocjitsu
mkdir -p results/rocjitsu
docker run --rm --init --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/home -e RJ_THREADS=2 -e RJ_BUILD_JOBS=2 \
  -v "$PWD:/work" -w /work gunrock-rocjitsu \
  rjsim gfx950 -- tests/rocjitsu/run-smoke.sh results/rocjitsu
```

`RJ_THREADS` controls rocjitsu's CPU thread budget. `RJ_BUILD_JOBS` controls
the CMake build parallelism. Repeat with `rjsim gfx942 -- ...` to test both
CI targets.

## Updating the image

The workflow uses a digest-pinned Triton AMD CI image containing TheRock ROCm,
Mirage, rocjitsu, and the `gfx942` and `gfx950` configurations.

When updating the base image, verify:

- `rocm-sdk path --root` resolves a complete development SDK.
- `bin/mirage`, `lib/librocjitsu.so`, and both JSON configs exist.
- Gunrock configures and builds with the image's HIP compiler and ROCm CMake
  packages.
- The smoke test reports `Number of errors : 0`.

## Security

The workflow has read-only GitHub permissions, does not persist checkout
credentials, and runs on an ephemeral GitHub-hosted runner. The base image is
pinned by digest. Ubuntu packages and Gunrock's existing CMake dependencies
are still downloaded during the build.
