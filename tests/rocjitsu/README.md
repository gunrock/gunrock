# GPU-less AMD tests

The rocjitsu job runs Gunrock HIP binaries on CPU-emulated AMD GPUs. Mirage
loads rocjitsu (ROCm Just-in-Time Suite), which intercepts KFD/DRM calls and
executes AMD GPU ISA on the CPU. No `/dev/kfd`, `/dev/dri`, physical GPU, or
privileged container is required.

## Test and resource strategy

The job uses one container and two sequential builds:

- `gfx950`: the enabled CTest suite, LLVM host coverage, and the 70% gate.
- `gfx942`: one architecture-compatibility CTest covering runtime discovery,
  allocation, a kernel launch, synchronization, and a device-to-host result.

The full build is reused for every `gfx950` test. The smaller `gfx942` build
compiles only `gunrock_architecture_smoke`. Running the full suite twice would
add compiler and emulator time without adding behavior coverage. The image is
digest-pinned; caching its multi-gigabyte layers in GitHub Actions would cost
more storage and transfer than rebuilding the small project-specific layer.

CTest discovers GoogleTest cases immediately before execution, inside the
emulator. This keeps build-only CUDA, ROCm, Windows, and PyGunrock jobs from
running tests while avoiding a hand-maintained test list.

## Coverage definition

Coverage is LLVM source line coverage from the Clang host compilation of
`gunrock_tests`. The gate is a hard 70% over compiler-emitted first-party host
mappings under `include/gunrock/`.

The scope excludes:

- tests, examples, benchmarks, generated files, and external dependencies;
- the bundled NIST Matrix Market helpers under `include/gunrock/io/detail/`;
- AMD device-only operators and primitives whose counters cannot be returned
  by rocjitsu (`framework/operators/`, except the host `for` wrapper, plus
  `util/math.hxx` and `util/load_store.hxx`).

rocjitsu executes the device code for correctness, but LLVM's host profile
runtime cannot collect counters from emulated AMDGPU code objects. The reported
percentage therefore does not claim device line coverage.

Triangle-counting tests remain registered with the `hardware-required` CTest
label. This rocjitsu release reports memory-wait hazards for their intersection
kernel and returns zero counts, so the job excludes exactly that label. The
tests are still compiled and remain available on physical GPUs.

## Run locally

Prerequisites are x86-64 Linux, Docker, roughly 20 GB of free disk, and enough
RAM for one compiler/emulator process.

```bash
docker build -t gunrock-rocjitsu docker/rocjitsu
mkdir -p results/rocjitsu
docker run --rm --init --shm-size 4g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/home -e RJ_THREADS=2 -e RJ_BUILD_JOBS=2 \
  -v "$PWD:/work" -w /work gunrock-rocjitsu \
  tests/rocjitsu/run-tests.sh results/rocjitsu
```

`RJ_THREADS` controls rocjitsu's CPU thread budget. `RJ_BUILD_JOBS` controls
CMake build parallelism.

## Updating the image

The Dockerfile uses a digest-pinned Triton AMD CI image containing TheRock
ROCm, Mirage, rocjitsu, and the `gfx942` and `gfx950` configurations. When
updating it, verify that the SDK path resolves, both emulator configurations
exist, the CTest suite passes, and the coverage report is produced.
