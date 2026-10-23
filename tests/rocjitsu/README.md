# GPU-less AMD tests

The rocjitsu job runs Gunrock HIP binaries on CPU-emulated AMD GPUs. Mirage
loads rocjitsu (ROCm Just-in-Time Suite), which intercepts KFD/DRM calls and
executes AMD GPU ISA on the CPU. No `/dev/kfd`, `/dev/dri`, physical GPU, or
privileged container is required.

## Test and resource strategy

The job uses one container and four sequential builds:

- `gfx950`: the enabled CTest suite, LLVM host coverage, and the 90% gate.
- `gfx1250` (CDNA5): the enabled suite plus the architecture test.
- `gfx942` (CDNA3): the focused architecture test.
- `gfx1201` (RDNA4): the focused architecture test.

The full build is reused for every test on each full-suite architecture. The
focused builds compile only `gunrock_architecture_smoke`, which covers runtime
discovery, allocation and copies, streams and events, atomics, kernel launch,
synchronization, and a boundary crossing two blocks. The image is
digest-pinned; caching its multi-gigabyte layers would cost more storage and
transfer than rebuilding the small project-specific layer.

AMD identifies `gfx1250` as CDNA5, not RDNA4. The pinned rocjitsu image models
it as an Instinct MI455X. RDNA4 is represented by `gfx1200` and `gfx1201`; this
job uses the image's Radeon AI PRO R9700 (`gfx1201`) configuration. See AMD's
[supported GPU table](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
and the [rocjitsu RDNA configuration change](https://github.com/ROCm/rocm-systems/pull/7360).

CTest discovers GoogleTest cases immediately before execution, inside the
emulator. This keeps build-only CUDA, ROCm, Windows, and PyGunrock jobs from
running tests while avoiding a hand-maintained test list.

## Coverage definition

Coverage is LLVM source line coverage from the Clang host compilation of
`gunrock_tests`. The gate is a hard 90% over compiler-emitted first-party host
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

Non-empty triangle-counting tests remain registered with the
`hardware-required` CTest label. This rocjitsu release reports memory-wait
hazards for their intersection kernel and returns zero counts, so the job
excludes exactly those two tests. The empty-graph contract is covered in the
enabled suite, while the hardware tests remain compiled and available on
physical GPUs.

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
ROCm, Mirage, and rocjitsu. When updating it, verify that the SDK path resolves;
the `gfx942`, `gfx950`, `gfx1201`, and `gfx1250` configurations exist; all
intended CTest depths pass; and the coverage report is produced.
