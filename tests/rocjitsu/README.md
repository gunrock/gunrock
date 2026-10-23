# GPU-less AMD tests

The rocjitsu job runs Gunrock HIP binaries on CPU-emulated AMD GPUs. Mirage
loads rocjitsu (ROCm Just-in-Time Suite), which intercepts KFD/DRM calls and
executes AMD GPU ISA on the CPU. No `/dev/kfd`, `/dev/dri`, physical GPU, or
privileged container is required.

## SDK installation

CI starts from a digest-pinned Ubuntu 24.04 image and installs
`rocm[libraries,devel]==10.3.0a20261023` from AMD's
[`whl-next` index](https://nightly.repo.amd.com/rocm/whl-next/). This is an
exactly pinned nightly alpha, not a stable ROCm release. `devel` alone is not
sufficient: Gunrock also links libraries including hipRAND and hipSPARSE.

The requirements files under `.github/requirements/` list every selected SDK
package and its artifact SHA-256. Installation uses `--require-hashes`,
preinstalls the pinned build backend, disables build isolation and dependency
resolution, and does not cache downloads. A missing, expired, or changed
artifact therefore fails the job instead of silently selecting another build.
The initialized SDK is about 6.6 GB, so CI does not cache its environment.

ROCm components retain their respective upstream licenses. The rocjitsu
component is MIT-licensed, but that does not make every bundled SDK component
MIT-licensed. Do not redistribute a wheel cache or initialized SDK without
reviewing and preserving the applicable notices.

## Test and resource strategy

The job uses one SDK environment and four sequential builds:

- `gfx950`: the enabled CTest suite, LLVM host coverage, and the 90% gate.
- `gfx1250` (CDNA5): the enabled suite plus the architecture test.
- `gfx942` (CDNA3): the focused architecture test.
- `gfx1201` (RDNA4): the focused architecture test.

The full build is reused for every test on each full-suite architecture. The
focused builds compile only `gunrock_architecture_smoke`, which covers runtime
discovery, allocation and copies, streams and events, atomics, kernel launch,
synchronization, and a boundary crossing two blocks. The smoke test checks the
HIP-reported `gcnArchName` and wave size against the requested target before it
launches a kernel. The launcher also uses a private runtime directory, and the
job rejects KFD, DXG, and DRM render devices so a hardware fallback cannot
false-pass.

AMD identifies `gfx1250` as CDNA5, not RDNA4. The pinned SDK models it as an
Instinct MI455X. RDNA4 is represented by `gfx1200` and `gfx1201`; this job uses
the SDK's Radeon AI PRO R9700 (`gfx1201`) configuration. See AMD's
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

The tested host is x86-64 Ubuntu 24.04. Allow roughly 10 GB of free disk and
enough RAM for one compiler/emulator process. Install `build-essential`,
`cmake`, `git`, `ninja-build`, `python3`, and `python3-venv`, then run:

```bash
python3 -m venv /tmp/gunrock-rocm-sdk
/tmp/gunrock-rocm-sdk/bin/pip install --no-cache-dir --no-deps \
  --only-binary=:all: --require-hashes \
  -r .github/requirements/rocjitsu-bootstrap.txt
/tmp/gunrock-rocm-sdk/bin/pip install --no-cache-dir --no-deps \
  --no-build-isolation --require-hashes \
  --index-url https://nightly.repo.amd.com/rocm/whl-next/ \
  -r .github/requirements/rocjitsu-sdk.txt
export PATH="/tmp/gunrock-rocm-sdk/bin:$PATH"
rocm-sdk init
mkdir -p results/rocjitsu
RJ_THREADS=2 RJ_BUILD_JOBS=2 \
  tests/rocjitsu/run-tests.sh results/rocjitsu
```

`RJ_THREADS` controls rocjitsu's CPU thread budget. `RJ_BUILD_JOBS` controls
CMake build parallelism. The test runner intentionally refuses to run when it
can see `/dev/kfd`, `/dev/dxg`, or a DRM render node.

## Quarterly update and rollback

AMD's nightly artifact retention is approximately 120 days, so refresh this
pin at least quarterly:

1. Select an exact nightly alpha that the rocjitsu maintainers support.
2. Download the source distribution, build backend, and all selected component
   wheels into a clean temporary directory with dependency resolution disabled.
3. Verify each downloaded artifact and replace the exact versions and SHA-256
   values in both requirements files. Update `ROCM_SDK_VERSION` in the workflow
   in the same commit.
4. Perform a cold, no-cache install and `rocm-sdk init`. Confirm Mirage and all
   four configuration files resolve below the derived SDK root.
5. Run the complete `gfx950` and `gfx1250` suites, the `gfx942` and `gfx1201`
   checks, the architecture mismatch negative test, and the 90% coverage gate.
6. Record setup time, test time, initialized size, and package availability in
   the pull request. Never describe or select a nightly as the latest stable
   ROCm release.

The retention policy is described in TheRock's
[repository RFC](https://github.com/ROCm/TheRock/blob/main/docs/rfcs/RFC0012-Repo-Structure.md).
There is no automatic fallback because it could hide a missing or mutated
artifact. To roll back, revert the SDK migration commit; Git restores the prior
digest-pinned Triton image path. Keep that rollback as a separate change from a
subsequent SDK pin update.
