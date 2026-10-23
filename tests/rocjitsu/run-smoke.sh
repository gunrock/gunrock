#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
arch=${RJ_ARCH:-gfx950}
build_dir=${RJ_BUILD_DIR:-/tmp/gunrock-rocjitsu-build-"$arch"}
result_dir=${1:-"$repo_root/results/rocjitsu"}

mkdir -p "$result_dir" "${HOME:-/tmp}/.cache"
result_dir=$(cd "$result_dir" && pwd)

rocm-sdk init >/dev/null 2>&1 || true
rocm_root=$(rocm-sdk path --root)
export ROCM_PATH="$rocm_root"
export HIP_PATH="$rocm_root"
export CMAKE_PREFIX_PATH="$rocm_root${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CC="$rocm_root/lib/llvm/bin/clang"
export CXX="$rocm_root/lib/llvm/bin/clang++"

cmake -S "$repo_root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES="$arch" \
  -DESSENTIALS_AMD_BACKEND=ON \
  -DESSENTIALS_NVIDIA_BACKEND=OFF \
  -DESSENTIALS_BUILD_EXAMPLES=ON \
  -DESSENTIALS_BUILD_TESTS=OFF \
  -DESSENTIALS_BUILD_BENCHMARKS=OFF \
  -DESSENTIALS_FETCH_ROCM_LIBS=OFF

cmake --build "$build_dir" --target bfs --parallel "${RJ_BUILD_JOBS:-2}"

set +e
"$build_dir/bin/bfs" \
  --market "$repo_root/tests/rocjitsu/tiny.mtx" \
  --src 0 \
  --validate \
  --advance_load_balance block_mapped \
  --enable_filter \
  --filter_algorithm predicated \
  2>&1 | tee "$result_dir/bfs.log"
status=${PIPESTATUS[0]}
set -e

if [[ $status -ne 0 ]]; then
  exit "$status"
fi

if ! grep -Fq "Number of errors : 0" "$result_dir/bfs.log"; then
  echo "RocJITsu smoke test: BFS did not match the CPU reference" >&2
  exit 1
fi

echo "RocJITsu smoke test passed for $arch"
