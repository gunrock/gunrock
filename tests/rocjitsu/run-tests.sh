#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result_dir=${1:-"$repo_root/results/rocjitsu"}
build_root=${RJ_BUILD_ROOT:-/tmp/gunrock-rocjitsu-build}
build_jobs=${RJ_BUILD_JOBS:-2}

mkdir -p "$result_dir" "${HOME:-/tmp}/.cache"
result_dir=$(cd "$result_dir" && pwd)

rocm-sdk init >/dev/null 2>&1 || true
rocm_root=$(rocm-sdk path --root)
export ROCM_PATH="$rocm_root"
export HIP_PATH="$rocm_root"
export CMAKE_PREFIX_PATH="$rocm_root${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CC="$rocm_root/lib/llvm/bin/clang"
export CXX="$rocm_root/lib/llvm/bin/clang++"

configure() {
  local build_dir=$1
  local arch=$2
  local build_type=$3
  local coverage=$4

  cmake -S "$repo_root" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DCMAKE_HIP_ARCHITECTURES="$arch" \
    -DESSENTIALS_AMD_BACKEND=ON \
    -DESSENTIALS_NVIDIA_BACKEND=OFF \
    -DESSENTIALS_BUILD_EXAMPLES=OFF \
    -DESSENTIALS_BUILD_TESTS=ON \
    -DESSENTIALS_BUILD_BENCHMARKS=OFF \
    -DESSENTIALS_FETCH_ROCM_LIBS=OFF \
    -DESSENTIALS_ENABLE_COVERAGE="$coverage"
}

full_arch=gfx950
full_build="$build_root-$full_arch"
full_results="$result_dir/$full_arch"
profile_dir="$full_results/profiles"
mkdir -p "$full_results"
rm -rf "$profile_dir"
mkdir -p "$profile_dir"

start=$SECONDS
configure "$full_build" "$full_arch" Debug ON
cmake --build "$full_build" --target gunrock_tests \
  --parallel "$build_jobs"
full_build_seconds=$((SECONDS - start))

start=$SECONDS
timeout --signal=TERM 15m \
  rjsim "$full_arch" -- \
  env LLVM_PROFILE_FILE="$profile_dir/gunrock-%p.profraw" \
  ctest --test-dir "$full_build" -N \
    --label-regex '^rocjitsu-full$' \
  2>&1 | tee "$full_results/discovery.log"

timeout --signal=TERM 15m \
  rjsim "$full_arch" -- \
  env LLVM_PROFILE_FILE="$profile_dir/gunrock-%p.profraw" \
  ctest --test-dir "$full_build" \
    --output-on-failure \
    --no-tests=error \
    --parallel 1 \
    --label-regex '^rocjitsu-full$' \
    --output-junit "$full_results/ctest.xml" \
  2>&1 | tee "$full_results/ctest.log"
full_test_seconds=$((SECONDS - start))

shopt -s nullglob
profiles=("$profile_dir"/*.profraw)
shopt -u nullglob
if (( ${#profiles[@]} == 0 )); then
  echo "rocjitsu coverage: no LLVM profiles were produced" >&2
  exit 1
fi

profdata="$full_results/coverage.profdata"
"$rocm_root/lib/llvm/bin/llvm-profdata" merge --sparse \
  "${profiles[@]}" -o "$profdata"
python3 "$repo_root/tests/rocjitsu/check-coverage.py" \
  --llvm-cov "$rocm_root/lib/llvm/bin/llvm-cov" \
  --binary "$full_build/bin/gunrock_tests" \
  --profile "$profdata" \
  --source-root "$repo_root" \
  --minimum 90 \
  --output "$full_results/coverage.json" \
  2>&1 | tee "$full_results/coverage.log"

cdna5_arch=gfx1250
cdna5_build="$build_root-$cdna5_arch"
cdna5_results="$result_dir/$cdna5_arch"
mkdir -p "$cdna5_results"

start=$SECONDS
configure "$cdna5_build" "$cdna5_arch" Release OFF
cmake --build "$cdna5_build" \
  --target gunrock_tests gunrock_architecture_smoke \
  --parallel "$build_jobs"
cdna5_build_seconds=$((SECONDS - start))

start=$SECONDS
timeout --signal=TERM 15m \
  rjsim "$cdna5_arch" -- \
  ctest --test-dir "$cdna5_build" -N \
    --label-regex '^(rocjitsu-full|rocjitsu-architecture)$' \
  2>&1 | tee "$cdna5_results/discovery.log"

timeout --signal=TERM 15m \
  rjsim "$cdna5_arch" -- \
  ctest --test-dir "$cdna5_build" \
    --output-on-failure \
    --no-tests=error \
    --parallel 1 \
    --label-regex '^(rocjitsu-full|rocjitsu-architecture)$' \
    --output-junit "$cdna5_results/ctest.xml" \
  2>&1 | tee "$cdna5_results/ctest.log"
cdna5_test_seconds=$((SECONDS - start))

printf '%s\n' \
  "gfx950_build_seconds=$full_build_seconds" \
  "gfx950_test_seconds=$full_test_seconds" \
  "gfx1250_build_seconds=$cdna5_build_seconds" \
  "gfx1250_test_seconds=$cdna5_test_seconds" \
  > "$result_dir/runtime.txt"

run_architecture_smoke() {
  local arch=$1
  local build="$build_root-$arch"
  local results="$result_dir/$arch"
  local start
  local build_seconds
  local test_seconds
  mkdir -p "$results"

  start=$SECONDS
  configure "$build" "$arch" Release OFF
  cmake --build "$build" --target gunrock_architecture_smoke \
    --parallel "$build_jobs"
  build_seconds=$((SECONDS - start))

  start=$SECONDS
  timeout --signal=TERM 5m \
    rjsim "$arch" -- \
    ctest --test-dir "$build" \
      --output-on-failure \
      --no-tests=error \
      --parallel 1 \
      --label-regex '^rocjitsu-architecture$' \
      --output-junit "$results/ctest.xml" \
    2>&1 | tee "$results/ctest.log"
  test_seconds=$((SECONDS - start))

  printf '%s\n' \
    "${arch}_build_seconds=$build_seconds" \
    "${arch}_test_seconds=$test_seconds" \
    >> "$result_dir/runtime.txt"
}

run_architecture_smoke gfx942
run_architecture_smoke gfx1201

echo "rocjitsu suite passed"
