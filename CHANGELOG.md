# Changelog

## v2.2.0

### What's Changed

#### 🎉 Algorithms and Operators

* [AMD/NVIDIA] Implement merge-path load-balanced advance operator by [@neoblizz](https://github.com/neoblizz) in [#1084](https://github.com/gunrock/gunrock/pull/1084)
  - Add true CTA-level merge-path advance operator using two-level partitioning (tile-level + thread-level binary search)
  - Add runtime load balance selection via `--advance_load_balance` flag
  - Add unit test for merge_path advance operator
  - Support block_mapped, merge_path, and thread_mapped strategies

#### 🐛 Bug Fixes

* Fix CMake HIP flags to avoid gfx942 macro collision with rocPRIM in [#1084](https://github.com/gunrock/gunrock/pull/1084)
* Fix enactor frontier pointer handling (remove reinterpret_cast) in [#1084](https://github.com/gunrock/gunrock/pull/1084)

#### 🏡 API Changes/Improvements

* [AMD] Consolidate into a single framework with both CUDA & ROCm backends by [@neoblizz](https://github.com/neoblizz) in [#1081](https://github.com/gunrock/gunrock/pull/1081)
  - Introduced CMake options `ESSENTIALS_NVIDIA_BACKEND` and `ESSENTIALS_AMD_BACKEND` to select the backend
  - Backend-specific dependencies and compiler definitions are now set conditionally
  - HIP-specific flags and include paths added for AMD
  - Updated GitHub Actions workflows to test both NVIDIA and AMD backends
  - Ubuntu workflow now installs either CUDA or ROCm/HIP as needed
  - Windows workflow updated to use the latest runner and supports NVIDIA backend

#### 📚 Documentation

* Docs: Deprecate develop and dev by [@neoblizz](https://github.com/neoblizz) in [#1086](https://github.com/gunrock/gunrock/pull/1086)
* Update documentation for HIP/AMD usage in [#1081](https://github.com/gunrock/gunrock/pull/1081)

#### 🗑️ Removals

* Remove DAWN algorithm (not sufficiently different from classic SSSP/BFS) in [#1084](https://github.com/gunrock/gunrock/pull/1084)

**Full Changelog**: https://github.com/gunrock/gunrock/compare/v2.1.0...v2.2.0

---

## v2.1.0

### What's Changed

#### 🎉 Algorithms and Operators

*   Support for CUDA 12.5.0 and SM 90. by [@neoblizz](https://github.com/neoblizz) in [#1072](https://github.com/gunrock/gunrock/pull/1072)
*   Add performance analysis by [@annielytical](https://github.com/annielytical) in [#920](https://github.com/gunrock/gunrock/pull/920)
*   Performance Analysis Updates by [@annielytical](https://github.com/annielytical) in [#923](https://github.com/gunrock/gunrock/pull/923)
*   fix bc bug (introduced with perf analysis) by [@annielytical](https://github.com/annielytical) in [#1035](https://github.com/gunrock/gunrock/pull/1035)
*   main/develop merge by [@neoblizz](https://github.com/neoblizz) in [#1044](https://github.com/gunrock/gunrock/pull/1044)
*   Fixing custom frontier-based merge-path approach. Not fully functiona… by [@neoblizz](https://github.com/neoblizz) in [#1045](https://github.com/gunrock/gunrock/pull/1045)

#### 🐛 Bug Fixes

*   MST Bugfix by [@neoblizz](https://github.com/neoblizz) in [#1075](https://github.com/gunrock/gunrock/pull/1075)

#### 🏡 API Changes/Improvements

*   Adds support for performance metrics collection. by [@neoblizz](https://github.com/neoblizz) in [#921](https://github.com/gunrock/gunrock/pull/921)
*   Improvement to existing build system. by [@neoblizz](https://github.com/neoblizz) in [#1032](https://github.com/gunrock/gunrock/pull/1032)
*   Support for `CMAKE_CUDA_ARCHITECTURE`. by [@neoblizz](https://github.com/neoblizz) in [#1042](https://github.com/gunrock/gunrock/pull/1042)
*   Variable was being ignored. by [@neoblizz](https://github.com/neoblizz) in [#1043](https://github.com/gunrock/gunrock/pull/1043)
*   Update Graph Builder API by [@annielytical](https://github.com/annielytical) in [#1046](https://github.com/gunrock/gunrock/pull/1046)

### New Contributors

*   [@marjerie](https://github.com/marjerie) made their first contribution in [#1049](https://github.com/gunrock/gunrock/pull/1049)
*   [@lxrzlyr](https://github.com/lxrzlyr) made their first contribution in [#1051](https://github.com/gunrock/gunrock/pull/1051)

**Full Changelog**: [2.0.0...v2.1.0](https://github.com/gunrock/gunrock/compare/2.0.0...v2.1.0)
