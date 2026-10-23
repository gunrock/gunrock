# Tests

- `unit/` contains the C++/HIP GoogleTest suite registered with CTest.
- `python/` contains PyGunrock pytest tests.
- `data/` contains compact deterministic fixtures shared by tests.
- `rocjitsu/` contains the GPU-less AMD runner and coverage policy.

Build the C++ tests with `-DESSENTIALS_BUILD_TESTS=ON`, then use CTest rather
than invoking a test binary directly:

```bash
cmake -S . -B build -DESSENTIALS_BUILD_TESTS=ON
cmake --build build --target gunrock_tests
ctest --test-dir build --output-on-failure
```

CTest discovery is deferred until test time. CI jobs that only build test
targets do not require a GPU.

PyGunrock tests remain part of the Python package build but are collected from
the repository-level hierarchy:

```bash
cd python
pytest ../tests/python
```
