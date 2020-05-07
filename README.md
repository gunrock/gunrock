# gunrock/essentials
Goal of this project is to condense Gunrock into essential pieces, and opt for a much more modular design promoting code reuse and changes when needed. 

An example of such module is the extensive support provided using `gunrock::array` abstraction (`Array1D` in Gunrock version 1.x.x), internally we can choose to use any number of ways to allocate, free, set array values, but externally the interface should be clear and concise. And be able to simply `#include <gunrock/datastructs/array.cuh>` is very desirable without the need for being tied to other elements within the Gunrock library.

As CUDA evolves, we should simply be able to change the internal calls of these modules and stay up-to-date with the most performant method of doing something across the different CUDA architectures.

- Promote namespaces
- Modular design
- Documentation
- Manual error handling is tedious 
- Avoid user-defined-loops
- Reduce code-size
- Use externals when available
- Make core code readable/modifiable
- **Merge to gunrock/gunrock (major release)**

> Follow along at [gunrock/essentials](https://github.com/gunrock/gunrock/projects/4).

# conflicts
One annoyance with the whole library is the need C-API for Python wrappers. This goes against the whole idea of neatly templated C++ code, and I *do not like it at all*. C interface doesn't support extended `__device__ __host__` lambdas either, and the code is just very tedious to write and maintain at the C-API level. We need to find a clean way to support a Python API that doesn't need a C-interface and hooks directly into C++.