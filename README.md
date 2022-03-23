# **Essentials:** High-Performance C++ GPU Graph Analytics 
[![Ubuntu](https://github.com/gunrock/essentials/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/gunrock/essentials/actions/workflows/ubuntu.yml) [![Windows](https://github.com/gunrock/essentials/actions/workflows/windows.yml/badge.svg)](https://github.com/gunrock/essentials/actions/workflows/windows.yml) [![Code Quality](https://github.com/gunrock/essentials/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/gunrock/essentials/actions/workflows/codeql-analysis.yml) [![Ubuntu: Testing](https://github.com/gunrock/essentials/actions/workflows/ubuntu-tests.yml/badge.svg)](https://github.com/gunrock/essentials/actions/workflows/ubuntu-tests.yml)

**Gunrock/Essentials** is a CUDA library for graph-processing designed specifically for the GPU. It uses a **high-level**, **bulk-synchronous**, **data-centric abstraction** focused on operations on vertex or edge frontiers. Gunrock achieves a balance between performance and expressiveness by coupling high-performance GPU computing primitives and optimization strategies, particularly in the area of fine-grained load balancing, with a high-level programming model that allows programmers to quickly develop new graph primitives that scale from one to many GPUs on a node with small code size and minimal GPU programming knowledge.

## Quick Start Guide

Before building Gunrock make sure you have **CUDA Toolkit**[^1] installed on your system. Other external dependencies such as `NVIDIA/thrust`, `NVIDIA/cub`, etc. are automatically fetched using `cmake`.

```shell
git clone https://github.com/gunrock/essentials.git
cd essentials
mkdir build && cd build
cmake .. 
make sssp # or for all algorithms, use: make -j$(nproc)
bin/sssp ../datasets/chesapeake/chesapeake.mtx
```
[^1]: Preferred **CUDA v11.5.1 or higher** due to support for stream ordered memory allocators (e.g. `cudaFreeAsync()`).

## Getting Started with Gunrock

- [👻 (GitHub Template) `essentials` project example](https://github.com/gunrock/applications)
- [Gunrock's documentation](https://github.com/gunrock/essentials/wiki)
- [Gunrock's overview](https://github.com/gunrock/essentials/wiki/Overview)
- [Gunrock's programming model](https://github.com/gunrock/essentials/wiki/Programming-Model)
- [Publications](https://github.com/gunrock/essentials/wiki/Publications) and [presentations](https://github.com/gunrock/essentials/wiki/Presentations)
- [Essentials](https://github.com/gunrock/essentials) versus [Gunrock](https://github.com/gunrock/gunrock)[^2]

[^2]: Essentials is the future of Gunrock. The idea is to take the lessons learned from Gunrock to a new design, which simplifies the effort it takes to **(1)** implement graph algorithms, **(2)** add internal optimizations, **(3)** conduct future research. One example is Gunrock's SSSP, implemented in 4-5 files with 1000s of lines of code versus in essentials, it is a single file with less than 200 lines of code. Our end goal with essentials is possibly releasing it as a `v2.0.0` for Gunrock.

## How to Cite Gunrock
Thank you for citing our work.

```tex
@article{Wang:2017:GGG,
  author =	 {Yangzihao Wang and Yuechao Pan and Andrew Davidson
                  and Yuduo Wu and Carl Yang and Leyuan Wang and
                  Muhammad Osama and Chenshan Yuan and Weitang Liu and
                  Andy T. Riffel and John D. Owens},
  title =	 {{G}unrock: {GPU} Graph Analytics},
  journal =	 {ACM Transactions on Parallel Computing},
  year =	 2017,
  volume =	 4,
  number =	 1,
  month =	 aug,
  pages =	 {3:1--3:49},
  doi =		 {10.1145/3108140},
  ee =		 {http://arxiv.org/abs/1701.01170},
  acmauthorize = {http://dl.acm.org/authorize?N45082},
  url =		 {http://escholarship.org/uc/item/9gj6r1dj},
  code =	 {https://github.com/gunrock/gunrock},
  ucdcite =	 {a115},
}
```

## Copyright and License

Gunrock is copyright The Regents of the University of California. The library, examples, and all source code are released under [Apache 2.0](https://github.com/gunrock/essentials/blob/master/LICENSE).
