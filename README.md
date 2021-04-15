# **Essentials:** High-Performance C++ GPU Graph Analytics
**Gunrock/Essentials** is a CUDA library for graph-processing designed specifically for the GPU. It uses a **high-level**, **bulk-synchronous**, **data-centric abstraction** focused on operations on vertex or edge frontiers. Gunrock achieves a balance between performance and expressiveness by coupling high-performance GPU computing primitives and optimization strategies, particularly in the area of fine-grained load balancing, with a high-level programming model that allows programmers to quickly develop new graph primitives that scale from one to many GPUs on a node with small code size and minimal GPU programming knowledge.

## Quick Start Guide

Before building Gunrock make sure you have **CUDA Toolkit 11 or higher** installed on your system. Other external dependencies such as `NVIDIA/thrust`, `NVIDIA/cub`, etc. are automatically fetched using `cmake`.

```shell
git clone https://github.com/gunrock/essentials.git
cd essentials
mkdir build && cd build
cmake .. 
make sssp # or for all applications, use: make -j$(nproc)
bin/sssp ../datasets/chesapeake.mtx
```

## Getting Started with Gunrock

- **Tutorial:** [Gunrock's programming model]()
- **Tutorial:** [Write a graph primitive within Gunrock]()
- **Tutorial:** [Building Gunrock]()
- [API reference documentation]()
- [Performance analysis]()
- [Publications](https://gunrock.github.io/docs/#/gunrock/publications_and_presentations) and [presentations](https://gunrock.github.io/docs/#/gunrock/publications_and_presentations?id=presentations)

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

Gunrock is copyright The Regents of the University of California, 2021. The library, examples, and all source code are released under [Apache 2.0](https://github.com/gunrock/essentials/blob/master/LICENSE).