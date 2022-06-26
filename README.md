# [Essentials: High-Performance C++ GPU Graph Analytics](https://github.com/gunrock/essentials/wiki)
[![Ubuntu](https://github.com/gunrock/essentials/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/gunrock/essentials/actions/workflows/ubuntu.yml) [![Windows](https://github.com/gunrock/essentials/actions/workflows/windows.yml/badge.svg)](https://github.com/gunrock/essentials/actions/workflows/windows.yml) [![Code Quality](https://github.com/gunrock/essentials/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/gunrock/essentials/actions/workflows/codeql-analysis.yml) [![Ubuntu: Testing](https://github.com/gunrock/essentials/actions/workflows/ubuntu-tests.yml/badge.svg)](https://github.com/gunrock/essentials/actions/workflows/ubuntu-tests.yml)

**Gunrock/Essentials** is a CUDA library for graph-processing designed specifically for the GPU. It uses a **high-level**, **bulk-synchronous**, **data-centric abstraction** focused on operations on vertex or edge frontiers. Gunrock achieves a balance between performance and expressiveness by coupling high-performance GPU computing primitives and optimization strategies, particularly in the area of fine-grained load balancing, with a high-level programming model that allows programmers to quickly develop new graph primitives that scale from one to many GPUs on a node with small code size and minimal GPU programming knowledge.

## Quick Start Guide

- [Gunrock's Documentation](https://github.com/gunrock/essentials/wiki)

Before building Gunrock make sure you have **CUDA Toolkit**[<sup>[1]</sup>](#footnotes) installed on your system. Other external dependencies such as `NVIDIA/thrust`, `NVIDIA/cub`, etc. are automatically fetched using `cmake`.

```shell
git clone https://github.com/gunrock/essentials.git
cd essentials
mkdir build && cd build
cmake .. 
make sssp # or for all algorithms, use: make -j$(nproc)
bin/sssp ../datasets/chesapeake/chesapeake.mtx
```

## How to Cite Gunrock & Essentials
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

```tex
@InProceedings{Osama:2022:EOP,
  author =	 {Muhammad Osama and Serban D. Porumbescu and John D. Owens},
  title =	 {Essentials of Parallel Graph Analytics},
  booktitle =	 {Proceedings of the Workshop on Graphs,
                  Architectures, Programming, and Learning},
  year =	 2022,
  series =	 {GrAPL 2022},
  month =	 may,
  pages =	 {314--317},
  doi =		 {10.1109/IPDPSW55747.2022.00061},
  url =          {https://escholarship.org/uc/item/2p19z28q},
}
```

## Copyright and License

Gunrock is copyright The Regents of the University of California. The library, examples, and all source code are released under [Apache 2.0](https://github.com/gunrock/essentials/blob/master/LICENSE).

<a class="anchor" id="1"></a>
## Footnotes
1. Preferred **CUDA v11.5.1 or higher** due to support for stream ordered memory allocators (e.g. `cudaFreeAsync()`).
2. Essentials is intended as a future release of Gunrock. You can read more about in our vision paper: [Essentials of Parallel Graph Analytics](https://escholarship.org/content/qt2p19z28q/qt2p19z28q_noSplash_38a658bccc817ba025517311a776840f.pdf).
