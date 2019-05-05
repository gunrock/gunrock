<p align="center">
  <a href="https://github.com/gunrock/gunrock/"><img src="https://github.com/gunrock/docs/raw/master/source/images/GunrockLogo150px.png"></a>
  <br>
  <a href="https://github.com/gunrock/gunrock/releases/tag/v1.0"><img src="https://img.shields.io/badge/gunrock-v1.0-blue.svg"></a>
  <a href="http://daisy.ece.ucdavis.edu:8080/job/gunrock/job/master/"><img src="http://daisy.ece.ucdavis.edu:8080/buildStatus/icon?job=gunrock/master" alt="Build Status"></a>
  <a href="https://developer.nvidia.com/gpu-accelerated-libraries"><img src="https://img.shields.io/badge/nvidia-accelerated%20libraries-green.svg?logo=nvidia" alt="NVIDIA Accelerated Libraries"></a>
  <br>
  <a href="https://github.com/gunrock/gunrock/blob/master/LICENSE.TXT"><img src="https://img.shields.io/github/license/gunrock/gunrock.svg" alt="Apache 2"></a>
  <a href="https://github.com/gunrock/gunrock/issues"><img src="https://img.shields.io/github/issues/gunrock/gunrock.svg" alt="Issues Open"></a>
  <a href="https://codecov.io/gh/gunrock/gunrock"><img src="https://codecov.io/gh/gunrock/gunrock/branch/master/graph/badge.svg" /></a>
</p>
<h1 id="gunrock-gpu-graph-analytics" align="center">Gunrock: GPU Graph Analytics</h1>

**Gunrock** is a CUDA library for graph-processing designed specifically for the GPU. It uses a **high-level**, **bulk-synchronous**, **data-centric abstraction** focused on operations on a vertex or edge frontier. Gunrock achieves a balance between performance and expressiveness by coupling high performance GPU computing primitives and optimization strategies with a high-level programming model that allows programmers to quickly develop new graph primitives with small code size and minimal GPU programming knowledge. For more details, see [Gunrock's Overview](http://gunrock.github.io/docs/#overview).


<table style="font-size: 12px;display: inline-table;"><thead>
<tr>
  <th><strong>Service</strong></th>
  <th><strong>System</strong></th>
  <th><strong>Environment</strong></th>
  <th><strong>Status</strong></th>
</tr>
</thead><tbody>
<tr>
  <td><a href="https://jenkins.io/">Jenkins</a></td>
  <td>Ubuntu 18.04.2 LTS</td>
  <td>CUDA 10.1, NVIDIA Driver 418.39, GCC/G++ 7.3</td>
  <td><a href="http://daisy.ece.ucdavis.edu:8080/blue/organizations/jenkins/gunrock/activity"><img src="http://daisy.ece.ucdavis.edu:8080/buildStatus/icon?job=gunrock/master" alt="Build Status"></a></td>
</tr>
</tbody></table>

## Quick Start Guide

Before building Gunrock make sure you have **CUDA 7.5 or higher** (recommended CUDA 9 or higher) installed on your Linux system. We also support building Gunrock on docker images using the provided docker files under `docker` subdirectory. For complete build guide, see [Building Gunrock](https://gunrock.github.io/docs/#building-gunrock).

<pre class="highlight mid-column-code shell tab-shell">
<code>git clone --recursive https://github.com/gunrock/gunrock/
cd gunrock
mkdir build && cd build
cmake ..
make -j$(nproc)
make test</code>
</pre>

## Getting Started with Gunrock

- To learn more about Gunrock and its programming model, see [Gunrock's Overview](http://gunrock.github.io/docs/#overview).
- For information on building Gunrock, see [Building Gunrock](http://gunrock.github.io/docs/#building-gunrock).
- Tutorial: [How to write a graph primitive within Gunrock]().
- The `examples` subdirectory has a comprehensive list of test applications for most the functionality of Gunrock.
- [API Reference documentation](http://gunrock.github.io/gunrock) (generated using doxygen).
- Find our [publications](http://gunrock.github.io/docs/#publications), [presentations](http://gunrock.github.io/docs/#presentations), and [results and analysis](http://gunrock.github.io/docs/#results-and-analysis).

## Copyright and License

Gunrock is copyright The Regents of the University of California, 2013&ndash;2019. The library, examples, and all source code are released under [Apache 2.0](https://github.com/gunrock/gunrock/blob/master/LICENSE.TXT).
