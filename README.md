<p align="center">
  <a href="https://github.com/gunrock/gunrock/"><img src="https://github.com/gunrock/docs/raw/master/source/images/GunrockLogo150px.png"></a>
  <br>
  <a href="https://github.com/gunrock/gunrock/releases/tag/v0.4"><img src="https://img.shields.io/badge/gunrock-0.4-blue.svg"></a>
  <br>
  <a href="http://mario.ece.ucdavis.edu:8080/job/gunrock/job/dev/"><img src="http://mario.ece.ucdavis.edu:8080/buildStatus/icon?job=gunrock/dev" alt="Build Status"></a>
  <a href="https://github.com/gunrock/gunrock/blob/master/LICENSE.TXT"><img src="https://img.shields.io/github/license/gunrock/gunrock.svg" alt="Apache 2"></a>
  <a href="https://github.com/gunrock/gunrock/issues"><img src="https://img.shields.io/github/issues/gunrock/gunrock.svg" alt="Issues Open"></a>
</p>
<h1 id="gunrock-gpu-graph-analytics" align="center">Gunrock: GPU Graph Analytics</h1>

Gunrock is a CUDA library for graph-processing designed specifically for the GPU. It uses a high-level, bulk-synchronous, data-centric abstraction focused on operations on a vertex or edge frontier. Gunrock achieves a balance between performance and expressiveness by coupling high performance GPU computing primitives and optimization strategies with a high-level programming model that allows programmers to quickly develop new graph primitives with small code size and minimal GPU programming knowledge.

For more details, please visit our [website](http://gunrock.github.io/), read [Why Gunrock](#why-gunrock), our TOPC 2017 paper [Gunrock: GPU Graph Analytics](http://escholarship.org/uc/item/9gj6r1dj), look at our [results](#results), and find more details in our [publications](#publications). See [Release Notes](http://gunrock.github.io/gunrock/doc/latest/release_notes.html) to keep up with the our latest changes.

Gunrock is featured on NVIDIA's list of [GPU Accelerated Libraries](https://developer.nvidia.com/gpu-accelerated-libraries) as the only external library for GPU graph analytics.

<table style="font-size: 12px;"><thead>
<tr>
  <th><strong>Service</strong></th>
  <th><strong>System</strong></th>
  <th><strong>Environment</strong></th>
  <th><strong>Status</strong></th>
</tr>
</thead><tbody>
<tr>
  <td><a href="https://jenkins.io/">Jenkins</a></td>
  <td>Ubuntu 16.04.4 LTS</td>
  <td>CUDA 10.0, GCC/G++ 5.4, Boost 1.58.0</td>
  <td><a href="http://mario.ece.ucdavis.edu:8080/blue/organizations/jenkins/gunrock/activity"><img src="http://mario.ece.ucdavis.edu:8080/buildStatus/icon?job=gunrock/dev" alt="Build Status"></a></td>
</tr>
</tbody></table>

## Quickstart

<pre class="highlight mid-column-code shell tab-shell">
<code>
git clone --recursive https://github.com/gunrock/gunrock/
cd gunrock
mkdir build
cd build
cmake ..
make -j$(nproc)
make test
</code>
</pre>

## Gunrock Source Code

<table style="font-size: 12px;"><thead>
<tr>
  <th><a href="http://gunrock.github.io/gunrock/doc/latest/pages.html">Related Pages</a></th>
  <th><a href="http://gunrock.github.io/gunrock/doc/latest/modules.html">Modules</a></th>
  <th><a href="http://gunrock.github.io/gunrock/doc/latest/namespaces.html">Namespaces</a></th>
  <th><a href="http://gunrock.github.io/gunrock/doc/latest/annotated.html">Data Structures</a></th>
  <th><a href="http://gunrock.github.io/gunrock/doc/latest/files.html">Files</a></th>
</tr>
</thead><tbody>
</tbody></table>

## Getting Started with Gunrock

- For Frequently Asked Questions, see the [FAQ](#faq).

- For information on building Gunrock, see [Building Gunrock](#building-gunrock).

- The "tests" subdirectory included with Gunrock has a comprehensive test application for most the functionality of Gunrock.

- For the programming model we use in Gunrock, see [Programming Model](#programming-model).

- We have also provided code samples for how to use [Gunrock's C interface](https://github.com/gunrock/gunrock/tree/master/shared_lib_tests) and how to [call Gunrock primitives from Python](https://github.com/gunrock/gunrock/tree/master/python), as well as [annotated code](http://gunrock.github.io/gunrock/doc/annotated_primitives/annotated_primitives.html) for two typical graph primitives.

- For details on upcoming changes and features, see the [Road Map](http://gunrock.github.io/gunrock/doc/latest/road_map.html).

## Results and Analysis

We are gradually adding summaries of our results to these web pages (please let us know if you would like other comparisons). These summaries also include a table of results along with links to the configuration and results of each individual run. We detail our [methodology for our measurements here](#methodology-for-graph-analytics-performance).

- [Gunrock performance compared with other engines for graph analytics](https://gunrock.github.io/docs/engines_topc.html)
- [Setting parameters for direction-optimized BFS](http://gunrock.github.io/gunrock/doc/latest/md_stats_do_ab_random.html)
- [Gunrock results on different GPUs](https://gunrock.github.io/docs/gunrock_gpus.html)
- [Gunrock BFS throughput as a function of frontier size](https://gunrock.github.io/docs/frontier.html)
- [Multi-GPU Gunrock Speedups](https://gunrock.github.io/docs/mgpu_speedup.html) and [Multi-GPU Gunrock Scalability](https://gunrock.github.io/docs/mgpu_scalability.html)
- [Multi-GPU Gunrock Partition Performance](https://gunrock.github.io/docs/mgpu_partition.html)
- [Comparison to Groute](http://gunrock.github.io/docs/groute.html)

For reproducibility, we maintain Gunrock configurations and results in our github [gunrock/io](https://github.com/gunrock/io/tree/master/gunrock-output) repository.

We are happy to run experiments with other engines, particularly if those engines output results in our JSON format / a format that can be easily parsed into JSON format.

## Reporting Problems

To report Gunrock bugs or request features, please file an issue directly using [Github](https://github.com/gunrock/gunrock/issues).

<!-- TODO: Algorithm Input Size Limitations -->

## Publications

Yuechao Pan, Roger Pearce, and John D. Owens. **Scalable Breadth-First Search on a GPU Cluster**. In Proceedings of the 31st IEEE International Parallel and Distributed Processing Symposium, IPDPS 2018, May 2018. [[http](https://escholarship.org/uc/item/9bd842z6)]

Yangzihao Wang, Yuechao Pan, Andrew Davidson, Yuduo Wu, Carl Yang, Leyuan Wang, Muhammad Osama, Chenshan Yuan, Weitang Liu, Andy T. Riffel, and John D. Owens. **Gunrock: GPU Graph Analytics**. ACM Transactions on Parallel Computing, 4(1):3:1&ndash;3:49, August 2017. [[DOI](http://dx.doi.org/10.1145/3108140) | [http](http://escholarship.org/uc/item/9gj6r1dj)]

Yuechao Pan, Yangzihao Wang, Yuduo Wu, Carl Yang, and John D. Owens.
**Multi-GPU Graph Analytics**.  In Proceedings of the 31st IEEE International Parallel and Distributed Processing Symposium, IPDPS 2017, pages 479&ndash;490, May/June 2017.
[[DOI](http://dx.doi.org/10.1109/IPDPS.2017.117) |
[http](http://escholarship.org/uc/item/39r145g1)]

Yangzihao Wang, Sean Baxter, and John D. Owens. **Mini-Gunrock: A Lightweight Graph Analytics Framework on the GPU**. In Graph Algorithms Building Blocks, GABB 2017, pages 616&ndash;626, May 2017. [[DOI](http://dx.doi.org/10.1109/IPDPSW.2017.116) | [http](https://escholarship.org/uc/item/5wm061tr)]

Leyuan Wang, Yangzihao Wang, Carl Yang, and John D. Owens. **A Comparative Study on Exact Triangle Counting Algorithms on the GPU**. In Proceedings of the 1st High Performance Graph Processing Workshop, HPGP '16, pages 1&ndash;8, May 2016.
[[DOI](http://dx.doi.org/10.1145/2915516.2915521) |
[http](http://www.escholarship.org/uc/item/9hf0m6w3)]

Yangzihao Wang, Andrew Davidson, Yuechao Pan, Yuduo Wu, Andy Riffel, and John D. Owens.
**Gunrock: A High-Performance Graph Processing Library on the GPU**.
In Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, [PPoPP '16](http://conf.researchr.org/home/ppopp-2016), pages 11:1&ndash;11:12, March 2016. Distinguished Paper. [[DOI](http://dx.doi.org/10.1145/2851141.2851145) | [http](http://escholarship.org/uc/item/6xz7z9k0)]

Yuduo Wu, Yangzihao Wang, Yuechao Pan, Carl Yang, and John D. Owens.
**Performance Characterization for High-Level Programming Models for GPU Graph
Analytics**. In IEEE International Symposium on Workload Characterization,
IISWC-2015, pages 66&ndash;75, October 2015. Best Paper finalist. [[DOI](http://dx.doi.org/10.1109/IISWC.2015.13) | [http](http://escholarship.org/uc/item/2t69m5ht)]

Carl Yang, Yangzihao Wang, and John D. Owens.
**Fast Sparse Matrix and Sparse Vector Multiplication Algorithm on the GPU**.
In Graph Algorithms Building Blocks, GABB 2015, pages 841&ndash;847, May 2015.
[[DOI](http://dx.doi.org/10.1109/IPDPSW.2015.77) | [http](http://www.escholarship.org/uc/item/1rq9t3j3)]

Afton Geil, Yangzihao Wang, and John D. Owens.
**WTF, GPU! Computing Twitter's Who-To-Follow on the GPU**.
In Proceedings of the Second ACM Conference on Online Social Networks,
COSN '14, pages 63&ndash;68, October 2014.
[[DOI](http://dx.doi.org/10.1145/2660460.2660481) | [http](http://escholarship.org/uc/item/5xq3q8k0)]

## Presentations

GTC 2018, **Latest Development of the Gunrock Graph Processing Library on GPUs**, March 2018. [[slides](http://on-demand.gputechconf.com/gtc/2018/presentation/s8594-latest-development-of-the-gunrock-graph-processing-library-on-gpus.pdf) | [video](http://on-demand.gputechconf.com/gtc/2018/video/S8594/)]

GTC 2018, **Writing Graph Primitives with Gunrock**, March 2018. [[slides](https://github.com/gunrock/gunrock/blob/master/doc/Writing-Gunrock-Primitives.pdf) | [video](http://on-demand.gputechconf.com/gtc/2018/video/S8586/)]

GTC 2016, **Gunrock: A Fast and Programmable Multi-GPU Graph Processing Library**, April 2016. [[slides](http://on-demand.gputechconf.com/gtc/2016/presentation/s6374-yangzihao-wang-gunrock.pdf)]

NVIDIA [webinar](http://info.nvidianews.com/gunrock-webinar-reg-0416.html), April 2016. [[slides](http://tinyurl.com/owens-nv-webinar-160426)]

GPU Technology Theater at SC15, **Gunrock: A Fast and Programmable Multi-GPU Graph processing Library**, November 2015. [[slides](http://images.nvidia.com/events/sc15/pdfs/SC5139-gunrock-multi-gpu-processing-library.pdf) | [video](http://images.nvidia.com/events/sc15/SC5139-gunrock-multi-gpu-processing-library.html)]

GTC 2014, **High-Performance Graph Primitives on the GPU: design and Implementation of Gunrock**, March 2014. [[slides](http://on-demand.gputechconf.com/gtc/2014/presentations/S4609-hi-perf-graph-primitives-on-gpus.pdf) | [video](http://on-demand.gputechconf.com/gtc/2014/video/S4609-hi-perf-graph-primitives-on-gpus.mp4)]

## Gunrock Developers

- [Yangzihao Wang](http://www.idav.ucdavis.edu/~yzhwang/),
  University of California, Davis

- [Yuechao Pan](https://sites.google.com/site/panyuechao/home), University of California, Davis

- [Yuduo Wu](http://www.yuduowu.com/),
  University of California, Davis

- [Carl Yang](http://web.ece.ucdavis.edu/~ctcyang/),
  University of California, Davis

- [Leyuan Wang](http://www.ece.ucdavis.edu/~laurawly/),
  University of California, Davis

- Weitang Liu, University of California, Davis

- [Muhammad Osama](http://www.ece.ucdavis.edu/~mosama/),
  University of California, Davis

- Chenshan Shari Yuan, University of California, Davis

- Andy Riffel, University of California, Davis

- [Huan Zhang](http://www.huan-zhang.com/),
  University of California, Davis

- [John Owens](http://www.ece.ucdavis.edu/~jowens/),
  University of California, Davis

## Acknowledgments

Thanks to the following developers who contributed code: The connected-component implementation was derived from code written by Jyothish Soman, Kothapalli Kishore, and P. J. Narayanan and described in their IPDPSW '10 paper *A Fast GPU Algorithm for Graph Connectivity* ([DOI](http://dx.doi.org/10.1109/IPDPSW.2010.5470817)). The breadth-first search implementation and many of the utility functions in Gunrock are derived from the [b40c](http://code.google.com/p/back40computing/) library of [Duane Merrill](https://sites.google.com/site/duanemerrill/). The algorithm is described in his PPoPP '12 paper *Scalable GPU Graph Traversal* ([DOI](http://dx.doi.org/10.1145/2370036.2145832)). Thanks to Erich Elsen and Vishal Vaidyanathan from [Royal Caliber](http://www.royal-caliber.com/) and the [Onu](http://www.onu.io/) Team for their discussion on library development and the dataset auto-generating code. Thanks to Adam McLaughlin for his technical discussion. Thanks to Oded Green for his technical discussion and an optimization in the CC primitive. Thanks to the [Altair](https://altair-viz.github.io/) and [Vega-lite](https://vega.github.io/vega-lite/) teams in the [Interactive Data Lab](http://idl.cs.washington.edu/) at the University of Washington for graphing help. We appreciate the technical assistance, advice, and machine access from many colleagues at NVIDIA: Chandra Cheij, Joe Eaton, Michael Garland, Mark Harris, Ujval Kapasi, David Luebke, Duane Merrill, Josh Patterson, Nikolai Sakharnykh, and Cliff Woolley.

This work was funded by the DARPA HIVE program under AFRL Contract FA8650-18-2-7835, the DARPA XDATA program under AFRL Contract FA8750-13-C-0002, by NSF awards OAC-1740333, CCF-1629657, OCI-1032859, and CCF-1017399, by DARPA STTR award D14PC00023, and by DARPA SBIR award W911NF-16-C-0020. Our XDATA principal investigator was Eric Whyne of [Data Tactics Corporation](http://www.data-tactics.com/) and our DARPA program manager is [Mr. Wade Shen](http://www.darpa.mil/staff/mr-wade-shen) (since 2015), and before that Dr. Christopher White (2012&ndash;2014). Thanks to Chris, Wade, and DARPA business manager Gabriela Araujo for their support during the XDATA program.

## Copyright and Software License

Gunrock is copyright The Regents of the University of California, 2013&ndash;2018. The library, examples, and all source code are released under [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).
