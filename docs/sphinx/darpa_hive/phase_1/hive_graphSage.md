# GraphSAGE

GraphSAGE is a way to fit graphs into a neural network: instead of getting the
embedding of a vertex from all its neighbors' features as in conventional
implementations, GraphSAGE selects some 1-hop neighbors, some 2-hop neighbors
connected to those 1-hop neighbors, and computes the embedding based on the
features of the 1-hop and 2-hop neighbors. The embedding can be considered as
a vector containing hash values describing the interesting properties of a
vertex.

During the training process, the adjacency matrix and features of the nodes in the graph are fed into a neural network. The
parameters (the `W` arrays in the algorithm) are updated after each batch by the
difference between the predicted labels and the real labels. The per-vertex
features won't change, but the parameters will, so the GraphSAGE computation needs
to be performed for each batch. Ultimately it should connect to the training
process to complete a workflow. However, the training part is pure matrix
operations, and the year 1 deliverable only focuses on the graph related
portion, which is the GraphSAGE implementation.

## Summary of Results

The vertex embedding part of the GraphSAGE algorithm is implemented in the
Gunrock framework using custom CUDA kernels, utilizing block-level
parallelism, that allow a shorter running time. For the embedding part alone, the GPU
implementation is 7.5X to 15X on P100, and 20X to 30X on V100,
faster than an OpenMP implementation using 32 threads. The GPU hardware, especially
the memory system, has high utilizations from these custom kernels. It is still
unclear how to expose block-level parallelism for more general usage in
other applications in Gunrock.

Connecting the vertex embedding with the neural network training part, and
making the GraphSAGE workflow complete, would be an interesting task for year 2.
Testing on the complete workflow for prediction accuracy and running speed will
be more meaningful.

## Summary of Gunrock Implementation

Gunrock's implementation for the 1st year is only the embedding / inferencing
phase, without the training phase. It is based on Algorithm 2 with K=2 of the GraphSAGE
paper ("Inductive Representation Learning on Large Graphs",
<https://arxiv.org/abs/1706.02216>).

Given a graph G, the inputs are per-vertex features, weight matrices W^k, and a
non-linear activation function (ReLu); the output from GraphSAGE is the embedding vector
of each vertex. The current Gunrock implementation randomly selects neighbors
from the neighborhood, and simple changes to the code can enable other
selection methods, such as weighted uniform, importance sampling (used by
FaseGCN), or a random walk probability like DeepWalk or Node2Vec (used by
PinSage). An aggregator is a function to accumulate data from the
selected neighbors; the current implementation uses the Mean aggregator,
and it can be changed to other accumulation functions easily.

The pseudocode of Gunrock's implementation follows. The current
implementation uses custom CUDA kernels, because block-level parallelism is critical for
faster running speed; all other functions, such as memory management,
load and store routines, block level parallel reduction, and graph accesses
(get degree, get neighbors, etc.) are provided by the framework or the utility
functions of Gunrock. The GraphSAGE  algorithm uses B2, B1 and B0 for the
sources, the 1-hop neighbors and the 2-hop neighbors; for
easier understanding of the code, we use `sources`, `children` and `leafs` for
these three groups of vertices instead. To manage the memory usage of
intermediate data, batches of source vertices are processed one by one, with each
of size B.

```
source_start := 0;
While (source_source < V)
    num_sources := (source_start + B > V) ? V - source_start : B;

    // Kernel1: pick children and leafs
    For children# from 0 to num_children_per_source x num_sources:
        source := source_start + (children# / num_children_per_source);
        child := SelectNeighbor(source);
        leafs_feature := {0};
        For i from 1 to num_leafs_per_child:
            leaf := SelectNeighbor(child);
            leafs_feature += feature[leaf];

        children[children#] := child;
        leafs_features[children#] := leafs_feature;

    // Kernel2: Child-centric computation
    For children# from 0 to num_children_per_source x num_sources:
        child_temp := ReLu(concatenate(
            feature[children[children#]] x Wf1,
            Mean(leafs_features[children#]) x Wa1));
        child_temp /= sqrt(sum(child_temp));
        children_temp    [children# / num_children_per_source] += child_temp;
        children_features[children# / num_children_per_source] += feature[child];

    // Kernel3: Source-centric computation
    For source# from 0 to num_sources - 1:
        source := source_start + source#;
        source_temp := ReLu(concatenate(
            feature[source] x Wf1,
            Mean(children_features[source#] x Wa1)));
        source_temp /= sqrt(sum(source_temp));

        result := ReLu(concatenate(
            source_temp x Wf2,
            Mean(children_temp[source#]) x Wa2));
        result /= sqrt(sum(result));
        source_embedding[source#] := result;

    Move source_embedding from GPU to CPU;
    source_start += B;
```

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

CUDA should have been installed; `$PATH` and `$LD_LIBRARY_PATH` should have been
set correctly to use CUDA. The current Gunrock configuration assumes boost
(1.58.0 or 1.59.0) and Metis are installed; if not, changes need to be made in
the Makefiles. DARPA's DGX-1 has both installed when the tests are performed.

```
git clone --recursive https://github.com/gunrock/gunrock/ \
  -b dev-refactor
cd gunrock
git submodule init
git submodule update
mkdir build
cd build
cmake ..
cd ../tests/sage
make
```

At this point, there should be an executable `test_sage_<CUDA version>_x86_64`
in `tests/sage/bin`.

The datasets are assumed to have been placed in `/raid/data/hive`, and converted
to proper matrix market format (`.mtx`). At the time of testing, `pokec`, `amazon`,
`flickr`, `twitter` and `cit-Patents` are available in that directory.

Note that GraphSage is an inductive representation learning algorithm,
so it reasonable to assume that there is no dangling vertices in the graph.
**Before running GraphSAGE, please remove the dangling vertices from the graph.**
In the case when dangling vertices are present, the dangling vertices themselves
will be treated as their neighbors.

The testing is done with Gunrock using `dev-refactor` branch at commit `0ed72d5`
(Oct. 25, 2018), using CUDA 9.2 with NVIDIA driver 390.30 on a Tesla P100 GPU in
the DGX-1 machine, and using CUDA 10.0 with NVIDIA driver 410.66 on a Tesla V100
GPU in Bowser (an machine used by the Gunrock team in UC Davis).

### Running the application

#### Application specific parameters

The W arrays used by GraphSAGE should be produced by the training process; without
the training part at the moment, we use some given datasets or randomly generate
them if not available. The array input files are floating-point values in plain
text format; examples can be found in the `gunrock/app/sage` directory of the
gunrock repo.

```
--Wa1 : std::string, default =
        <weight matrix for W^1 matrix in algorithm 2, aggregation part>
         dimension 64 by 128 for pokec;
         It should be leaf feature length by a value you want for W2 layer
--Wa1-dim1 : int, default = 128
        Wa1 matrix column dimension
--Wa2 : std::string, default =
        <weight matrix for W^2 matrix in algorithm 2, aggregation part>
         dimension 256 by 128 for pokec;
         It should be child_temp length by output length
--Wa2-dim1 : int, default = 128
        Wa2 matrix column dimension
--Wf1 : std::string, default =
        <weight matrix for W^1 matrix in algorithm 2, feature part>
         dimension 64 by 128 for pokec;
         It should be child feature length by a value you want for W2 layer
--Wf1-dim1 : int, default = 128
        Wf1 matrix column dimension
--Wf2 : std::string, default =
        <weight matrix for W^2 matrix in algorithm 2, feature part>
         dimension 256 by 128 for pokec;
         It should be source_temp length by output length
--Wf2-dim1 : int, default = 128
        Wf2 matrix column dimension
--feature-column : std::vector<int>, default = 64
        feature column dimension
--features : std::string, default =
        <features matrix>
    dimension |V| by 64 for pokec;
--omp-threads : int, default = 32
    number of threads to run CPU reference
--num-children-per-source : std::vector<int>, default = 10
        number of sampled children per source
--num-leafs-per-child : std::vector<int>, default = -1
        number of sampled leafs per child; default is the same as num-children-per-source
--batch-size : std::vector<int>, default = 65536
        number of source vertex to process in one iteration
```

#### Example Command
```bash
./bin/test_sage_9.2_x86_64 \
 market /raid/data/hive/pokec/pokec.mtx  --undirected \
 --Wf1 ../../gunrock/app/sage/wf1.txt \
 --Wa1 ../../gunrock/app/sage/wa1.txt \
 --Wf2 ../../gunrock/app/sage/wf2.txt \
 --Wa2 ../../gunrock/app/sage/wa2.txt \
 --features ../../gunrock/app/sage/features.txt \
 --num-runs=10 \
 --batch-size=16384
```

When `Wf1`, `Wa1`, `Wf2`, `Wa2`, or `features` are not available, random values
are used. The embeddings may be not useful at all, but the memory access patterns
and the computation workload are the same, regardless of whether the random
inputs are used. Using the random inputs enable us to test on any graph without
requiring the associative arrays.

### Output

The outputs are in the `sage` directory; look for the .txt files. An example is
shown below for the `pokec` dataset:

```
Loading Matrix-market coordinate-formatted graph ...
  ... Loading progress ...
Converting 1632803 vertices, 44603928 directed edges ( ordered tuples) to CSR format...Done (0s).
Degree Histogram (1632803 vertices, 44603928 edges):
    Degree 0: 0 (0.000000 %)
    Degree 2^0: 163971 (10.042301 %)
    Degree 2^1: 201604 (12.347111 %)
    Degree 2^2: 238757 (14.622523 %)
    Degree 2^3: 273268 (16.736128 %)
    Degree 2^4: 298404 (18.275567 %)
    Degree 2^5: 265002 (16.229882 %)
    Degree 2^6: 148637 (9.103180 %)
    Degree 2^7: 38621 (2.365319 %)
    Degree 2^8: 4089 (0.250428 %)
    Degree 2^9: 418 (0.025600 %)
    Degree 2^10: 23 (0.001409 %)
    Degree 2^11: 3 (0.000184 %)
    Degree 2^12: 3 (0.000184 %)
    Degree 2^13: 3 (0.000184 %)

==============================================
 feature-column=64 num-children-per-source=10 num-leafs-per-child=-1
Computing reference value ...
==============================================
rand-seed = 1540498523
==============================================
CPU Reference elapsed: 25242.941406 ms.
Embedding validation: PASS
==============================================
 batch-size=512
Using randomly generated Wf1
Using randomly generated Wa1
Using randomly generated Wf2
Using randomly generated Wa2
Using randomly generated features
Using advance mode LB
Using filter mode CULL
rand-seed = 1540498553
==============================================
==============================================
Run 0 elapsed: 2047.366858 ms, #iterations = 3190
... More runs
==============================================
==============================================
Run 9 elapsed: 2013.216972 ms, #iterations = 3190
Embedding validation: PASS
[Sage] finished.
 avg. elapsed: 2042.503190 ms
 iterations: 3190
 min. elapsed: 2009.524107 ms
 max. elapsed: 2243.710995 ms
 load time: 1600.46 ms
 preprocess time: 3997.880000 ms
 postprocess time: 2106.487989 ms
 total time: 26634.360075 ms
```

There is 1 OMP reference run on the CPU for each combination of {`feature-column`,
`num-children-per-source`, `num-leafs-per-child`}, with the timing reported after
`CPU Reference elapsed`. There are 10 GPU runs for each combination of the
previous three parameters, plus `batch-size`. The computation workload of the
GPU runs are the same as the reference CPU run, with different batch sizes. The
GPU timing is reported after `Run x elapsed:`, and the average running time of
the 10 GPUs is reported after `avg. elapsed`.

The mathematical formula of the GraphSAGE algorithm is relatively simple and repeats
itself three times in the form of `Normalize(C x Wf + Mean(D) x Wa)`. Because of
this simplicity, it is still possible to verify the implementation by visually
inspecting the code. The resulted embeddings are also checked for the L2 norm,
which should be close to 1 for every vertex. Because the neighbor selection
process is inherently random, it would be very difficult to do a number-by-number checking with other implementations, including the reference. A more
meaningful regression test will look at the training or validation
accuracy when the full workflow is completed, which is a possibility for
future work in year 2.

## Performance and Analysis

The OpenMP and GPU implementations are measured for runtime. It would be
additionally useful to validate the successful rate of the whole pipeline with the training process in place (perhaps in year 2).

The datasets used for experiments are:

| Dataset | V       | E        |
|---------|--------:|---------:|
| flickr  | 105938  | 4633896  |
| amazon  | 548551  | 1851744  |
| pokec   | 1632803 | 44603928 |
| cit-Patents | 3774768 | 33037894 |
| twitter | 7199978 | 43483326 |
| europe-osm | 50912018 | 108109320 |

The running times in milliseconds are listed below, for both machines. F stands
for the length of features, C stands for the number of children per source,
the same as the number of leafs per child, and B notes the batch size that
produces the shortest running time on GPU among {512, 1024, 2048, 4096, 8192,
16384}.

The running time on DGX-1 with Tesla P100 GPU:

| Dataset | F   | C   | B     | Gunrock GPU | OpenMP   | Speedup |
|---------|----:|----:|------:|----------:|-----------:|-----:|
| flickr  |  64 |  10 | 16384 |   113.431 |   1607.468 | 14.17|
| flickr  |  64 |  25 | 16384 |   248.523 |   2630.659 | 10.59|
| flickr  |  64 | 100 |  2048 |  1139.007 |  10702.981 |  9.40|
| flickr  | 128 |  10 | 16384 |   192.916 |   1800.150 |  9.33|
| flickr  | 128 |  25 |  8192 |   442.226 |   4022.799 |  9.10|
| flickr  | 128 | 100 |  2048 |  2116.955 |  20655.732 |  9.76|
| amazon  |  64 |  10 | 16384 |   579.342 |   8905.530 | 15.37|
| amazon  |  64 |  25 | 16384 |  1235.168 |  18034.229 | 14.60|
| amazon  |  64 | 100 |  4096 |  5486.910 |  45337.011 |  8.26|
| amazon  | 128 |  10 | 16384 |   976.759 |   9418.961 |  9.64|
| amazon  | 128 |  25 | 16384 |  2193.159 |  29645.709 | 13.52|
| amazon  | 128 | 100 |  4096 | 10112.228 |  80677.594 |  7.98|
| pokec   |  64 |  10 | 16384 |  1744.602 |  25242.941 | 14.47|
| pokec   |  64 |  25 | 16384 |  3806.404 |  34517.180 |  9.07|
| pokec   |  64 | 100 |  2048 | 17486.692 | 133920.000 |  7.66|
| pokec   | 128 |  10 | 16384 |  2950.606 |  41414.535 | 14.04|
| pokec   | 128 |  25 |  8192 |  6791.829 |  74983.391 | 11.04|
| pokec   | 128 | 100 |  2048 | 32000.378 | 297979.438 |  9.31|
| cit-Patents |  64 |  10 | 16384 |  4005.860 |  52094.125 | 13.00|
| cit-Patents |  64 |  25 | 16384 |  8541.500 |  93715.789 | 10.97|
| cit-Patents |  64 | 100 |  4096 | 38494.688 | 293333.781 |  7.62|
| cit-Patents | 128 |  10 | 16384 |  6784.752 |  95639.117 | 14.10|
| cit-Patents | 128 |  25 |  8192 | 15250.170 | 146539.250 |  9.61|
| cit-Patents | 128 | 100 |  4096 | 70074.883 | 530910.375 |  7.58|
| twitter |  64 |  10 | 16384 |  7594.364 | 103753.961 | 13.66|
| twitter |  64 |  25 | 16384 | 16212.790 | 162819.531 | 10.04|
| twitter |  64 | 100 |  4096 | 73488.745 | 636224.625 |  8.66|
| twitter | 128 |  10 | 16384 | 12753.125 | 147705.375 | 11.58|
| twitter | 128 |  25 | 16384 | 28827.354 | 282497.438 |  9.80|
| twitter | 128 | 100 |  4096 |134027.966 |1105741.500 |  8.25|
| europe_osm | 64 |  10 | 16384 |  53521.982 |  611449.625 | 11.42 |
| europe_osm | 64 |  25 | 16384 | 113740.739 | 1016479.938 |  8.94 |
| europe_osm | 64 | 100 |  4096 | 509313.472 | 4441408.500 |  8.72 |

The running time on Bowser with a Tesla V100 GPU

| Dataset | F   | C   | B     | Gunrock GPU | OpenMP   | Speedup vs. CPU| Speedup vs. P100 |
|---------|----:|----:|------:|----------:|-----------:|-----:|-----:|
| flickr  |  64 |  10 | 16384 |    39.894 |   1094.377 | 27.43| 2.90 |
| flickr  |  64 |  25 |  8192 |    91.810 |   2177.953 | 23.72| 2.76 |
| flickr  |  64 | 100 |  1024 |   448.888 |   8641.063 | 19.25| 2.57 |
| flickr  | 128 |  10 | 16384 |    65.131 |   1849.658 | 28.40| 2.95 |
| flickr  | 128 |  25 |  4096 |   156.965 |   3981.435 | 25.37| 2.84 |
| flickr  | 128 | 100 |   512 |   802.966 |  16471.338 | 20.51| 2.70 |
| amazon  |  64 |  10 | 16384 |   196.079 |   6142.780 | 31.33| 2.95 |
| amazon  |  64 |  25 |  8192 |   423.145 |  12674.514 | 29.95| 2.92 |
| amazon  |  64 | 100 |  2048 |  1963.691 |  49205.359 | 25.06| 2.79 |
| amazon  | 128 |  10 | 16384 |   324.067 |   9978.198 | 30.79| 3.10 |
| amazon  | 128 |  25 |  8192 |   733.181 |  21726.697 | 29.63| 2.99 |
| amazon  | 128 | 100 |  2048 |  3361.894 |  86967.164 | 25.87| 3.01 |
| pokec   |  64 |  10 | 16384 |   602.116 |  17111.664 | 28.42| 2.84 |
| pokec   |  64 |  25 |  8192 |  1379.450 |  35887.129 | 26.02| 2.71 |
| pokec   |  64 | 100 |  1024 |  6793.011 | 140751.234 | 20.72| 2.54 |
| pokec   | 128 |  10 | 16384 |  1000.310 |  28987.953 | 28.98| 2.96 |
| pokec   | 128 |  25 |  4096 |  2366.202 |  63863.352 | 26.99| 2.82 |
| pokec   | 128 | 100 |   512 | 11859.789 | 265325.281 | 22.37| 2.64 |
| cit-Patents |  64 |  10 | 16384 |  1374.782 |  38803.195 | 28.22| 2.95 |
| cit-Patents |  64 |  25 |  8192 |  3006.717 |  78112.516 | 25.98| 2.87 |
| cit-Patents |  64 | 100 |  2048 | 13910.529 | 291278.125 | 20.94| 2.78 |
| cit-Patents | 128 |  10 | 16384 |  2276.261 |  65735.234 | 28.88| 2.99 |
| cit-Patents | 128 |  25 |  8192 |  5201.184 | 141957.922 | 27.29| 2.96 |
| cit-Patents | 128 | 100 |  2048 | 23922.527 | 560321.438 | 23.42| 2.94 |
| twitter |  64 |  10 | 16384 |  2575.303 |  72372.484 | 28.10| 2.91 |
| twitter |  64 |  25 |  8192 |  5658.345 | 148938.422 | 26.32| 2.84 |
| twitter |  64 | 100 |  2048 | 26468.915 | 569925.500 | 21.53| 2.77 |
| twitter | 128 |  10 | 16384 |  4270.454 | 125431.766 | 29.37| 2.98 |
| twitter | 128 |  25 |  8192 |  9744.686 | 267841.563 | 27.49| 2.93 |
| twitter | 128 | 100 |  2048 | 45604.444 |1098039.375 | 24.08| 2.93 |
| europe_osm | 64 |  10 | 16384 |  18440.253 |  497153.969 | 26.96 | 2.90 |
| europe_osm | 64 |  25 | 16384 |  39883.421 | 1008675.500 | 25.29 | 2.85 |
| europe_osm | 64 | 100 |  4096 | 184128.123 | 3825227.500 | 20.77 | 2.77 |

### Implementation limitations

- **Memory usage** Gunrock's GPU implementation uses `B x (C x (Wf2.x + F + 1) +
2Wf2.x + 2R.x) x 4` bytes in additional to the features that takes up `VF x 4`
bytes and the graph itself. Because the batch size B can be adjusted, the main
memory consumption is from the feature array. For a P100 with 16 GB memory, if
the feature length is 64, the maximum number of vertices it can handle before
hitting OOM is about 60 million. The largest dataset tested so far is the
`europe_osm` dataset with 50.9M vertices and 108M edges.

- **Data types** The vertex Ids and edge Ids are both presented as 32-bit unsigned
integers. Input features, weights, output embeddings and intermedia results are
represented as 32-bit floating-point numbers. A not so recent trend in machine
learning research is to use less precision in neural networks. Half precision /
16-bit floating points are quite common, and supported by recent GPUs. Using
half precision cuts the computation time to about half, as compared to single
precision, and also cuts the memory usage to store the features in half. It would
be interesting to see what would happen if the data type is changed to half
precision.

- **Graph types** The training process requires the graph to be undirected
(enforced by `--undirected` in Gunrock's command line parameters). The behavior of
sampling a zero-length neighbor list is undefined, and currently it will return
the source vertex itself.

### Comparison against existing implementations

Because computation on each vertex is independent from other vertices, GraphSAGE is an
embarrassingly parallel problem when parallelized across vertices. Running a simple
test with the `pokec` dataset, feature length as 64, num_children_per_source and
num_leafs_per_child both at 10, the serial run (with omp-threads forced to 1)
on the DGX-1 takes 366390.250 ms, as compared to 25242.941 ms using 32 threads;
using 32 threads is about 14.5X faster than a single thread, which shows GraphSAGE
scales pretty well on the CPU.

Comparing the running time of a 32-thread OpenMP and Gunrock's GPU implementation,
the P100 is about 7.5X to 15X faster. Increasing the feature length from 64 to
128 roughly doubles the computation workload, and the speedup does not change
very much for datasets with more than about 1M vertices. However, increasing the
number of children per source and the number of leafs per child decreases the
speedup. The OMP's running time increases slower than the number of neighbors
selected, while the GPU's running time increases faster than the number of neighbors
selected. This may be attributed to the cache effect: the parallelism on CPU is
limited, so data reuse rate is high, even when the number of neighbors increases;
however, the number of source or children processed on the GPU at the same time
is much larger, with a much bigger working set, and this decreases the cache hit
rate, so results in longer running time.

Also interesting is comparing the runtime on V100 and P100: V100 is
about 3X faster than P100 when running GraphSAGE. This large performance difference
is caused by the different limiting factors when running GraphSAGE on these two GPUs;
details are in the the Performance limitations section. Compared to OpenMP, the V100
is about 20X to 30X faster.

### Performance limitations

We profiled GraphSAGE with the `pokec` dataset on a Titan Xp GPU (profiling on P100
caused an internal error in the profiler itself; Titan Xp has roughly the same SM
design as P100, but has only 30 SMs vs. 56 on the P100; P100 also has 16 GB
HMB2 memory, and Titan Xp only has 12 GB GDDR5X; runtime on Titan Xp and
P100 is similar). kernel2 takes up about 60% of
the computation of an batch, 10.64 ms out of 17.76 ms for `pokec` with 64 feature
length, 10 neighbors, and batch size at 16384. Kernel1 takes 1.56 ms, or 8.8%,
and Kernel3 takes 5.52 ms, or 31.1%.

Further detail on the profile of Kernel2 on the Titan Xp shows the
utilization of the memory system:

| Type        | Transactions | Bandwidth     | Utilization |
|---------------|-----------:|--------------:|-------------|
| Shared Loads  |    2129920 |   31.997 GB/s | |
| Shared Stores |    1638400 |   24.613 GB/s | |
| Shared Total  |    3768320 |   56.611 GB/s | Idle to low |
| Local Loads   |          0 |        0 GB/s | |
| Local Stores  |          0 |        0 GB/s | |
| Global Loads  | 2685009922 | 1575.871 GB/s | |
| Global Stores |          0 |        0 GB/s | |
| Texture Reads |  671252480 | 2521.025 GB/s | |
| **Unified Total** | **3356262402** | **4096.897 GB/s** | **High** |
| L2 Reads      |  264170192 |  992.145 GB/s | |
| L2 Writes     |   10485773 |   39.381 GB/s | |
| L2 Total      |  274655965 | 1031.526 GB/s | Low to medium |
| Device Reads  |    2181833 |    8.194 GB/s | |
| Device Writes |     543669 |    2.042 GB/s | |
| Device Total  |    2725502 |   10.236 GB/s | Idle to low |

![Sage_Pokec_TianXp](../attachments/sage/pokec_TitanXp.png "Memory statics")

It's clear that the unified cache is almost fully utilized, at 4 TBps out of
the 5 TBps theoretical upper bound, and is the bottleneck.
This is because the `W` arrays and the intermediate arrays are highly reusable.

Running the same experiment on the V100 shows a different picture within the memory
system:

| Type        | Transactions | Bandwidth     | Utilization |
|---------------|-----------:|--------------:|-------------|
| Shared Loads  |    2138599 |   74.166 GB/s | |
| Shared Stores |    1640842 |   56.904 GB/s | |
| Shared Total  |    3779441 |  131.070 GB/s | Idle to low |
| Local Loads   |          0 |        0 GB/s | |
| Local Stores  |          0 |        0 GB/s | |
| Global Loads  |  419594240 | 3637.862 GB/s | |
| Global Stores |          0 |        0 GB/s | |
| Texture Reads |  177448960 | 6153.897 GB/s | |
| Unified Total |  597043200 | 9791.759 GB/s | Medium |
| L2 Reads      |    8209326 |   71.174 GB/s | |
| L2 Writes     |    5243176 |   45.458 GB/s | |
| L2 Total      |  274655965 |  116.633 GB/s | Idle to low |
| Device Reads  |    2402833 |   20.832 GB/s | |
| Device Writes |     571925 |    4.959 GB/s | |
| Device Total  |    2974758 |   25.791 GB/s | Idle to low |

![Sage_Pokec_V100](../attachments/sage/pokec_V100.png "Memory statics")

On the V100, kernel2 only takes 3.982 ms (about 60% of 6.67 ms per batch), and the
unified-cache throughput increases to 9.8 TBps, more than double than on the
Titan Xp. In fact, the theoretical upper bound of V100's L1 throughput is 28
TBps, resulted from doubling each SM's load throughput from 128 bytes per
cycle to 256 bytes per cycle, and increasing the SM count to 80. This is the
reason why the V100 can outperform P100 and Titan Xp by about 3X. The performance
bottleneck is no longer the memory system, and switches to integer
computations, which comes mainly from array index calculation. This particular
kernel also takes up 32 registers per thread, which is the limit for full GPU
occupancy. Storing intermediate index calculation results would help if the
register usage is not so high.

## Next Steps

### Alternate approaches

**Things we tried that didn't really work**

An simple implementation that uses a thread to process the per-source or per-child
computation is coded, but it runs about 10X slower than the current
implementation that uses a block to process such units. One reason is the use
of block-level parallel primitives, such as block reduce. Another reason is that
by using a whole block, instead of a thread, to process the same computation,
the working set is greatly reduced together with the parallelism, and the whole
working set can fit into the cache system. When using higher parallelism, the
working set is larger, and forced to be evicted into the global memory, and
creates a bottleneck. Actually during the experiment, for the thread-level
parallelism implementation, reducing the number of blocks of the kernel improves
its running time, the opposite to normally what would be expected from running
other kernels.

### Gunrock implications

One thing that Gunrock does not provide, or intentionally hides, is block-level parallelism. However, it comes in handy when implementing the custom
kernels for GraphSAGE: each block can process a one-dimension vector, with each
thread holding one element, then use a block-level reduce to get the sum of those
elements; this way is highly efficient, and actually reduces the parallelism and with it the size of the working set of data.

### Notes on multi-GPU parallelization

The main memory usage is due to feature data, so it is critical to not
duplicate features. The side effect is the computation needs to be divided
into child-centric and source-centric parts, and exchange data in between. It
should be scalable, and easy to implement.

### Notes on dynamic graphs

GraphSAGE does not have a dynamic graph component, but it should able to work on a
dynamic graph. Some of the data may be reusable if the graph has not been
significantly changed, but the resulting memory requirement to store the intermediate
data may make data reuse impossible.

### Notes on larger datasets

If the dataset is so large that the graph and the per-vertex feature data are
larger than the combined GPU memory, it's possible to accumulate the features
of leafs and children on CPU, transfer the data on to GPU, and perform the
computation. Because of the cost of the transfer, runtime will increase significantly as compared to the cases when
the full data can fit in the GPU memory, but whether that will cause an increase above
the OpenMP implementation is still unknown.

### Notes on other pieces of this workload

The main part of GraphSAGE workflow is actually the training process, which will be
outside of Gunrock, provided by TensorFlow, PyTorch or other machine learning
libraries. How to connect the training with the Gunrock GPU implementation is
the main task for this workload going forward.

### Research potential

The GPU implementation of the embedding part runs a lot faster than on the CPU,
and hits a few GPU hardware limitations. It should have comparable runtime
to other GPU implementations.
But it's only useful as a part of the whole workload and achieves comparable
prediction accuracy as conventional / reference implementations.

An interesting question raised from the GraphSAGE GPU implementation is
whether it is useful to expose block-level parallelism to higher-level
programming models/APIs, and if so, how to do that. It's clear that working at the block level
provides benefits, such as the ability to use block-level primitives like scan
and reduce. But it also comes with costs, most importantly, requiring the programmer to have
knowledge about the GPU hardware. It also reduces the
portability of the implementation, because two-level parallelism may not
exist on other processors.
