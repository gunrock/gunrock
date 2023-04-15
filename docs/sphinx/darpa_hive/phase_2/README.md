# HIVE Phase 2 Report&colon; Executive Summary

> This is a port from v1 docs, some links might be broken. Please see: https://gunrock.github.io/docs/#/hive/

This report ([version history](#version-summary) [[HTML](hive_version_summary.md)])  is also located online at the following URL: <hive_phase2_summary>. Links currently work better in the PDF version than the HTML version.

Herein UC Davis produces the following deliverables that it promised to deliver in Phase 2:

- Implementation of DARPA HIVE v0 apps as single-node, multi-GPU applications using the [Gunrock](https://github.com/gunrock/gunrock) framework
- Performance characterization of these applications across multiple GPUs
- Analysis of the limits of scalability for these applications

In our writeup, we first [describe how to reproduce our results](#running-the-applications) ([HTML](hive_run_apps_phase2)) and then [describe the scalability behavior of our ForAll operator](#gunrocks-forall-operator) ([HTML](hive_forall_phase2)) .

We begin with a table that summarizes the scalability behavior for each application, then a longer description of each application:

| Application | Scalability behavior |
| ----------- | -------------------- |
| Scan Statistics | Bottlenecked by single-GPU and communication |
| GraphSAGE | Bottlenecked by network bandwidth between GPUs |
| Application Classification | Bottlenecked by network bandwidth between GPUs |
| Geolocation | Bottlenecked by network bandwidth between GPUs |
| Community Detection (Louvain) | Application is nonfunctional |
| Local Graph Clustering (LGC) | Bottlenecked by single-GPU and communication |
| Graph Projections | Limited by load imbalance |
| GraphSearch | Bottlenecked by network bandwidth between GPUs |
| Seeded Graph Matching (SGM) | We observe great scaling |
| Sparse Fused Lasso | Maxflow kernel is serial |
| Vertex Nomination | We observe weak scaling |

## [App: Scan Statistics](#scan-statistics) ([HTML](hive_SS_phase2))

We rely on Gunrock's multi-GPU `ForALL` operator to implement Scan Statistics. We see no scaling and in general performance degrades as we sweep from one to sixteen GPUs. The application is likely bottlenecked by the single GPU intersection operator that requires a two-hop neighborhood lookup and accessing an array distributed across multiple GPUs.

## [App: GraphSAGE](#graphsage) ([HTML](hive_Sage_phase2))

We rely on Gunrock's multi-GPU `ForALL` operator to implement GraphSAGE. We see no scaling as we sweep from one to sixteen GPUs due to communication over GPU interconnects.

## [App: Application Classification](#application-classification) ([HTML](hive_ac_phase2))

We re-forumlate the `application_classification` workload to improve memory locality and admit a natural multi-GPU implementation.  We then parallelized the core computational region of `application_classification` across GPUs.  For the kernels in that region that do not require communication between GPUs, we attain near-perfect scaling.  Runtime of the entire application remains bottlenecked by network bandwidth between GPUs.  However, mitigating this bottleneck should be possible further optimization of the memory layout.

## [App: Geolocation](#geolocation) ([HTML](hive_geolocation_phase2))

We rely on Gunrock's multi-GPU `ForALL` operator to implement Geolocation as the entire behavior can be described within a single-loop like structure. The core computation focuses on calculating a spatial median, and for multi-GPU `ForAll`, that work is split such that each GPU gets an equal number of vertices to process. We see a minor speed-up on a DGX-A100 going from 1 to 3 GPUs on a twitter dataset, but in general, due to the communication over the GPU-GPU interconnects for all the neighbors of each vertex, there's a general pattern of slowdown going from 1 GPU to multiple GPUs, and no scaling is observed.

## [App: Community Detection (Louvain)](#community-detection-louvain) ([HTML](hive_louvain_phase2))

The application has a segmentation fault and is currently nonfunctional.

## [App: Local Graph Clustering (LGC)](#local-graph-clustering-lgc) ([HTML](hive_pr_nibble_phase2))

We rely on Gunrock's multi-GPU `ForALL` operator to implement Local Graph Clustering and observe no scaling as we increase from one to sixteen GPUs. The application is likely bottlenecked by single-GPU filter and advance operators and communication across NVLink necessary to access arrays distributed across GPUs.

## [App: Graph Projections](#graph-projections) ([HTML](hive_proj_phase2))

We implemented a multi-GPU version of sparse-sparse matrix multiplication, based on chunking the rows of the left hand matrix.  This yields a communication-free implementation with good scaling properties.  However, our current implementation remains partially limited by load imbalance across GPUs.

## [App: GraphSearch](#graphsearch) ([HTML](hive_rw_phase2))

We rely on a Gunrock's multi-GPU `ForALL` operator to implement GraphSearch as the entire behavior can be described within a single-loop like structure. The core computation focuses on determining which neighbor to visit next based on uniform, greedy, or stochastic functions. Each GPU is given an equal number of vertices to process. No scaling is observed, and in general we see a pattern of decreased performance as we move from 1 to 16 GPUs due to random neighbor access across GPU interconnects.



## [App: Seeded Graph Matching (SGM)](#seeded-graph-matching-sgm) ([HTML](hive_sgm_phase2))

Multi-GPU SGM experiences considerable speed-ups over single GPU implementation with a near linear scaling if the dataset being processed is large enough to fill up the GPU. We notice that ~$1$ million nonzeros sparse-matrix is a decent enough size for us to show decent scaling as we increase the number of GPUs. The misalignment for this implementation is also synthetically generated (just like it was for Phase 1, the bottleneck is still the `|V|x|V|` allocation size).

## [App: Sparse Fused Lasso](#sparse-fused-lasso) ([HTML](hive_sparse_graph_trend_filtering_phase2))

Sparse Fused Lasso (or Sparse Graph Trend Filtering) relies on a Maxflow algorithm. As highlighted in the Phase 1 report, a sequential implementation of Maxflow outperforms a single-GPU implementation, and the actual significant core operation of SFL is a serial normalization step that cannot be parallelized to a single GPU, let alone multiple GPUs. Therefore, we refer readers to the phase 1 report for this workload. Parallelizing across multiple GPUs is not beneficial.


## [App: Vertex Nomination](#vertex-nomination) ([HTML](hive_vn_phase2))

We implemented `vertex_nomination` as a standalone CUDA program, and achieve good weak scaling performance by eliminating communication during the `advance` phase of the algorithm and using a frontier representation that allows an easy-to-compute reduction across devices.

---

We also produce web versions of our [scalability plots](plots/) and [scalability tables of results](tables/).
