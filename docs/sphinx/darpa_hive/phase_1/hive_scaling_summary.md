# Scaling analysis for HIVE applications: Executive Summary

At the request of Dr. Bryan Jacobs, we summarize our [November 2018
report on scalability](hive_scaling). Our Phase 2 deliverables are slated for completion in May 2021. These deliverables include same-node multi-GPU implementations of the eleven v0 applications. We understand that HIVE management would like an earlier assessment of predicted Gunrock performance on these applications. While we are only just getting underway with our implementation, and thus full results will not be available until May, we hope to help HIVE management with intermediate assessments and results so that DARPA can make the best decisions for the program. This summary is our initial effort toward this goal.

Our November 2018 report contained a lengthy and detailed assessment of Gunrock application scalability. This summary attempts to provide a higher-level summary for rapid digestion and assessment.

## Scaling to Multiple Processors

Let's begin with a non-computer analogy. Consider a workload of "digging a hole" and a baseline of one worker digging one hole. We wish to add more workers, allowing us to either dig one hole faster or dig more holes in the same amount of time.

In the HIVE context, consider a particular graph algorithm run on a particular dataset, evaluated on a system with one GPU. In an ideal world, if we added GPUs to the system so that we had $n$ GPUs, we could reduce the runtime by a factor of $n$ or, alternatively, allow us to process a dataset $n$ times larger with the same runtime. The former is called _strong scalability_ (more GPUs but the same dataset); the latter is _weak scalability_ (more GPUs and a dataset that grows with the number of GPUs).

To return to our hole analogy, strong scalability reflects how much faster $n$ workers can dig one hole; weak scalability reflects how much faster $n$ workers can dig $n$ holes. Strong scalability is generally harder to achieve than weak scalability. If $n$ workers are all digging one hole, each worker may not be fully busy, but it is more likely that if each worker digs his/her own hole, those workers will each be fully occupied. In the HIVE context, a dataset that is large enough to saturate a single GPU may not saturate any of those GPUs when divided among them.

For many reasons, perfect scalability is rare. For the purposes of this summary, we classify these reasons into two categories.

- **Cost of inter-GPU communication.** The single-GPU implementation has all of its data resident on that GPU. In contrast, a multi-GPU implementation distributes its memory across the GPUs. Most applications require transferring data between GPUs during the computation, and that communication path is slower than the local memory bandwidth on one GPU. This transfer takes time, and thus reduces performance and scalability.
- **Load imbalance across GPUs.** GPUs must coordinate their execution. While this coordination may potentially be cheap, one common way to coordinate is through the concept of a "barrier" in a program, where all GPUs must reach that barrier before any proceed. If those GPUs take a different amount of time to reach the barrier, all GPUs must wait at the barrier until the slowest one completes its work. This wait time reduces performance and thus scalability.

The dominant parallel programming model at the small scale is termed "bulk synchronous parallel" (BSP). In the BSP model, we divide the input across our GPUs and run the kernel in parallel on each GPU on its share of the input. Once that kernel is complete on all GPUs, we exchange data between GPUs and when that is complete, we start the next kernel.

BSP implementations have non-ideal scalability in practice for both the reasons above:

- BSP has a phase for communication between GPUs. No computation is done in this phase. This limits scalability.
- BSP has a barrier at the end of each phase. Any load imbalance within a phase across GPUs also limits scalability.

## Practical Limitations to Scalability

We can quantify these two limitations---cost of inter-GPU communication and load imbalance across GPUs---in the following ways.

### Cost of Inter-GPU Communication

The primary metric we will use to evaluate the scaling penalty due to inter-GPU communication is the ratio between on-GPU computation and inter-GPU communication. In the BSP model, this corresponds to the time spent in computation phases vs. the time spent communicating. In general, technology trends indicate that on-GPU computation increases in performance more quickly than chip-to-chip communication, so we expect the scalability of BSP implementations to generally decrease (slowly) over time.

We can quantify scalability with respect to inter-GPU communication with two (related) metrics:

1. The ratio between computation and communication. A high ratio indicates better scaling. Our report indicates that "Specific to the DGX-1 system with P100 GPUs, a ratio larger than about 10 to 1 is expected for an app to have at least marginal scalability."
2. The volume of data transferred on each communication phase. Implementations that must send only a small amount of data between GPUs should scale fairly well; implementations that send a lot will scale poorly. In our original report, we analyzed communication volume in great detail (results we will summarize below).

These two metrics are related since having more communication volume implies a lower computation-to-communication ratio.

### Inter-GPU Load Imbalance

The runtime of a BSP compute phase is gated by the slowest processor. Thus it is critical to give an equal workload to each processor. This is an extremely challenging task given that the workload itself may change on each iteration and. If we knew the workload ahead of time, we could precompute a data distribution across nodes that was roughly balanced. This process, however, is expensive (offline data partitioners may take hours) and is particularly problematic for scale-free graphs whose interconnectedness basically makes any partition bad.

Partitioners must balance between minimizing communication (impacting "cost of inter-GPU communication" above) and load balance. In our [2017 IPDPS paper on multi-GPU computing](http://dx.doi.org/10.1109/IPDPS.2017.117), we basically concluded that using offline partitioners offered little benefit. Offline partitioners were expensive and produced partitions with less communication, but in so doing had poorer load balance. We found that a (cheap) random partitioner was just as good in terms of overall results, at least for scale-free graphs. The benefit of near-perfect load balance (with a random partition) canceled out the additional communication requirements, and since the random partitioner was much easier and cheaper, we reported results with this.

Non-scale-free graphs (e.g., road networks) are both easier to partition and likely benefit more from a better partition. Nonetheless we feel, in general, that the problem of partitioning is orthogonal to the development of a system for graph analytics, and thus primarily distinguish our projected scalability characteristics by communication volume.

## Summary of our Analysis

In our analysis, we looked at the 9 applications that were based on Gunrock (seeded graph matching and application classification were implemented in Graph BLAS and not analyzed; however, analyses for matrix-based applications such as these two are much more well-established and straightforward).

At a very high level, perhaps the most important distinguishing feature between good and bad scalability is what is sent on every communication phase. Graphs have vertices and edges, and particularly in scale-free graphs, many more edges than vertices. If on every communication phase, we are sending data per _vertex_ (or proportional to the number of vertices), we generally project that we would scale fairly well. If we send data per _edge_, we generally would scale poorly. This is a direct consequence of the increased communication volume if we are sending data per edge as opposed to data per vertex.

While we analyze the scalabilities in more detail in the full report, we summarize them here:

**Good scalability** GraphSAGE, geolocation using UVM or peer accesses, and local graph clustering belong to this group. They share some algorithmic signatures: the whole graph needs to be visited at least once in every iteration, and visiting each edge involves nontrivial computation. The communication costs are roughly at the level of $V$. As a result, the computation vs. communication ratio is larger than $E : V$. PageRank is a standard graph algorithm that falls in this group.

**Moderate scalability** This group includes Louvain, geolocation using explicit movement, vertex nomination, scan statistics, and graph projection. They either only visit part of the graph in an iteration, have only trivial computation during an edge visit, or communicate a little more data than $V$. The computation vs. communication is less than $E : V$, but still larger than 1 (or 1 operation : 4 bytes). They are still scalable on the DGX-1 system, but not as well as the previous group. Single source shortest path (SSSP) is an typical example for this group.

**Poor scalability** Random walk, graph search, and sparse fused lasso belong to this group. They need to send out some data for each vertex or edge visit. As a result, the computation vs communication ratio is less than 1 (or 1 operation : 4 bytes). They are very hard to scale across multiple GPUs. Random walk is an typical example.

If we ignore the non-peer access aspects of the DGX's inter-GPU communication network, we might hope that a "good scalability" application might scale to 8 GPUs with a scalability factor of, say, 5--7 times faster performance than 1 GPU; "moderate scalability" might mean 2--5; and "poor scalability" is likely to be 1--2. If desired, we could plug our performance results into this estimate to give a ballpark figure for DARPA's evaluation.
