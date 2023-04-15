# Sparse Fused Lasso

The Phase 1 report for SFL is found [here](hive_sparse_graph_trend_filtering).

> Given a graph where each vertex on the graph has a weight, _sparse fused lasso (SFL)_, also named _sparse graph trend filter (GTF)_, tries to learn a new weight for each vertex that is (1) sparse (most vertices have weight 0), (2) close to the original weight in the l2 norm, and (3) close to its neighbors' weight(s) in the l1 norm. This algorithm is usually used in main trend filtering (denoising). For example, an image (grid graph) with noisy pixels can be filtered with this algorithm to get a new image without the noisy pixels, which are "smoothed out" by its neighbors.
<https://arxiv.org/abs/1410.7690>

## Scalability Summary

Maxflow kernel is serial

## Summary of Results

Sparse Fused Lasso (or Sparse Graph Trend Filtering) relies on a Maxflow algorithm. As highlighted in the Phase 1 report, a sequential implementation of Maxflow outperforms a single-GPU implementation, and the actual significant core operation of SFL is a serial normalization step that cannot be parallelized to a single GPU, let alone multiple GPUs. Therefore, we refer readers to the phase 1 report for this workload. Parallelizing across multiple GPUs is not beneficial.
