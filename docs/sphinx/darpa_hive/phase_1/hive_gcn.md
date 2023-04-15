# Graph Convolutional Network

The most promising use case is in Semi-supervised Learning where we are given a set of nodes, each with some observed numeric attributes x<sub>i</sub>.

Now, we `predict an output/label for each node` based on partial observations i.e. labels for some, but not all, of the nodes.

We might also be given a set of weighted edges, summarised by an adjacency matrix A. The main assumption is that when predicting the output yi for node i, the attributes and connectivity of nearby nodes provide useful side information or additional context.


## Summary of Results

-- Talk to JDO about this. Write it last, probably.

## Summary of Gunrock Implementation

The implementation has been hugely guided by

- http://proceedings.mlr.press/v97/wu19e.html
- https://arxiv.org/abs/1609.02907


### The GCN algorithm can be mapped into the following steps:

1. Initialization
-   Data Reading/Parsing
-   Parameter Initialization
-   [Random initialization of weight matrices](https://github.com/achalagarwal/gunrock/blob/d0202e3bbb88560bc97666675c0a94aa9e491c9c/gunrock/app/GuNNrock/gcn_problem.cuh#L225) W<sub>0</sub> and W<sub>1</sub>

[comment]: <> (Forward propagation and Backpropagation are explained for a single epoch, the same process is iterated for --num_iterations)

<sup><sub>__Forward Propagation__</sub></sup>

2. Edge Dropout
  - [With probability `p`, mask (disable) an edge value](https://github.com/achalagarwal/gunrock/blob/d0202e3bbb88560bc97666675c0a94aa9e491c9c/gunrock/app/GuNNrock/dropout/dropout.cuh#L53)
  - Results in new edge values
3. Edge Weight Sparse Multiplication (Neighborhood Gather)
  - [Multiplication of edge values with trainable weights](https://github.com/achalagarwal/gunrock/blob/d0202e3bbb88560bc97666675c0a94aa9e491c9c/gunrock/app/GuNNrock/sparseMatMul/sparseMatMul_enactor.cuh#L89)
  - Results in the XW<sub>0</sub> matrix
4. Graph Neighbor Sum (Aggregation)
  - [Summing the neighbour vectors for each vertex](https://github.com/achalagarwal/gunrock/blob/d0202e3bbb88560bc97666675c0a94aa9e491c9c/gunrock/app/GuNNrock/graphsum/graphsum_enactor.cuh#L99)
  - Results in the AXW<sub>0</sub> matrix
5. ReLU
  - on the AXW<sub>0</sub> matrix
6. Dropout
  - on the rectified AXW<sub>0</sub> matrix
7. Multiplication of W<sub>1</sub> weight matrix
  - results in AXW<sub>0</sub>W<sub>1</sub>
8. Repeat Graph Neighbor Sum 
  - [Summing the neighbour vectors for each vertex](https://github.com/achalagarwal/gunrock/blob/d0202e3bbb88560bc97666675c0a94aa9e491c9c/gunrock/app/GuNNrock/graphsum/graphsum_enactor.cuh#L99)
  - Results in the AAXW<sub>0</sub>W<sub>1</sub> matrix
9. Cross Entropy Loss
  - Compute training loss and likewise gradients of AAXW<sub>0</sub>W<sub>1</sub> matrix
  - Start Backprop
  
<sup><sub>__Backward Propagation__</sub></sup>

10. backprop for 8.
  - Results in the gradients of AXW<sub>0</sub>W<sub>1</sub> matrix
11. backprop for 7. 
  - Compute the gradients of W<sub>1</sub> matrix and stores it to update the W<sub>1</sub> weight matrix later
  - Results in the gradients of AXW<sub>0</sub> matrix
12. backprop for 6. 
  - Results in the updated gradients of AXW<sub>0</sub> matrix
13. backprop for 5. 
  - Results in the updated gradients of AXW<sub>0</sub> matrix
14. backprop for 4. 
  - Results in the updated gradients of XW<sub>0</sub> matrix
15. backprop for 3.
  - Compute the gradients of W<sub>0</sub> matrix and stores it to update the W<sub>0</sub> weight matrix later
[16.]: <> (backprop 2 does not exist as that computation does not involve any trainable weight)
16. Update weight matrices W<sub>0</sub> and W<sub>1</sub>
  - Use the new weight matrices in the next epoch (iteration)

<sup><sub>__End of training__</sub></sup>

17. Export the trained weight matrices along with the loss/accuracy/runtime metrics

### The description of the lower level operators used to implement some of the steps described above:

The 17 steps above share computation patterns

1. To update an Array1D, the `ForEach` op has been used

```CUDA
 GUARD_CU (arr.ForEach (
          [params]__host__ __device__(ValueT &x) {
            x = update(x, params);
          }
      ))
```

2. Neighborhood Gather / Scatter
Gather neigbhorhood features (X) for all vertices after multiplication with edge weight (weights)

```CUDA
    auto denseMM =
        [X, output, C, W] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
      for (int i = 0; i < C; i++) {
        atomicAdd(output + src * C + i, W[edge_id] * X[dest * C + i]);
      }
      return true;
    };
   
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V> (
            graph.csr (), &local_vertices, null_ptr, oprtr_parameters,
            denseMM));
```

3. To aggregate feature matrix M (shape: YxZ) along the adjacency list

```CUDA
    auto sumNeighbors =
        [M, output, Z] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
     
      for (int i = 0; i < Z; i++)
        atomicAdd(output + src * Z + i, *(M + dest * Z + i));
      return true;
    };
    
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V> (
            graph.csr (), &local_vertices, null_ptr, oprtr_parameters,
            sumNeighbors));
    
```


### What was implemented with respect to the entire workflow?

The various modules viz.:

- Activation Function (ReLU)
- Dropout
- Graph Sum
- Loss Function (Cross Entropy)
- SpMM Multiplication
- Manual differentiation for all the operators
- ...

## How To Run This Application on DARPA's DGX-1


### Prereqs/input

1. Build Gunrock -- https://github.com/gunrock/gunrock/pull/805
2. Make sure the following datafiles are available:
  - feature file (edge weights / node vectors)
  - split file (for each vertex, a value 0/1/2 to specify train/test/validation node
  - graph file (adjacency list format)
  
### Running the application

<!-- <code> -->
```bash
# build gunrock 

# cd to bin folder
cd ./build/bin

# run gcn binary
./gcn --feature_file <featurefile> --graph_file <graphfile> --split_file <splitfile>

# vary parameters like: Number of training iterations, silent run, etc. 
```
<!-- </code> -->

Note: This run / these runs are faster on DARPA's DGX-1.

### Output

1. Relevant data is printed per epoch:

- Training loss/acc
- Validation loss/acc
- Time taken

2. Output after training:

- test loss/acc
- time taken by various operators

#### To extract weights:

Uncomment [call to `Extract()` function](https://github.com/achalagarwal/gunrock/blob/d0202e3bbb88560bc97666675c0a94aa9e491c9c/gunrock/app/GuNNrock/gcn_app.cu#L119) which provides both W<sub>0</sub> and W<sub>1</sub> trained matrices

#### How do you make sure your output is correct/meaningful? (What are you comparing against?)

- Some of the operators have unittests
- Backpropagation has been verified using Autodiff in Python 
- Manual verification of the overall algorithm with the theoretical reference

## Performance and Analysis

@JDO
Will be adding results for more datasets, are there any specific metrics that you think I should focus on? I could share the highlevel metrics from nvsight?

```bash

# citeseer dataset (default)
time ./gcn --max_iter=1000

real	0m6.419s
user	0m5.263s
sys	0m2.112s
```

Average training time: **5.73ms** </br>
Lowest training time: **3.3ms** </br>
Highest training time: **11.8ms** </br>

**Modules (% time taken)**
1. Graph Sum (Total: 45%, Forward: 30%, Backprop: 15%)
2. Sparse Matrix Multiplication (Total: 23.4%, Forward: 15.2%, Backprop: 8.2%)
3. Cross Entropy Loss (15.6%)
4. Matrix Multiplication (Total: 22.5%, Forward: 10.2%, Backprop: 12.3%)


### Implementation limitations

- No provision for using local weight matrices (to checkpoint training on disk)
- No provision for Learning Rate modulation
- No provsion for hyperparamter grid search

e.g.:

- Size of dataset that fits into GPU memory (what is the specific limitation?)
- Restrictions on the type/nature of the dataset

### Comparison against existing implementations

- Reference implementation (python: https://github.com/Tiiiger/SGC + https://github.com/zhouchunpong/Simplifying-Graph-Convolutional-Networks)

The accuracy shouldn't be affected, the performance benchmark will be added (TODO)


- 

### Performance limitations

e.g., random memory access?

## Next Steps

### Alternate/Next approaches

1. Auto backpropagation

Integrate autodiff in Gunrock so that a developer does not need to generate backpropagation code manually. https://github.com/mitsuba-renderer/enoki, https://github.com/mitsuba-renderer/enoki

2. Use optimised Sparse Matrix multiplication and aggregations

Currently we use pure atomics for all such operations and shifting to better/optimised algorithms will make our training faster

3. Providing a queue of graphs to read from disk (depends on application)

To better leverage the speed that gunrock provides for training GNNs, we should batch our graphs from CPU to GPU so that the I/O time is minimized

4. Move to better GNN architectures

GNN research has led to various different GNN architectures that perform better on certain datasets/tasks. Providing support for a canonical set of operators to support multiple GNN architectures.


### Gunrock implications

> What did we learn about Gunrock? What is hard to use, or slow? What potential Gunrock features would have been helpful in implementing this workflow?

1. Gunrock as a framework for GNN would not be for the masses, it is better suited as a library to a Python Interface so that users can quickly iterate over their code etc.

2. Gunrock has optimised traversal and frontier operators that make certain operations faster but providing support for optimised implementations of matrix multiplication / sparse matrix multiplication / support for 2D arrays / quick integration between apps amongst themselves etc. will make it faster for users to develop their architectures


### Notes on multi-GPU parallelization

> What will be the challenges in parallelizing this to multiple GPUs on the same node?

1. Effectively dividing across multiple GPUs

- Model Parallelization

Easy as we can have a glue module that receives data from all the split components and completes the pipeline

- Large Graph (Monolithic model)

There is some work done on graph partitioning for GNN training specifically, that can be leveraged.
Secondly, all the provided operators need to support multi gpu mode and that will be plenty work.

### Notes on dynamic graphs

(Only if appropriate)

> Does this workload have a dynamic-graph component? </br>
Not currently but it could benefit from support for dynamic graphs (Pooling operators)

> If so, what are the implications of that? </br>
Pooling operators have been shown to help increase the quality as well as the performance of training

> How would your implementation change? What support would Gunrock need to add? </br>
Gunrock needs to provide support for Union-Find on graphs, edge contraction etc. 



### Notes on larger datasets

What if the dataset was larger than can fit into GPU memory or the aggregate GPU memory of multiple GPUs on a node? What implications would that have on performance? What support would Gunrock need to add?

### Notes on other pieces of this workload

Briefly: What are the important other (non-graph) pieces of this workload? Any thoughts on how we might implement them / what existing approaches/libraries might implement them?
