# Geolocation

Infers user locations using the location (latitude, longitude) of friends through spatial label propagation. Given a graph `G`, geolocation examines each vertex `v`'s neighbors and computes the spatial median of the neighbors' location list. The output is a list of predicted locations for all vertices with unknown locations.

## Summary of Results

Geolocation or geotagging is an interesting parallel problem, because it is among the few that exhibits the dynamic parallelism pattern within the compute. The pattern is as follows; there is parallel compute across nodes, each node has some serial work and within the serial work there are several parallel math operations. Even without leveraging dynamic parallelism within CUDA (kernel launches within a kernel), Geolocation performs well on the GPU environment because it mainly requires simple math operations, instead of complicated memory movement schemes.

However, the challenge within the application is load balancing this simple compute, such that each processor has roughly the same amount of work. Currently, in Gunrock, we map Geolocation using the `ForAll()` compute operator with optimizations to exit early (performing less work and fewer reads). Even without addressing load balancing issue with a complicated balancing scheme, on the HIVE datasets we achieve a 100x speedup with respect to the CPU reference code, implemented using C++ and OpenMP, and ~533x speedup  with respect to the GTUSC implementation. We improve upon the algorithm by avoiding a global gather and a global synchronize, and using 3x less memory than the GTUSC reference implementation.

## Summary of Gunrock Implementation

There are two approaches we took to implement Geolocation within Gunrock:

- **[Fewer Reads] Global Gather:** uses two `compute` operators as `ForAll()`. The first `ForAll()` is a `gather` operation, gathering all the values of neighbors with known locations for an active vertex `v`, and the second `ForAll()` uses those values to compute the `spatial_center` where the spatial center of a list's points is the center of those points on the earth's surface.

```python
def gather_op(Vertex v):
    for neighbor in G.neighbors(v):
        if isValid(neighbor.location):
            locations_list[v].append(neighbor.location)

def compute_op(Vertex v):
    if !isValid(v.location):
        v.location = spatial_center(locations_list[v])
```

- **[Less Memory] Repeated Compute:** skips the global gather and uses only one `compute` operator as a `ForAll()` to find the spatial center of every vertex. During the spatial center computation, instead of iterating over all valid neighbors (where valid neighbor is a neighbor with a known location), we iterate over all neighbors for each vertex, doing more random reads than the global gather approach, but using `3x` less memory.

```python
def spatial_center(Vertex v):
    if !isValid(v.location):
        v.location = spatial_median(neighbors_list[v])
```

- **[Optimization] Early Exit:** fuses the global gather approach with the repeated compute, by performing one local gather for every vertex within the spatial center operator (without a costly device barrier), and exiting early if a vertex `v` has only one or two valid neighbors:

```python
def spatial_center(Vertex v):
    if !isValid(v.location):
        if v.valid_locations == 1:
            v.location = valid_neighbor[v].location:
            exit
        else if v.valid_locations == 2:
            v.location = mid_point(valid_neighbors[v].location)
        else:
            v.location = spatial_median(neighbors_list[v])
```

### Comparing Global Gather vs. Repeated Compute

| Approach         | Memory Usage | Memory Reads/Vertex  | Device Barriers | Largest Dataset (P100) |
|------------------|--------------|----------------------|-----------------|------------------------|
| Global Gather    | $O(3 \cdot \cardinality{E})$     | # of valid locations | 1               | ~160M Edges            |
| Repeated Compute | $O(\cardinality{E})$       | degree of vertex     | 0               | ~500M Edges            |


**Note:** `spatial_median()` is defined as center of points on earth's surface -- given a set of points `Q`, the function computes the point `p` such that: `sum([haversine_distance(p, q) for q in Q])` is minimized. See `gunrock/app/geo/geo_spatial.cuh` for details on the spatial median implementation.

## How To Run This Application on DARPA's DGX-1

### Prerequisites

```shell
git clone --recursive https://github.com/gunrock/gunrock -b dev-refactor
cd gunrock/tests/geo/
make clean && make
```

### HIVE Data Preparation

Prepare the data, skip this step if you are just running the sample dataset. Assuming we are in `tests/geo` directory:

```shell
export TOKEN= # get this Authentication TOKEN from
              # https://api-token.hiveprogram.com/#!/user
wget --header "Authorization:$TOKEN" \
  https://hiveprogram.com/data/_v0/geotagging/instagram/instagram.tar.gz
tar -xzvf instagram.tar.gz && rm instagram.tar.gz
cd instagram/graph
cp ../../generate-data.py ./
python generate-data.py
```

This will generate two files, `instagram.mtx` and `instagram.labels`, which can be used as an input to the geolocation app.

### Running the application

Application specific parameters:

```
--labels-file
    file name containing node ids and their locations.

--geo-iter
    number of iterations to run geolocation or (stop condition).
    (default = 3)

--spatial-iter
    number of iterations for spatial median computation.
    (default = 1000)

--geo-complete
    runs geolocation for as many iterations as required
    to find locations for all nodes.
    (default = false because it uses atomics)

--debug
    Debug label values, this prints out the entire labels
    array (longitude, latitude).
    (default = false)
```

Example command-line:

```shell
# geolocation.mtx is a graph based on chesapeake.mtx dataset
./bin/test_geo_10.0_x86_64 --graph-type=market --graph-file=./geolocation.mtx \
  --labels-file=./locations.labels --geo-iter=2 --geo-complete=false
```

Sample input (labels):

```
% Nodes Latitude Longitude
39 2 2
1 37.7449063493 -122.009432884
2 37.8668048274 -122.257973253
4 37.869112506 -122.25910604
6 37.6431858915 -121.816156983
11 37.8652346572 -122.250634008
19 38.2043433677 -114.300341275
21 36.7582225593 -118.167916598
22 33.9774659389 -114.886512278
30 39.2598884729 -106.804662071
31 37.880443573 -122.230147039
39 9.4276164485 -110.640705659
```

Sample output:

```
Loading Matrix-market coordinate-formatted graph ...
Reading from ./geolocation.mtx:
  Parsing MARKET COO format edge-value-seed = 1539674096
 (39 nodes, 340 directed edges)...
Done parsing (0 s).
  Converting 39 vertices, 340 directed edges ( ordered tuples) to CSR format...
Done converting (0s).
Labels File Input: ./locations.labels
Loading Labels into an array ...
Reading from ./locations.labels:
  Parsing LABELS
 (39 nodes)
Done parsing (0 s).
Debugging Labels -------------
 (nans represent unknown locations)
    locations[ 0 ] = < 37.744907 , -122.009430 >
    locations[ 1 ] = < 37.866806 , -122.257973 >
    locations[ 2 ] = < nan , nan >
    locations[ 3 ] = < 37.869114 , -122.259109 >
     ...
    locations[ 35 ] = < nan , nan >
    locations[ 36 ] = < nan , nan >
    locations[ 37 ] = < nan , nan >
    locations[ 38 ] = < 9.427616 , -110.640709 >
__________________________
______ CPU Reference _____
--------------------------
 Elapsed: 0.267029
Initializing problem ...
Number of nodes for allocation: 39
Initializing enactor ...
Using advance mode LB
Using filter mode CULL
nodes=39
__________________________
0        0       0       queue3          oversize :      234 ->  342
0        0       0       queue3          oversize :      234 ->  342
--------------------------
Run 0 elapsed: 11.322021, #iterations = 2
Node [ 0 ]: Predicted = < 37.744907 , -122.009430 > Reference = < 37.744907 , -122.009430 >
Node [ 1 ]: Predicted = < 37.866806 , -122.257973 > Reference = < 37.866806 , -122.257973 >
Node [ 2 ]: Predicted = < 9.427616 , -110.640709 > Reference = < 9.427616 , -110.640709 >
Node [ 3 ]: Predicted = < 37.869114 , -122.259109 > Reference = < 37.869114 , -122.259109 >
...
Node [ 35 ]: Predicted = < 37.864429 , -122.199409 > Reference = < 37.864429 , -122.199409 >
Node [ 36 ]: Predicted = < 23.755602 , -115.803055 > Reference = < 37.807079 , -122.134163 >
Node [ 37 ]: Predicted = < 37.053715 , -115.913658 > Reference = < 37.053719 , -115.913628 >
Node [ 38 ]: Predicted = < 9.427616 , -110.640709 > Reference = < 9.427616 , -110.640709 >
0 errors occurred.
[geolocation] finished.
 avg. elapsed: 11.322021 ms
 iterations: 2
 min. elapsed: 11.322021 ms
 max. elapsed: 11.322021 ms
 load time: 68.671 ms
 preprocess time: 496.136000 ms
 postprocess time: 0.463009 ms
 total time: 508.110046 ms
```

### Output

When `quick` mode is disabled, the application performs the CPU reference implementation, which is used to validate the results from the GPU implementation by comparing the predicted latitudes and longitudes of each vertex with the CPU reference implementation. Further correctness checking was performed by comparing results to the [HIVE reference implementation](https://gitlab.hiveprogram.com/ggillary/geotagging.git).

Geolocation application also supports the `quiet` mode, which allows the user to skip the output and just report the performance metrics (note, this will run the CPU implementation in the background without any output).

## Performance and Analysis

Runtime is the key metric for measuring performance for Geolocation. We also check for prediction accuracy of the labels, but that is a threshold for correctness. If a certain threshold is not met (while comparing results to the CPU reference code), the output is considered incorrect and that run is invalid. Therefore, for the report we just focus on runtime.

### Implementation limitations

Geolocation is also one of the few applications that exhibits a dynamic parallelism pattern:

- Parallel compute across the nodes,
- Serial compute per node, and
- Parallel compute within the serial compute per node.

One way to implement this will use the `ForAll()` operator for the parallel compute across the nodes, a simple while loop for the serial compute per node, and finally multiple `neighbor_reduce()` operators for the parallel work within the serial while loop. Currently, we do not have a way to support this within Gunrock, but moving forward we can potentially leverage kernel launch within a kernel ("dynamic parallelism") to address this limitation.

### Comparison against existing implementations

| GPU  | Dataset            | $\cardinality{V}$    | $\cardinality{E}$     | Iters | Spatial Iters | GTUSC (16 threads) | Gunrock (CPU)  | Gunrock (GPU) |
|------|-----------------|-------------|--------------|----------|------------|----------------|-------------------|---------------|
| P100 | sample             | 39       | 170       | 10         | 1000          | N/A                | 0.144005       | 0.022888      |
| P100 | instagram          | 23731995 | 82711740  | 10         | 1000          | 8009.491 ms        | 1589.884033    | 15.113831     |
| V100 | twitter            | 50190344 | 488078602 | 10         | 1000          | N/A                | 9216.666016    | 46.108007     |

On a workload that fills the GPU, Gunrock outperforms GT's OpenMP C++ implementation by ~533x. Comparing Gunrock's GPU vs. CPU performance, we see that Gunrock's GPU version outperforms the CPU implementation by 100x. There is a lack of available datasets against which we can compare performance, so we use only the provided instagram and twitter datasets, and a toy sample for a sanity check on NVIDIA's P100 with 16GB of global memory and V100 with 32GB of global memory. All tested implementations meet the criteria of accuracy, which is validated against the output of the original python implementation.

- [HIVE reference implementation](https://gitlab.hiveprogram.com/ggillary/geotagging.git) uses distributed PySpark.
- [GTUSC implementation](https://gitlab.hiveprogram.com/gtusc/geotagging) uses C++ OpenMP.

### Performance limitations

As discussed later in the "Alternate approaches" section, the current implementation of geolocation uses a compute operator with minimal load balancing. In cases where the graph is not so nicely distributed (where there is a great deal of difference in the degrees of vertices), the entire application will suffer significantly from load imbalance.

Profiling the application shows 98.78% of the compute time in GPU activities is in the `spatial_median` kernel, which gives us a good direction to focus our efforts on load-balancing the workloads within the operator. Specifically, we must target the `for` loops iterating over the neighbor list for spatial center calculations.

## Next Steps

### Alternate approaches

- **Neighborhood Reduce w/ Spatial Center:** We can perform better load balancing by leveraging a neighbor-reduce (`advance` operator + `cub::DeviceSegmentedReduce`) instead of using a compute operator. In graphs where the degrees of nodes vary a lot, the compute operator will be significantly slower than a load-balanced advance + segmented reduce.

- **Push Based Approach:** Instead of gathering all the locations from all the neighbors of an active vertex, we could instead perform a scatter of valid locations of all active vertices to their neighbors; this is a push approach vs. our current implementation's pull. Similar to the global gather approach, a push-based geolocation could also suffer from load imbalance, where some vertices will have to broadcast their valid locations to a long list of neighbors, while others will only have few neighbors to update. A push-based approach will also require a device synchronize before the spatial center computation, but may perform better by using an `advance_op` with an atomic update (note, pull is done using a `ForAll()`).

### Gunrock implications

- **The `predicted` atomic:** Geolocation and some other applications exhibit the same behavior where the algorithm stops when all vertices' labels are predicted or determined. In Geolocation's case, when a location for all nodes is predicted, geolocation converges. We currently implement this with a loop and an atomic. This needs to be more of a core operation (mini-operator) such that when `isValidCount(labels|V|) == |V|`, a stop condition is met. Currently, we sidestep this issue by using a number-of-iterations parameter to determine the stop condition.

- **Parallel -> Serial -> Parallel:** As discussed earlier, Gunrock currently doesn't have a way to address the dynamic parallelism problem, or even a kernel launch within a kernel. In geolocation's case, these minor parallel work inside the serial loop need to be multiple neighbor reduce.

### Notes on multi-GPU parallelization

The challenging part for a multi-GPU Geolocation would be to obtain the updated node location from a separate device if the two vertices on different devices share an edge. An interesting approach here would be leveraging the P2P memory bandwidth with the new NVLink connectors to exchange a small amount of updates across the NVLink's memory lane; other ways are simply using direct accesses or explicit data movement. This is detailed more in the scaling documentation, but the communication model for multi-GPU geolocation could be done in the following way:

```
do
    Local geo location updates on local vertices;
    Broadcast local vertices' updates;
while no more update.
```

### Notes on dynamic graphs

Streaming graphs is an interesting problem for the Geolocation application, because when predicting the location of a certain node, if another edge is introduced, the location of the vertex has to be recomputed entirely. This can still be done in an iterative manner, where if a node was inserted as a neighbor to a vertex, that vertex's predicted location will be marked invalid and during the next iteration it will be computed again along with all the other invalid vertices (locations).

### Notes on larger datasets

If the datasets are larger than a single or multi-GPU's aggregate memory, the straightforward solution would be to let Unified Virtual Memory (UVM) in CUDA automatically handle memory movement.

### Notes on other pieces of this workload

Geolocation calls a lot of CUDA math functions (`sin`, `cos`, `atan`, `atan2`, `median`, `mean`, `fminf`, `fmaxf`, etc.).  Some of these micro-workloads can also leverage the GPU's parallelism; for example, a mean could be implemented using `reduce-mean/sum`. We currently don't have these math operators exposed within Gunrock in such a way they can be used in graph applications.

### Research Potential

Further research is required to study Geolocation's dynamic parallelism pattern, it's memory access behavior, compute resource utilization, implementation details (API and core) and load balancing strategies for dynamic parallelism on the GPUs. Studying and understanding this pattern can allow us to create a more generalized approach for load balancing `parallel -> serial -> parallel` type of problems. It further invokes the question of studying when dynamic parallelism is better than mapping an algorithm to a more conventional static approach (if possible).
