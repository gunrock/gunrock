# Gunrock v0.5 Release Notes {#release_notes}
Release 0.5, Upcoming

Release 0.5 is mainly a API refactor, with some feature updates:
- New operator interfaces
- New graph representations
- New frontier structure
- New test driver
- Restructured enactor routines
- New parameter handling
- New 1D operators
- Other code restructuring
- Optional Boost dependency

## ChangeLog
- Operators (i.e. advance, filter)
    - Take in lambda functions for per-element operations, instead of static
      functions in a structure. `<algo>_functor.cuh` is merged into
      `<algo>_enactor.cuh`
    - Use `OprtrParameters` structure to keep inputs, except for the graph,
      input / output frontiers, and the lambdas
    - `KernelPolicy` is defined within each operator, instead of in the enactor
    - Templatized options (Idempotence, mark-preds, advance types, reduce ops,
      reduce types, etc.) are provided as a combined 32bit `OprtrFlag`
    - Queue index and selector are automatically changed by the operator when
      needed


- Graph representation
    - A single structure encloses all graph related data
    - Different representations (CSR, CSC, COO, etc.) can be selected based
      on algorithmic needs
    - New graph representations could be added without changing other parts of
      Gunrock, except operator implementation that handles how to traverse such
      new representation
    - CPU, GPU and sub-graphs use the same graph data structure, no more
      `GraphSlice` and `GRGraph`


- Frontier
    - A single structure `gunrock/app/frontier.cuh:Frontier`
      encloses all frontier related data


- Test driver
    - Allows multiple graph types (`64bit-VertexT`, `64bit-SizeT`,
      `64bit-ValueT`, directed vs. undirected) and multiple parameters
      combinations to run in a single execution
    - Allows result validation for each run, instead of only the last run
    - Result validation without reference for BFS and SSSP
    - Moved common functions into `gunrock/app/test_base.cuh`    
    - Moved CPU reference code and result validation into
      `gunrock/app/<algo>/<algo>_test.cuh`


- Enactor
    - Common functions moved into `gunrock/app/enactor_base.cuh`
    - Use OpenMP to maintain controlling threads on CPU
    - Use instances of `Iteration` instead of static access to its functions


- Command line parameters
    - A dedicated `Parameters` struct to store all running parameters
    - Need to define parameters via. `Use` function before using them
    - Command line is parsed by `get_opt`
    - `Set` to set parameter values
    - `Get` to get parameter values
    - Handles vectors as parameter values


- 1D operators for Array1D
    - Per-element operations, e.g. `ForAll` and `ForEach`
    - Vector-Vector operations, e.g. `Add`, `Minus`, `Mul`, `Div`, `Mad`, `Set`
    - Vector-Scalar operations
    - Sort


- Code restructuring
    - Partitioners moved from `gunrock/app` to `gunrock/partitioner`
    - `LB` operator moved from `gunrock/oprtr/edge_map_partitioned_forward` to
      `gunrock/oprtr/LB_advance`
    - `TWC` operator moved from 'gunrock/oprtr/edge_map_forward' to
      `gunrock/oprtr/TWC_advance`


- Optional Boost dependency
    - Utility functions changed to C++11 or implemented
    - CPU references implemented for BFS and SSSP, and will be called when BOOST
      is not available
    - `info` will use RapidJson-based implementation, when Boost is not available

## Known Issues

- Multi-GPU framework not tested
- Operators have decreased performance, due to more than 32 registers used by
  a single thread in the kernels
- RGG and GRMAT generators not working
- SSSP may have incorrect predecessors, due to data racing in marking the
  predecessors within the operator kernels

# Gunrock v0.4 Release Notes
Release 0.4
8th November 2016

Gunrock release 0.4 is a feature release that adds

 - New optimizations to both advance and filter operators
 - Multi-iteration support for BFS, SSSP, BC, CC and PR
 - Better error handling
 - Updates on several interfaces
 - Overall performance improvement for both single and multi-GPU execution

## ChangeLog
 - Integrated direction-optimizing BFS with normal BFS. Now for BFS
   there is only one executable, named bfs. The direction-optimizing
   switch is enabled by the command-line option
   `--direction-optimized`.
 - Added three new strategies for advance (triggered by setting
   `ADVANCE_MODE` accordingly):
    - `ALL_EDGES`, optimized for advance on all edges with all vertices
      of the graph. With `ALL_EDGES`, there is no need to use sorted
      search for load balancing, just binary search over the whole row
      offsets array; used in CC.
    - `LB_CULL`, fused LB advance with a subsequent CULL filter; used in
      BFS, SSSP and BC.
    - `LB_LIGHT_CULL`, fused `LB_LIGHT` advance with a subsequent CULL
      filter; used in BFS, SSSP and BC.
 - Added three new strategies of filter (triggered by setting
   `FILTER_MODE` accordingly):
    - `COMPACTED_CULL`, optimized on several culling heuristics
    - `SIMPLIFIED`, another implementation of the CULL filter, without
      some optimizations
    - `BY_PASS`, optimized for a filter with no elements
      to remove from the input frontier; used in CC and PR.
 - Added multi-iteration support for BFS, SSSP, BC, CC and PR. Users
   can set the number of iterations to run and specify the source node
   for each run (if necessary) via `InitSetup()` defined in gunrock.h.

## Known Issues
 - HITS and SALSA do not have CPU reference yet
 - HITS, SALSA, and who-to-Follow do not have multi-GPU support yet
 - An out-of-memory error (for graphs that approach the memory limit
   of GPUs) will cause result validation to fail
