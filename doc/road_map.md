Road Map {#road_map}
=====================

 - **Framework:** We are exploring more operators such as neighborhood
   reduction and segmented intersection. Generally we want to find the right
   set of operators that can abstract most graph primitives while delivering
   high performance.

 - **API:** We would like to make an API refactoring to simplify parameter 
   passing and to isolate parts of the library that dependencies are not
   necessary. The target is to make the frontier concept more clear, and
   to promote code reuse.

 - **Primitives:** Our near-term goal is to graduate several primitives in dev
   branch including A* search, weighted label propagation, subgraph matching,
   triangle counting, and clustering coefficients; implement maximal
   independent set, max flow, and graph coloring algorithms, build better
   support for bipartite graph algorithms, and explore community detection
   algorithms. Our long term goals include algorithms on dynamic graphs,
   multi-level priority queue support, graph partitioning, and more flexible
   and scalable multi-GPU algorithms.

