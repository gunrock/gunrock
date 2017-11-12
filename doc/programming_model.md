Programming Model              {#programming_model}
=================

This page describes the programming model we use in Gunrock.

Gunrock targets graph computations that are generally expressed as "iterative convergent processes". By "iterative," we mean operations that may require running a series of steps repeatedly; by "convergent," we mean that these iterations allow us to approach the correct answer and terminate when that answer is reached. Many graph-computation programming models target a similar goal.

Many of these programming models focus on sequencing steps of _computation_.  Gunrock differs from these programming models in its focus on _manipulating a data structure_. We call this data structure a _frontier_ of vertices or edges. The frontier represents the subset of vertices or edges that is actively participating in the computation. Gunrock operators input one or more frontiers and output one or more frontiers.

Generically, graph operations can often be expressed via a _push_ abstraction (graph elements "push" local private updates into a shared state) or a _pull_ abstraction (graph elements "pull" updates into their local private state) ([Besta et al. publication on push-vs.-pull, HPDC '17](https://htor.inf.ethz.ch/publications/index.php?pub=281)). Gunrock's programming model supports both of these abstractions. (For instance, Gunrock's direction-optimized breadth-first-search supports both push and pull BFS phases. [Mini-Gunrock](https://github.com/gunrock/mini) supports pull-based BFS and PR.) Push-based approaches may or may not require synchronization (such as atomics) for correct operation; this depends on the primitive. Gunrock's idempotence optimization (within its BFS implementation) is an example of a push-based primitive that does not require atomics.

Operators                       {#operators}
=========

In the current Gunrock release, we support four operators.

+ Advance: An _advance_ operator generates a new frontier from the current frontier by visiting the neighbors of the current frontier. A frontier can consist of either vertices or edges, and an advance step can input and output either kind of frontier. Advance is an irregularly-parallel operation for  two reasons: 1)~different vertices in a graph have different numbers of neighbors and 2)~vertices share neighbors. Thus a vertex in an input frontier map to multiple output items. An efficient advance is the most significant challenge of a GPU implementation.

+ Filter:  A _filter_ operator generates a new frontier from the current frontier by choosing a subset of the current frontier based on programmer-specified criteria. Each input item maps to zero or one output items.

+ Compute: A _compute_ operator defines an operation on all elements (vertices or edges) in its input frontier. A programmer-specified compute operator can be used together with all three traversal operators. Gunrock performs that operation in parallel across all elements without regard to order.

+ Segmented intersection: A _segmented intersection_ operator takes two input node frontiers with the same length, or an input edge frontier, and generates both the number of total intersections and the intersected node IDs as the
  new frontier.

We note that compute operators can often be fused with a neighboring operator into a single kernel. This increases producer-consumer locality and improves performance. Thus within Gunrock, we express compute operators as "functors", which are automatically merged into their neighboring operators. Within Gunrock, we express functors in one of two flavors:

+ Cond Functor:
Cond functors input either a vertex id (as in `VertexCond`) or the source id
and the dest id of an edge (as in `EdgeCond`). They also input data specific to
the problem being solved to decide whether the vertex or the edge is valid in
the outgoing frontier.

+ Apply Functor:
Apply functors take the same set of arguments as Cond functors, but perform
user-specified computation on the problem-specific data.

Creating a New Graph Primitive         {#new_primitive}
==============================

To create a new graph primitive, we first put all the problem-specific data
into a data structure. For BFS, we need a per-node label value and a per-node
predecessor value; for CC, we need a per-edge mark value, a per-node component
id value, etc. Then we map the algorithm into the combination of the above
three operators. Next, we need to write different functors for these operators.
Some graph algorithms require only one functor (BFS), but some graph algorithms
need more (CC needs seven). Finally, we write an enactor to load the proper
operator with the proper functor. We provide a graph primitive template. The
problem, functor, and enactor files are under gunrock/app/sample, and the
driver code is under tests/sample.
