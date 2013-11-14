Programming Model              {#programming_model}
=================

This page describes the programming model we use in Gunrock.

Operators                       {#operators}
=========

Gunrock supports three operators that form the basis of graph
computation:

1. Forward edge mapping:
This operator inputs a vertex frontier, traverses all the edges that
start with vertices in this frontier, and returns all unvisited
vertices as the new frontier.

2. Backward edge mapping:
This operator also inputs a vertex frontier and outputs a new
frontier. However, it instead traverses all *unvisited* vertices and
outputs unvisited vertices that share an edge with current vertices.

3. Vertex mapping:
This operator inputs a queue of vertices, culls some vertices, and
outputs the rest into a new vertex queue. It can also operate on edges
as what we have done in CC problem.


Functors                        {#functors}
========

The operators traverse edges and vertices in a graph. To do the actual
computation, we need functors. Functors are divided into two
categories:

1. Cond Functor:
Cond functors input either a vertex id (as in `VertexCond`) or the
source id and the dest id of an edge (as in `EdgeCond`). They also
input data specific to the problem being solved to decide whether the
vertex or the edge is valid in the outgoing frontier.

2. Apply Functor:
Apply functors take the same set of arguments as Cond functors, but
perform user-specified computation on the problem-specific data.

Create New Graph Primitive         {#new_primitive}
==========================

To create a new graph primitive, we first put all the problem-specific
data into a data structure. For BFS, we need a per-node label value
and a per-node predecessor value; for CC, we need a per-edge mark
value, a per-node component id value, etc. Then we map the algorithm
into the combination of the above three operators. Next, we need to
write different functors for these operators. Some graph algorithms
require only one functor (BFS), but some graph algorithms need more
(CC needs seven). Finally, we write an enactor to load the proper
operator with the proper functor.
