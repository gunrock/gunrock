Programming Model              {#programming_model}
=================

This page describes the programming model we use in Gunrock.

Operators                       {#operators}
=========

Gunrock supports two operators that form the basis of graph computation:

+ Advance:
This operator represents the most common operation in graph: advancing from one
frontier to another through edges. It takes either a vertex frontier or an edge
frontier as the input, visits the edges connect to the elements in the
frontier, and returns a new frontier which contains either the edges or the
vertices it reaches.

+ Filter:
This operator inputs a queue of elements, culls some elements by testing
whether it meets the criteria defined by users, and outputs the rest into a new
queue. It can also do computations while visiting the elements in the queue and
doing the validation test.


Functors                        {#functors}
========

The operators traverse edges and vertices in a graph. To do the actual
computation, we need functors. Functors are divided into two categories:

+ Cond Functor:
Cond functors input either a vertex id (as in `VertexCond`) or the source id
and the dest id of an edge (as in `EdgeCond`). They also input data specific to
the problem being solved to decide whether the vertex or the edge is valid in
the outgoing frontier.

+ Apply Functor:
Apply functors take the same set of arguments as Cond functors, but perform
user-specified computation on the problem-specific data.

Create New Graph Primitive         {#new_primitive}
==========================

To create a new graph primitive, we first put all the problem-specific data
into a data structure. For BFS, we need a per-node label value and a per-node
predecessor value; for CC, we need a per-edge mark value, a per-node component
id value, etc. Then we map the algorithm into the combination of the above
three operators. Next, we need to write different functors for these operators.
Some graph algorithms require only one functor (BFS), but some graph algorithms
need more (CC needs seven). Finally, we write an enactor to load the proper
operator with the proper functor.
