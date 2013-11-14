Programming Model              {#programming_model}
=================

This page describes the programming models we use in Gunrock library.

Operators                       {#operators}
=========

In Gunrock we have three operators which serve as the core of the graph computation:

1. Forward edge mapping:
This operator takes a vertex frontier as the input, traverses all the edges start with
vertices in this frontier and returns all the unvisited vertices as the new frontier.

2. Backward edge mapping:
This operator also takes a vertex frontier as the input, however it traverses all the
unvisited vertices and put those belong to the same edge with the current vertices in
the new frontier.

3. Vertex mapping:
This operator takes a queue of vertices, culls some vertices and puts the rest into a
new vertex queue. It can also work on all the edges as what we have done in CC problem.


Functors                        {#functors}
========

Now we have the operators to traverse edge/vertex in a graph. To do the actual computation,
we need functors. Functors are devided into two categories:

1. Cond Functor:
Cond functor takes either the current vertex id (as in the VertexCond) or the source id and
the dest id of an edge (as in the EdgeCond) and the problem specific data to decide whether
the vertex or the edge is valid in the outgoing frontier.

2. Apply Functor:
Apply functor takes the same set of arguments as the Cond functor. It applies a certain kind
of computation on the problem specific data.

Create New Graph Primitive         {#new_primitive}
==========================
To create a new graph primitive, we first need to put all the problem specific data into a
problem data structure. For BFS, we need per node label value, per node predecessor value;
for CC, we need per edge mark value, per node component id value, etc.. Then we need to fit
the algorithm into the combination of the above three operators. After that we need to write
different functors for these operators. Some graph algorithms require only one set of functor
(BFS), but some graph algorithm needs more set of functors (CC needs seven). Finally we write
an enactor to load the proper operator with proper functor. This can serve as a guide to any
process of creating new garph primitives for Gunrock library.
