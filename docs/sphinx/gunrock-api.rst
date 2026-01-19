Gunrock API Reference
======================

.. highlight:: c++

This page provides quick access to commonly-used algorithm and operator functions.
For complete API documentation including all overloads, see the auto-generated API reference.

Algorithms
----------

BFS (Breadth-First Search)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: gunrock::bfs::run

BC (Betweenness Centrality)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`bc.hxx <file_include_gunrock_algorithms_bc.hxx>` for BC algorithm documentation with multiple overloads.

Color (Graph Coloring)
^^^^^^^^^^^^^^^^^^^^^^

See :ref:`color.hxx <file_include_gunrock_algorithms_color.hxx>` for graph coloring algorithm documentation with multiple overloads.

Geo (Graph Embedding)
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: gunrock::geo::run

HITS
^^^^

.. doxygenfunction:: gunrock::hits::run

K-Core
^^^^^^

.. doxygenfunction:: gunrock::kcore::run

MST (Minimum Spanning Tree)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: gunrock::mst::run

PPR (Personalized PageRank)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`ppr.hxx <file_include_gunrock_algorithms_ppr.hxx>` for PPR algorithm documentation with multiple overloads.

PR (PageRank)
^^^^^^^^^^^^^

.. doxygenfunction:: gunrock::pr::run

SpGEMM
^^^^^^

.. doxygenfunction:: gunrock::spgemm::run

SpMV
^^^^

.. doxygenfunction:: gunrock::spmv::run

SSSP (Single-Source Shortest Path)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: gunrock::sssp::run

TC (Triangle Counting)
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: gunrock::tc::run

Operators
----------

.. doxygenfunction:: gunrock::operators::advance::execute(graph_t& G, operator_t op, frontier_t* input, frontier_t* output, work_tiles_t& segments, gcuda::multi_context_t& context)

.. doxygenfunction:: gunrock::operators::advance::execute(graph_t& G, enactor_type* E, operator_type op, gcuda::multi_context_t& context, bool swap_buffers = true)

.. doxygenfunction:: gunrock::operators::filter::execute(graph_t& G, operator_t op, frontier_t* input, frontier_t* output, gcuda::multi_context_t& context)

.. doxygenfunction:: gunrock::operators::filter::execute(graph_t& G, enactor_type* E, operator_t op, gcuda::multi_context_t& context, bool swap_buffers = true)

.. doxygenfunction:: gunrock::operators::batch::execute

.. doxygenfunction:: gunrock::operators::parallel_for::execute(frontier_t& f, func_t op, gcuda::multi_context_t& context) 

.. doxygenfunction:: gunrock::operators::parallel_for::execute(graph_t& G, func_t op, gcuda::multi_context_t& context)

.. doxygenfunction:: gunrock::operators::neighborreduce::execute

.. doxygenfunction:: gunrock::operators::uniquify::execute(frontier_t* input, frontier_t* output, gcuda::multi_context_t& context, bool best_effort_uniquification = false, const float uniquification_percent = 100)

.. doxygenfunction:: gunrock::operators::uniquify::execute(enactor_type* E, gcuda::multi_context_t& context, bool best_effort_uniquification = false, const float uniquification_percent = 100, bool swap_buffers = true)

Frontiers
----------

.. doxygenclass:: gunrock::frontier::frontier_t

.. doxygenclass:: gunrock::frontier::vector_frontier_t

.. doxygenenum:: gunrock::frontier::frontier_view_t

.. doxygenenum:: gunrock::frontier::frontier_kind_t