Graph
=======

.. doxygenclass:: gunrock::graph::graph_t

.. doxygenstruct:: gunrock::graph::vertex_pair_t

.. doxygenstruct:: gunrock::graph::edge_pair_t

Properties
-----------

.. doxygenstruct:: gunrock::graph::graph_properties_t

.. doxygenenum:: gunrock::graph::view_t

Functions on Graph
-------------------

.. doxygenfunction:: gunrock::graph::get_average_degree

.. doxygenfunction:: gunrock::graph::get_degree_standard_deviation

.. doxygenfunction:: gunrock::graph::build_degree_histogram

.. doxygenfunction:: gunrock::graph::remove_self_loops

Graph Builder
--------------

.. doxygenfunction:: gunrock::graph::build::from_csr(format::csr_t<space, vertex_t, edge_t, weight_t>& csr)