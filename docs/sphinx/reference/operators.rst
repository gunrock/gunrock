Graph Operators
======================

.. highlight:: c++

Advance Operator
-----------------

.. doxygenfunction:: gunrock::operators::advance::execute(graph_t& G, operator_t op, frontier_t* input, frontier_t* output, work_tiles_t& segments, gcuda::multi_context_t& context)

.. doxygenfunction:: gunrock::operators::advance::execute(graph_t& G, enactor_type* E, operator_type op, gcuda::multi_context_t& context, bool swap_buffers = true)

.. doxygenenum:: gunrock::operators::load_balance_t

.. doxygenenum:: gunrock::operators::advance_io_type_t

.. doxygenenum:: gunrock::operators::advance_direction_t

Filter Operator
----------------

.. doxygenfunction:: gunrock::operators::filter::execute(graph_t& G, operator_t op, frontier_t* input, frontier_t* output, gcuda::multi_context_t& context)

.. doxygenfunction:: gunrock::operators::filter::execute(graph_t& G, enactor_type* E, operator_t op, gcuda::multi_context_t& context, bool swap_buffers = true)

.. doxygenenum:: gunrock::operators::filter_algorithm_t

Batch Operator
----------------

.. doxygenfunction:: gunrock::operators::batch::execute

Parallel-For Operator
----------------------

.. doxygenfunction:: gunrock::operators::parallel_for::execute(frontier_t& f, func_t op, gcuda::multi_context_t& context) 

.. doxygenfunction:: gunrock::operators::parallel_for::execute(graph_t& G, func_t op, gcuda::multi_context_t& context)

.. doxygenenum:: gunrock::operators::parallel_for_each_t

Neighbor Reduce Operator
-------------------------

.. doxygenfunction:: gunrock::operators::neighborreduce::execute

Uniquify Operator
------------------

.. doxygenfunction:: gunrock::operators::uniquify::execute(frontier_t* input, frontier_t* output, gcuda::multi_context_t& context, bool best_effort_uniquification = false, const float uniquification_percent = 100)

.. doxygenfunction:: gunrock::operators::uniquify::execute(enactor_type* E, gcuda::multi_context_t& context, bool best_effort_uniquification = false, const float uniquification_percent = 100, bool swap_buffers = true)

.. doxygenenum:: gunrock::operators::uniquify_algorithm_t