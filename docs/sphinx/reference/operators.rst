Graph Operators
======================

.. highlight:: c++

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