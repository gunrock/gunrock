Gunrock API Reference
======================

.. highlight:: c++

Algorithms
----------

.. doxygenfunction:: gunrock::bfs::run

.. doxygenfunction:: gunrock::bc::run(graph_t& G, typename graph_t::vertex_type single_source, typename graph_t::weight_type* bc_values, std::shared_ptr<gcuda::multi_context_t> context)
   :outline:

.. doxygenfunction:: gunrock::bc::run(graph_t& G, typename graph_t::weight_type* bc_values)
   :outline:

.. doxygenfunction:: gunrock::color::run(graph_t& G, param_t& param, result_t<typename graph_t::vertex_type>& result, std::shared_ptr<gcuda::multi_context_t> context)
   :outline:

.. doxygenfunction:: gunrock::color::run(graph_t& G, typename graph_t::vertex_type* colors, operators::filter_algorithm_t filter_algorithm, std::shared_ptr<gcuda::multi_context_t> context)
   :outline:

.. doxygenfunction:: gunrock::geo::run

.. doxygenfunction:: gunrock::hits::run

.. doxygenfunction:: gunrock::kcore::run

.. doxygenfunction:: gunrock::mst::run

.. doxygenfunction:: gunrock::ppr::run(graph_t& G, param_t<typename graph_t::vertex_type, typename graph_t::weight_type>& param, result_t<typename graph_t::weight_type>& result, std::shared_ptr<gcuda::multi_context_t> context)
   :outline:

.. doxygenfunction:: gunrock::pr::run

.. doxygenfunction:: gunrock::spgemm::run

.. doxygenfunction:: gunrock::spmv::run

.. doxygenfunction:: gunrock::sssp::run

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