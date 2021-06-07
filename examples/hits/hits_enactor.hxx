#include <gunrock/applications/hits.hxx>
#include <gunrock/applications/application.hxx>

#pragma once

namespace gunrock{
namespace hits{

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  using gunrock::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void prepare_frontier(frontier_t<vertex_t>* f,
                        cuda::multi_context_t& context) override {
        //qqq need to prepare_frontier?
  }

  void loop(cuda::multi_context_t& context) override {
    // Data slice qqq
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto update = [] __host__ __device__(
                            vertex_t& source,
                            vertex_t& neighbor,
                            edge_t const& edge,
                            weight_t const& weight
                            ) ->bool{
    P->update_hub(source, neighbor);
    P->update_auth(neighbor, source);

    return true;
    };// end of update

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped,                                operators::advance_direction_t::forward,                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::vertices>(
                    G, E, update, context);

    // Normalize authority and hub
    P->norm_auth();
    P->norm_hub();

    // Swap buffer
    P->swap_buffer();

    // Update iterator
    P->update_iterator();

  }// end of loop

  bool is_converged(cuda::multi_context_t& context) override {
    auto P = this->get_problem();
    return P->is_converged();
  }

};  // struct enactor_t

}// namespace hits
}// namespace gunrock
