/**
 * @file spgemm.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse-Matrix-Matrix multiplication.
 * @version 0.1
 * @date 2022-01-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <gunrock/algorithms/algorithms.hxx>

// Thrust includes (scan, reduce)
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace gunrock {
namespace spgemm {

template <typename a_graph_t, typename b_graph_t>
struct param_t {
  a_graph_t& A;
  b_graph_t& B;
  param_t(a_graph_t& _A, b_graph_t& _B) : A(_A), B(_B) {}
};

template <typename c_graph_t>
struct result_t {
  c_graph_t& C;
  result_t(c_graph_t& _C) : C(_C) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  using edge_t = typename graph_t::edge_type;

  param_type param;
  result_type result;

  problem_t(graph_t& A,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<cuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(A, _context),
        param(_param),
        result(_result) {}

  thrust::device_vector<edge_t> estimated_nz_per_row;
  thrust::host_vector<edge_t> nnz;

  void init() override {
    auto A = this->get_graph();

    // Allocate memory for row-offsets of C = m + 1.
    auto row_offsets = result.C.get_row_offsets();
    memory::allocate(row_offsets,
                     (A.get_number_of_vertices() + 1) * sizeof(edge_t));

    estimated_nz_per_row.resize(A.get_number_of_vertices() + 1);
    nnz.resize(1);
  }

  void reset() override {
    auto policy = this->context->get_context(0)->execution_policy();
    auto g = this->get_graph();
    thrust::fill(policy, estimated_nz_per_row.begin(),
                 estimated_nz_per_row.end(), 0);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<cuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void loop(cuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();

    auto policy = this->context->get_context(0)->execution_policy();

    auto& A = P->param.A;
    auto& B = P->param.B;
    auto& C = P->result.C;

    auto row_offsets = C.get_row_offsets();
    auto column_indices = C.get_column_indices();
    auto nonzero_values = C.get_nonzero_values();

    auto& estimated_nz_per_row = P->estimated_nz_per_row;
    auto estimated_nz_ptr = estimated_nz_per_row.data().get();

    /// Step 1. Count the upperbound of number of nonzeros per row of C.
    auto upperbound_nonzeros =
        [=] __host__ __device__(vertex_t const& m,  // ... source (row index)
                                vertex_t const& k,  // neighbor (column index)
                                edge_t const& nz_idx,  // edge (row ↦ column)
                                weight_t const& nz     // weight (nonzero).
                                ) -> bool {
      // Compute number of nonzeros of the sparse-matrix C for each row.
      math::atomic::add(&estimated_nz_ptr[m], B.get_number_of_neighbors(k));
      return false;  // ignored.
    };

    operators::advance::execute<operators::load_balance_t::thread_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::none>(
        A, E, upperbound_nonzeros, context);

    /// Step X. Calculate upperbound of C's row-offsets.
    thrust::exclusive_scan(policy, P->estimated_nz_per_row.begin(),
                           P->estimated_nz_per_row.end(),
                           thrust::device_pointer_cast(row_offsets), edge_t(0),
                           thrust::plus<edge_t>());

    /// Step X. Calculate the upperbound of total number of nonzeros in the
    /// sparse-matrix C.
    thrust::copy(
        thrust::device_pointer_cast(row_offsets) + A.get_number_of_vertices(),
        thrust::device_pointer_cast(row_offsets) + A.get_number_of_vertices() +
            1,
        P->nnz.begin());

    edge_t estimated_nzs = P->nnz[0];

    /// Step . Allocate upperbound memory for C's values and column indices.
    memory::allocate(column_indices, estimated_nzs * sizeof(vertex_t));
    memory::allocate(nonzero_values, estimated_nzs * sizeof(weight_t));

    thrust::fill(policy, column_indices, column_indices + estimated_nzs, -1);
    thrust::fill(policy, nonzero_values, nonzero_values + estimated_nzs,
                 weight_t(0));

    /// Step X. Calculate C's column indices and values.
    auto gustavsons =
        [=] __device__(
            vertex_t const& m,  // ... source (A: row index)
            vertex_t const& k,  // neighbor (A: column index or B: row index)
            edge_t const& a_nz_idx,  // edge (A: row ↦ column)
            weight_t const& a_nz     // weight (A: nonzero).
            ) -> bool {
      // Get the number of nonzeros in row k of sparse-matrix B.
      auto offset = B.get_starting_edge(k);
      auto nnz = B.get_number_of_neighbors(k);

      auto c_offset = row_offsets[m];

      // Loop over all the nonzeros in row k of sparse-matrix B.
      for (edge_t b_nz_idx = offset; b_nz_idx < (offset + nnz); ++b_nz_idx) {
        auto n = B.get_destination_vertex(b_nz_idx);
        auto b_nz = B.get_edge_weight(b_nz_idx);
        auto c_nz_idx = c_offset + n;

        column_indices[c_nz_idx] = n;
        math::atomic::add(&nonzero_values[c_nz_idx], a_nz * b_nz);
      }
      return false;  // ignored.
    };

    operators::advance::execute<operators::load_balance_t::thread_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::none>(
        A, E, gustavsons, context);

    /// Step X. Fix-up, i.e., remove overestimated nonzeros and rellocate the
    /// storage as necessary.
    auto real_nonzeros = [=] __device__(vertex_t const& row) -> void {
      auto overestimated_nzs = 0;
      for (edge_t k = row_offsets[row]; k < row_offsets[row + 1]; ++k) {
        if (column_indices[k] == -1 && nonzero_values[k] == 0)
          overestimated_nzs++;
      }

      estimated_nz_ptr[row] -= overestimated_nzs;
    };

    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        A, real_nonzeros, context);

    thrust::exclusive_scan(policy, P->estimated_nz_per_row.begin(),
                           P->estimated_nz_per_row.end(),
                           thrust::device_pointer_cast(row_offsets), edge_t(0),
                           thrust::plus<edge_t>());

    auto it = thrust::copy_if(
        policy, column_indices, column_indices + estimated_nzs, column_indices,
        [] __device__(const vertex_t& x) -> bool { return x != -1; });

    auto C_nnz = thrust::distance(column_indices, it);

    auto itv = thrust::copy_if(policy, nonzero_values,
                               nonzero_values + estimated_nzs, nonzero_values,
                               [] __device__(const weight_t& nz) -> bool {
                                 return nz != weight_t(0);
                               });

    error::throw_if_exception(C_nnz != thrust::distance(nonzero_values, itv),
                              "NNZ check failed.");

    /// Step X. Finally set the sparse-matrix C's data, again.
    C.set(A.get_number_of_vertices(), C_nnz, row_offsets, column_indices,
          nonzero_values);
  }

  /**
   * @brief SpGEMM converges within one iteration.
   *
   * @param context The context of the execution (unused).
   * @return true returns true after one iteration.
   */
  virtual bool is_converged(cuda::multi_context_t& context) {
    return this->iteration == 0 ? false : true;
  }
};  // struct enactor_t

template <typename graph_t, typename b_graph_t, typename c_graph_t>
float run(graph_t& A,
          b_graph_t& B,
          c_graph_t& C,
          std::shared_ptr<cuda::multi_context_t> context =
              std::shared_ptr<cuda::multi_context_t>(
                  new cuda::multi_context_t(0))  // Context
) {
  using param_type = param_t<graph_t, b_graph_t>;
  using result_type = result_t<c_graph_t>;

  param_type param(A, B);
  result_type result(C);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(A, param, result, context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers:
  enactor_properties_t props;
  props.self_manage_frontiers = true;

  enactor_type enactor(&problem, context, props);
  return enactor.enact();
}

}  // namespace spgemm
}  // namespace gunrock