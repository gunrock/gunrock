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

template <typename graph_t>
struct param_t {
  graph_t& A;
  graph_t& B;
  param_t(graph_t& _A, graph_t& _B) : A(_A), B(_B) {}
};

template <typename csr_t>
struct result_t {
  csr_t& C;
  result_t(csr_t& _C) : C(_C) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  using edge_t = typename graph_t::edge_type;
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  param_type& param;
  result_type& result;

  problem_t(graph_t& A,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(A, _context),
        param(_param),
        result(_result) {}

  thrust::device_vector<edge_t> estimated_nz_per_row;
  thrust::host_vector<edge_t> nnz;

  void init() override {
    auto& A = this->param.A;
    // auto& C = this->result.C;

    estimated_nz_per_row.resize(A.get_number_of_vertices());
    nnz.resize(1);
  }

  void reset() override {
    auto policy = this->context->get_context(0)->execution_policy();
    thrust::fill(policy, estimated_nz_per_row.begin(),
                 estimated_nz_per_row.end(), 0);

    // Reset NNZ.
    nnz[0] = 0;

    // Reset sparse-matrix C.
    auto& C = this->result.C;
    C.row_offsets.clear();
    C.column_indices.clear();
    C.nonzero_values.clear();
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();

    auto policy = this->context->get_context(0)->execution_policy();

    auto& A = P->param.A;
    auto& B = P->param.B;
    auto& C = P->result.C;

    auto& row_offsets = C.row_offsets;
    auto& column_indices = C.column_indices;
    auto& nonzero_values = C.nonzero_values;

    auto& estimated_nz_per_row = P->estimated_nz_per_row;
    auto estimated_nz_ptr = estimated_nz_per_row.data().get();

    // Resize row-offsets.
    row_offsets.resize(A.get_number_of_vertices() + 1);

    thrust::fill(policy, estimated_nz_per_row.begin(),
                 estimated_nz_per_row.end(), 0);

    /// Step 1. Count the upperbound of number of nonzeros per row of C.
    auto upperbound_nonzeros =
        [=] __host__ __device__(vertex_t const& m,  // ... source (row index)
                                vertex_t const& k,  // neighbor (column index)
                                edge_t const& nz_idx,  // edge (row ↦ column)
                                weight_t const& nz     // weight (nonzero).
                                ) -> bool {
      // Compute number of nonzeros of the sparse-matrix C for each row.
      math::atomic::add(&(estimated_nz_ptr[m]), B.get_number_of_neighbors(k));
      return false;
    };

    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::none>(
        A, E, upperbound_nonzeros, context);

    /// Step X. Calculate upperbound of C's row-offsets.
    thrust::exclusive_scan(policy, P->estimated_nz_per_row.begin(),
                           P->estimated_nz_per_row.end(), row_offsets.begin(),
                           edge_t(0), thrust::plus<edge_t>());

    thrust::copy(row_offsets.begin() + A.get_number_of_vertices() - 1,
                 row_offsets.begin() + A.get_number_of_vertices(),
                 row_offsets.begin() + A.get_number_of_vertices());

    /// Step X. Calculate the upperbound of total number of nonzeros in the
    /// sparse-matrix C.
    thrust::copy(row_offsets.begin() + A.get_number_of_vertices(),
                 row_offsets.begin() + A.get_number_of_vertices() + 1,
                 P->nnz.begin());

    edge_t estimated_nzs = P->nnz[0];

    /// Step . Allocate upperbound memory for C's values and column indices.
    column_indices.resize(estimated_nzs, -1);
    nonzero_values.resize(estimated_nzs, weight_t(0));

    edge_t* row_off = row_offsets.data().get();
    vertex_t* col_ind = column_indices.data().get();
    weight_t* nz_vals = nonzero_values.data().get();

    /// Step X. Calculate C's column indices and values.
    auto gustavsons =
        [=] __host__ __device__(
            vertex_t const& m,  // ... source (A: row index)
            vertex_t const& k,  // neighbor (A: column index or B: row index)
            edge_t const& a_nz_idx,  // edge (A: row ↦ column)
            weight_t const& a_nz     // weight (A: nonzero).
            ) -> bool {
      // Get the number of nonzeros in row k of sparse-matrix B.
      auto offset = B.get_starting_edge(k);
      auto nnz = B.get_number_of_neighbors(k);
      auto c_offset = thread::load(&row_off[m]);

      // Loop over all the nonzeros in row k of sparse-matrix B.
      for (edge_t b_nz_idx = offset; b_nz_idx < (offset + nnz); ++b_nz_idx) {
        auto n = B.get_destination_vertex(b_nz_idx);
        auto b_nz = B.get_edge_weight(b_nz_idx);

        // Calculate c's nonzero index.
        std::size_t c_nz_idx = c_offset + n;

        // Assign column index.
        thread::store(&col_ind[c_nz_idx], n);

        // Accumulate the nonzero value.
        math::atomic::add(nz_vals + c_nz_idx, a_nz * b_nz);
      }
      return false;
    };

    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::none>(
        A, E, gustavsons, context);

    /// Step X. Fix-up, i.e., remove overestimated nonzeros and rellocate the
    /// storage as necessary.
    auto real_nonzeros = [=] __host__ __device__(vertex_t const& row) -> void {
      edge_t overestimated_nzs = 0;
      // For all nonzeros within the row of C.
      for (auto nz = row_off[row]; nz < row_off[row + 1]; ++nz) {
        // Find the invalid column indices and zero-values, they represent
        // overestimated nonzeros.
        if (col_ind[nz] == -1)
          overestimated_nzs += 1;
      }
      // Remove overestimated nonzeros.
      estimated_nz_ptr[row] -= overestimated_nzs;
    };

    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        A, real_nonzeros, context);

    thrust::exclusive_scan(policy, P->estimated_nz_per_row.begin(),
                           P->estimated_nz_per_row.end(), row_offsets.begin(),
                           edge_t(0), thrust::plus<edge_t>());

    thrust::copy(row_offsets.begin() + A.get_number_of_vertices() - 1,
                 row_offsets.begin() + A.get_number_of_vertices(),
                 row_offsets.begin() + A.get_number_of_vertices());

    /// Step X. Calculate the upperbound of total number of nonzeros in the
    /// sparse-matrix C.
    thrust::copy(row_offsets.begin() + A.get_number_of_vertices(),
                 row_offsets.begin() + A.get_number_of_vertices() + 1,
                 P->nnz.begin());

    auto itc = thrust::copy_if(
        policy, column_indices.begin(), column_indices.end(),
        column_indices.begin(),
        [] __device__(const vertex_t& x) -> bool { return x != -1; });

    auto itv = thrust::copy_if(policy, nonzero_values.begin(),
                               nonzero_values.end(), nonzero_values.begin(),
                               [] __device__(const weight_t& nz) -> bool {
                                 return nz != weight_t(0);
                               });

    auto idx_nnz = thrust::distance(column_indices.begin(), itc);
    auto nz_nnz = thrust::distance(nonzero_values.begin(), itv);

    std::cout << "idx_nnz ? nz_nnz : " << idx_nnz << " ? " << nz_nnz
              << std::endl;

    /// Step X. Make sure C is set.
    C.number_of_rows = A.get_number_of_vertices();
    C.number_of_columns = B.get_number_of_vertices();
    C.number_of_nonzeros = P->nnz[0];
  }

  /**
   * @brief SpGEMM converges within one iteration.
   *
   * @param context The context of the execution (unused).
   * @return true returns true after one iteration.
   */
  virtual bool is_converged(gcuda::multi_context_t& context) {
    return this->iteration == 0 ? false : true;
  }
};  // struct enactor_t

template <typename graph_t, typename csr_t>
float run(graph_t& A,
          graph_t& B,
          csr_t& C,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using param_type = param_t<graph_t>;
  using result_type = result_t<csr_t>;

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