// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sage_app.cuh
 *
 * @brief graphSage application
 */

namespace gunrock {
namespace app {
namespace sage {

#include <gunrock/app/sage/sage_app.cuh>

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "Wf1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<weight matrix for W^1 matrix in algorithm 2, feature part>\n"
      "\t dimension 64 by 128 for pokec;\n"
      "\t It should be child feature length by a value you want for W2 layer",
      __FILE__, __LINE__));

  // GUARD_CU(parameters.Use<int>(
  //     "Wf1-dim0",
  //     util::REQUIRED_ARGUMENT | util::SINGLE_VALUE |
  //     util::OPTIONAL_PARAMETER, 64, "Wf1 matrix row dimension",
  //     __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "Wf1-dim1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      128, "Wf1 matrix column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "Wa1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<weight matrix for W^1 matrix in algorithm 2, aggregation part>\n"
      "\t dimension 64 by 128 for pokec;\n"
      "\t It should be leaf feature length by a value you want for W2 layer",
      __FILE__, __LINE__));

  // GUARD_CU(parameters.Use<int>(
  //    "Wa1-dim0",
  //    util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
  //    64,
  //    "Wa1 matrix row dimension",
  //    __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "Wa1-dim1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      128, "Wa1 matrix column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "Wf2",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<weight matrix for W^2 matrix in algorithm 2, feature part>\n"
      "\t dimension 256 by 128 for pokec;\n"
      "\t It should be source_temp length by output length",
      __FILE__, __LINE__));

  // GUARD_CU(parameters.Use<int>(
  //     "Wf2-dim0",
  //     util::REQUIRED_ARGUMENT | util::SINGLE_VALUE |
  //     util::OPTIONAL_PARAMETER, 256, "Wf2 matrix row dimension",
  //     __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "Wf2-dim1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      128, "Wf2 matrix column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "Wa2",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<weight matrix for W^2 matrix in algorithm 2, aggregation part>\n"
      "\t dimension 256 by 128 for pokec;\n"
      "\t It should be child_temp length by output length",
      __FILE__, __LINE__));

  // GUARD_CU(parameters.Use<int>(
  //    "Wa2-dim0",
  //    util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
  //    256,
  //    "Wa2 matrix row dimension",
  //    __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "Wa2-dim1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      128, "Wa2 matrix column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "features",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<features matrix>\n"
      "\t dimension |V| by 64 for pokec;\n",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "feature-column",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      64, "feature column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "num-children-per-source",  // num_neigh1
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "number of sampled children per source", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "num-leafs-per-child",  // num_neight2
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "number of sampled leafs per child; default is the same as "
      "num-children-per-source",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "batch-size",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      65536, "number of source vertex to process in one iteration", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<int>(
      "rand-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "seed for random number generator; default will use time(NULL)", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "custom-kernels",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      true, "whether to use custom CUDA kernels", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "omp-threads",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      32, "number of threads to run CPU reference", __FILE__, __LINE__));

  return retval;
}

}  // namespace sage
}  // namespace app
}  // namespace gunrock

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform SSSP
 * @param[in]  sources     Sources to begin traverse, one for each run
 * @param[in]  mark_preds  Whether to output predecessor info
 * @param[out] distances   Return shortest distance to source per vertex
 * @param[out] preds       Return predecessors of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
double sage(const int               num_nodes,
            const int               num_edges,
            const int               *row_offsets,
            const int               *col_indices,
            const int               *edge_values,
            const int               *source_result,
            const int               num_runs,
            gunrock::util::Location allocated_on = gunrock::util::HOST
) {
  return sage<int, int, int, int>(num_nodes, num_edges, row_offsets,
    col_indices, edge_values, source_result, num_runs, allocated_on);
}
