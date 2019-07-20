// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_unitests.cu
 *
 * @brief Main test driver for all googletests.
 * @source
 * https://github.com/google/googletest/blob/master/googletest/docs/Primer.md
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

#include <gunrock/app/hello/hello_app.cu>
#include <gunrock/app/test_base.cuh>

#include <gtest/gtest.h>

/**
 * @brief: Gunrock: Google tests -- list of tests
 * found in this directory, testing core functionality
 * of gunrock: primitives, operators, device intrinsics,
 * etc.
 *
 */

// bug:: malloc_consolidate(): invalid chunk size
//#include "test_lib_pr.h"

// Tests the RepeatFor Operator
#include "test_repeatfor.h"

// Tests Segmented Reduction (device)
#include "test_segreduce.h"

using namespace gunrock;

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<bool>("googletest", util::OPTIONAL_PARAMETER, true,
                                "Example parameter for googletest", __FILE__,
                                __LINE__));

  return retval;
}

/******************************************************************************
 * Main
 ******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct {
  /**
   * @brief the actual main function, after type switching
   * @tparam VertexT    Type of vertex identifier
   * @tparam SizeT      Type of graph size, i.e. type of edge identifier
   * @tparam ValueT     Type of edge values
   * @param  parameters Command line parameters
   * @param  v,s,val    Place holders for type deduction
   * \return cudaError_t error message(s), if any
   */
  template <typename VertexT,  // Use int as the vertex identifier
            typename SizeT,    // Use int as the graph size type
            typename ValueT>   // Use int as the value type
  cudaError_t
  operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
    // CLI parameters
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    cudaError_t retval = cudaSuccess;
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;

  util::Parameters parameters("test unittests");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(UseParameters(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }

  // Run all tests using the google tests
  // framework.
  ::testing::InitGoogleTest(&argc, argv);
  RUN_ALL_TESTS();

  return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B |   // app::SIZET_U64B |
                           app::VALUET_F32B |  // app::VALUET_F64B |
                           app::DIRECTED | app::UNDIRECTED>(parameters,
                                                            main_struct());
}
