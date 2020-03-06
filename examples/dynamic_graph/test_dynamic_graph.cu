// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_dynamic_graph.cu
 *
 * @brief Simple test driver program for dynamic graph building.
 */

#include <gunrock/gunrock.h>


// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph defintions
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>

#include <gunrock/app/test_base.cuh>

using namespace gunrock;

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
    
    using WeightedGraphT = app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_EDGE_VALUES | 
                                    graph::HAS_CSR | 
                                    graph::HAS_COO |
                                    graph::HAS_DYN>;

    using UnweightedGraphT = app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_CSR | 
                                    graph::HAS_COO |
                                    graph::HAS_DYN>;

    using CSRGraphT = app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_EDGE_VALUES | 
                                    graph::HAS_CSR>;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    
    WeightedGraphT weighted_graph;
    //UnweightedGraphT unweighted_graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, weighted_graph));
    //GUARD_CU(graphio::LoadGraph(parameters, unweighted_graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());


    CSRGraphT csr_graph;

    SizeT nodes = weighted_graph.csr().nodes;
    SizeT edges = weighted_graph.csr().edges;

    csr_graph.csr().Allocate(nodes, edges, util::DEVICE | util::HOST);

    weighted_graph.dyn().ToCsr(csr_graph.csr());

    csr_graph.csr().Move(util::DEVICE, util::HOST);

    auto ref_graph = weighted_graph.csr();
    auto res_graph = csr_graph.csr();

    ref_graph.Sort();
    res_graph.Sort();


    for(SizeT node = 0; node < nodes; node++){
    	SizeT ref_start = ref_graph.row_offsets[node];
    	SizeT ref_end   = ref_graph.row_offsets[node + 1];

    	SizeT res_start = res_graph.row_offsets[node];
    	SizeT res_end   = res_graph.row_offsets[node + 1];

    	printf("%i:[ %i - %i] =? ",node, ref_start, ref_end);
    	printf("[ %i - %i] \n", res_start, res_end);

    	//
    	for(VertexT eid = ref_start; eid <= ref_end; eid++){
    		VertexT e_ref = ref_graph.column_indices[eid];
    		VertexT e_res = ref_graph.column_indices[eid];
        	printf("E: %i:[ %i =? %i] \n",eid, e_ref, e_res);

    	}
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test sssp");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B |
                           app::SIZET_U32B | 
                           app::VALUET_S32B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());


  //return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
  //                         app::SIZET_U32B | app::SIZET_U64B |
  //                         app::VALUET_S32B | app::DIRECTED | app::UNDIRECTED>(
  //    parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
