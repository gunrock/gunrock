// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kcore_test.cu
 *
 * @brief Test related functions for k-core
 */

#pragma once

namespace gunrock {
namespace app {
namespace kcore {

/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference vertex k-core decomposition implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
 * @param[out]  k_cores       Computed k-core value for each vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * return       double        Time taken for the k-core computation
 */
template <typename GraphT>
double CPU_Reference(const GraphT &graph,
                     typename GraphT::SizeT *k_cores,
                     bool quiet) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  //CPU reference implementation
  int *degrees = (int *)malloc(sizeof(int) * graph.nodes);
  bool *remove = (bool *)malloc(sizeof(bool) * graph.nodes);
  bool *deleted = (bool *)malloc(sizeof(bool) * graph.nodes);
  for (SizeT v = 0; v < graph.nodes; ++v) {
    degrees[v] = graph.CsrT::GetNeighborListLength(v);
    k_cores[v] = 0;
    remove[v] = false;
    if (degrees[v] == 0) {
      deleted[v] = true;
    }
    else {
      deleted[v] = false;
    }
  }

  SizeT numRemaining = 0;
  SizeT numToRemove = 0;

  for (SizeT k = 1; k < graph.nodes; ++k) {
    while (true) {
      numRemaining = 0;
      numToRemove = 0;
      for (SizeT v = 0; v < graph.nodes; ++v) {
        if (deleted[v] == false) {
          if (degrees[v] > k) {
            ++numRemaining;
          }
          else {
            remove[v] = true;
            k_cores[v] = k;
            degrees[v] = 0;
            ++numToRemove;
          }
        }
      }
      if (numToRemove == 0) break;
      else {
        for (SizeT v = 0; v < graph.nodes; ++v) {
          if (remove[v] == true) {
            SizeT eStart = graph.CsrT::GetNeighborListOffset(v);
            SizeT numNeighbors = graph.CsrT::GetNeighborListLength(v);
            for (SizeT e = 0; e < numNeighbors; ++e) {
              SizeT destVertex = graph.CsrT::GetEdgeDest(eStart + e);
              --degrees[destVertex];
            }
            remove[v] = false;
            deleted[v] = true;
          }
        }
      }
    }
    if (numRemaining == 0) break;
  }
    
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  return elapsed;
}

/**
 * @brief Validation of k-core results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Execution parameters
 * @param[in]  graph         Input graph
 * @param[in]  h_k_cores     Computed k-core values for each vertex
 * @paramin[]  ref_k_cores   Reference k-core values for each vertex
 * @param[in]  verbose       Whether to output detailed comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(util::Parameters &parameters,
                                        GraphT &graph,
                                        typename GraphT::SizeT *h_k_cores,
                                        typename GraphT::SizeT *ref_k_cores,
                                        bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");

  //Result validation and display
  if(!quick){
    for (SizeT v = 0; v < graph.nodes; ++v) {
      util::PrintMsg(std::to_string(v) + "\t" + 
                     std::to_string(h_k_cores[v]) + "\t" + 
                     std::to_string(ref_k_cores[v]), false);
      if (h_k_cores[v] != ref_k_cores[v]){
          num_errors++;
          util::PrintMsg(std::to_string(v) + "\t" + 
                         std::to_string(h_k_cores[v]) + "\t" + 
                         std::to_string(ref_k_cores[v]), true);
          util::PrintMsg("^^Error here GPU result, CPU result^^", true);
      }
    }

    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", true);
  }
  
  if(quick){
    for (SizeT v = 0; v < graph.nodes; ++v) {
      util::PrintMsg(std::to_string(v) + "\t" + 
                     std::to_string(h_k_cores[v]), !quiet);
    }
  }

  return num_errors;
}

}  // namespace kcore
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
