// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * topk_app.cu
 *
 * @brief Simple test driver program for computing Top K
 */

#include <cstdlib>
#include <stdio.h> 
#include <gunrock/app/topk/topk_enactor.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

using namespace gunrock::app::topk;

/**
 * @brief displays the top K results
 *
 */
template<typename VertexId, typename Value, typename SizeT>
void DisplaySolution(VertexId *h_node_id,
		     Value    *h_degrees,
		     SizeT    num_nodes)
{
  // at most display first 100 results
  if (num_nodes > 100) { num_nodes = 100; }
  printf("==> Top %d centrality nodes:\n", num_nodes);
  for (SizeT i = 0; i < num_nodes; ++i)
  { printf("%d %d\n", h_node_id[i], h_degrees[i]); }
  printf("\n");
  fflush(stdout);
}

/**
 * @brief Run TopK
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 *
 */
template <typename VertexId, typename SizeT, typename Value>
void topk_run(const SizeT    row_offsets_i,
	      const VertexId col_indices_i,
	      const SizeT    row_offsets_j,
	      const VertexId col_indices_j,
	      const SizeT    num_nodes,
	      const SizeT    num_edges,
	      const SizeT    top_nodes,
	      const int      data_type)
{
  
  // define the problem data structure for graph primitive
  typedef TOPKProblem<VertexId, SizeT, Value> Problem;
  
  // INSTRUMENT specifies whether we want to keep such statistical data
  // Allocate TopK enactor map 
  TOPKEnactor<INSTRUMENT> topk_enactor(g_verbose);
  
  // allocate problem on GPU, create a pointer of the TOPKProblem type 
  Problem *topk_problem = new Problem;
  
  // reset top_nodes if input k > total number of nodes
  if (top_nodes > num_nodes) { top_nodes = num_nodes; }
  
  // malloc host memory
  VertexId *h_node_id = (VertexId*)malloc(sizeof(VertexId) * top_nodes);
  Value    *h_degrees = ( Value* )malloc(sizeof( Value ) * top_nodes);
  
  // TODO: build graph_i and graph_j from csr arrays
  

  // copy data from CPU to GPU, initialize data members in DataSlice for graph
  util::GRError(topk_problem->Init(g_stream_from_host,
				   graph_i,
				   graph_j,
				   num_gpus), 
		"Problem TOPK Initialization Failed", __FILE__, __LINE__);
  
  // reset values in DataSlice for graph
  util::GRError(topk_problem->Reset(topk_enactor.GetFrontierType()), 
		"TOPK Problem Data Reset Failed", __FILE__, __LINE__);
  
  // launch topk enactor
  util::GRError(topk_enactor.template Enact<Problem>(context, 
						     topk_problem, 
						     top_nodes, 
						     max_grid_size), 
		"TOPK Problem Enact Failed", __FILE__, __LINE__);
  
  // copy out results back to CPU from GPU using Extract
  util::GRError(topk_problem->Extract(h_node_id,
				      h_degrees,
				      top_nodes),
		"TOPK Problem Data Extraction Failed", __FILE__, __LINE__);
  
  // display solution
  DisplaySolution(h_node_id, h_degrees, top_nodes);
  
  // cleanup if neccessary
  if (topk_problem) { delete topk_problem; }
  if (h_node_id)    { free(h_node_id); }
  if (h_degrees)    { free(h_degrees); }
  
  cudaDeviceSynchronize();
}

void topk_dispatch(const void *row_offsets_i,
		   const void *col_indices_i,
		   const void *row_offsets_j,
		   const void *col_indices_j,
		   size_t     num_nodes,
		   size_t     num_edges,
		   size_t     top_nodes,
		   const int  data_type);
{
  //TODO: add more supportive if necessary
  switch (VTXID_TYPE)
  {
  case VTXID_UINT:
    switch (VALUE_TYPE)
    {
    case VALUE_UINT:
      topk_run<unsigned int, unsigned int, unsigned int>
	((const unsigned int*)row_offsets_i,
	 (const unsigned int*)col_indices_i,
	 (const unsigned int*)row_offsets_j,
	 (const unsigned int*)col_indices_j,
	 num_nodes,  num_edges,  top_nodes); 
      break;
      
    case VALUE_FLOAT:
      topk_run<unsigned int, unsigned int, float>
	((const unsigned int*)row_offsets_i,
	 (const unsigned int*)col_indices_i,
	 (const unsigned int*)row_offsets_j,
	 (const unsigned int*)col_indices_j,
	 num_nodes,  num_edges,  top_nodes); 
      break;
      
    case VALUE_DOUBLE:
      topk_run<unsigned int, unsigned int, double>
	((const unsigned int*)row_offsets_i,
	 (const unsigned int*)col_indices_i,
	 (const unsigned int*)row_offsets_j,
	 (const unsigned int*)col_indices_j,
	 num_nodes,  num_edges,  top_nodes);

    }
  case VTXID_LONG:
    switch (VALUE_TYPE)
    {
    case VALUE_UINT:
      topk_run<long int, long int, unsigned int>
	((const unsigned int*)row_offsets_i,
	 (const unsigned int*)col_indices_i,
	 (const unsigned int*)row_offsets_j,
	 (const unsigned int*)col_indices_j,
	 num_nodes,  num_edges,  top_nodes); 
      break;
   
    case VALUE_FLOAT:
      topk_run<long int, long int, float>
	((const unsigned int*)row_offsets_i,
	 (const unsigned int*)col_indices_i,
	 (const unsigned int*)row_offsets_j,
	 (const unsigned int*)col_indices_j,
	 num_nodes,  num_edges,  top_nodes); 
      break;
    
    case VALUE_DOUBLE:
      topk_run<long int, long int, float>
	((const unsigned int*)row_offsets_i,
	 (const unsigned int*)col_indices_i,
	 (const unsigned int*)row_offsets_j,
	 (const unsigned int*)col_indices_j,
	 num_nodes,  num_edges,  top_nodes);
    }
  }
}

/* end */
