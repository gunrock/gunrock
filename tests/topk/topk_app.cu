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
 * @brief topk app implementation
 */

#include <cstdlib>
#include <stdio.h>
#include <gunrock/gunrock.h>
#include <gunrock/graphio/market.cuh>
#include <gunrock/app/topk/topk_enactor.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::topk;

/**
 * @brief Run TopK
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] node_ids return the top k nodes
 * @param[out] centrality_values return associated centrality
 * @param[in] original  graph to the CSR graph we process on
 * @param[in] reversed  graph to the CSR graph we process on
 * @param[in] top_nodes K value for top K problem
 *
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT>
void topk_run(
//Csr<VertexId, SizeT, Value> &graph_output,
  VertexId *node_ids,
  Value    *centrality_values,
  const Csr<VertexId, Value, SizeT> &graph_original,
  const Csr<VertexId, Value, SizeT> &graph_reversed,
  SizeT top_nodes)
{
  // preparations
  typedef TOPKProblem<VertexId, SizeT, Value> Problem;
  TOPKEnactor<false> topk_enactor(false);
  Problem *topk_problem = new Problem;

  // reset top_nodes if necessary
  top_nodes =
    (top_nodes > graph_original.nodes) ? graph_original.nodes : top_nodes;

  // initialization
  util::GRError(topk_problem->Init(
    false,
    graph_original,
    graph_reversed,
    1),
    "Problem TOPK Initialization Failed", __FILE__, __LINE__);

  // reset data slices
  util::GRError(topk_problem->Reset(
    topk_enactor.GetFrontierType()),
    "TOPK Problem Data Reset Failed", __FILE__, __LINE__);

  // launch topk enactor
  util::GRError(topk_enactor.template Enact<Problem>(
    topk_problem,
    top_nodes),
    "TOPK Problem Enact Failed", __FILE__, __LINE__);

  // copy out results back to cpu
  util::GRError(topk_problem->Extract(
    node_ids,
    centrality_values,
    top_nodes), "TOPK Problem Data Extraction Failed", __FILE__, __LINE__);

  // cleanup if neccessary
  if (topk_problem) { delete topk_problem; }

  cudaDeviceSynchronize();
}

void topk_dispatch(
  GunrockGraph       *graph_out,
  void               *node_ids,
  void               *centrality_values,
  const GunrockGraph *graph_in,
  size_t             top_nodes,
  GunrockDataType    data_type)
{
  //TODO: add more supportive datatypes if necessary

  switch (data_type.VTXID_TYPE)
    {
    case VTXID_INT:
      switch(data_type.SIZET_TYPE)
	{
	case SIZET_UINT:
	  switch (data_type.VALUE_TYPE)
	    {
	    case VALUE_INT:
	      {
		// case that VertexId, SizeT, Value are all have the type int

		// original graph
		Csr<int, int, int> graph_original(false);
		graph_original.nodes = graph_in->num_nodes;
		graph_original.edges = graph_in->num_edges;
		graph_original.row_offsets    = (int*)graph_in->row_offsets;
		graph_original.column_indices = (int*)graph_in->col_indices;

		// reversed graph
		Csr<int, int, int> graph_reversed(false);
		graph_reversed.nodes = graph_in->num_nodes;
		graph_reversed.edges = graph_in->num_edges;
		graph_reversed.row_offsets    = (int*)graph_in->col_offsets;
		graph_reversed.column_indices = (int*)graph_in->row_indices;

		topk_run<int, int, int>((int*)node_ids,
					(int*)centrality_values,
					graph_original,
					graph_reversed,
					top_nodes);

		// reset for free memory
		graph_original.row_offsets    = NULL;
		graph_original.column_indices = NULL;
		graph_reversed.row_offsets    = NULL;
		graph_reversed.column_indices = NULL;

		break;
	      }
	    case VALUE_FLOAT:
	      {
		// case that VertexId and SizeT have type int, Value is float
		/*
		// original graph
		Csr<int, float, int> graph_original(false);
		graph_original.nodes = graph_in->num_nodes;
		graph_original.edges = graph_in->num_edges;
		graph_original.row_offsets    = (int*)graph_in->row_offsets;
		graph_original.column_indices = (int*)graph_in->col_indices;

		// reversed graph
		Csr<int, float, int> graph_reversed(false);
		graph_reversed.nodes = graph_in->num_nodes;
		graph_reversed.edges = graph_in->num_edges;
		graph_reversed.row_offsets    = (int*)graph_in->col_offsets;
		graph_reversed.column_indices = (int*)graph_in->row_indices;

		topk_run<int, float, int>((int*)node_ids,
					  (float*)centrality_values,
					  graph_original,
					  graph_reversed,
					  top_nodes);

		// reset for free memory
		graph_original.row_offsets    = NULL;
		graph_original.column_indices = NULL;
		graph_reversed.row_offsets    = NULL;
		graph_reversed.column_indices = NULL;
		*/
		break;
	      }
	    }
	  break;
	}
      break;
    }
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
