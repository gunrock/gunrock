// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mst_problem.cuh
 *
 * @brief GPU Storage management Structure for MST Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace mst {

/**
 * @brief MST Problem structure stores device-side vectors for
 * doing Minimum Spanning Tree on the GPU.
 *
 * @tparam _VertexId
 * Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT
 * Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value
 * Type of float or double to use for computing MST value.
 * @tparam _USE_DOUBLE_BUFFER
 * Boolean type parameter which defines whether to use double buffer.
 *
 */
template <
  typename _VertexId,
  typename _SizeT,
  typename _Value,
  bool     _USE_DOUBLE_BUFFER>
struct MSTProblem : ProblemBase<_VertexId, _SizeT, _USE_DOUBLE_BUFFER>
{
  typedef _VertexId	VertexId;
  typedef _SizeT		SizeT;
  typedef _Value		Value;

  static const bool MARK_PREDECESSORS	 = true;
  static const bool ENABLE_IDEMPOTENCE = false;

  //Helper structures

  /**
   * @brief Data slice structure which contains MST problem specific data.
   */
  struct DataSlice
  {
  	SizeT    *d_labels;

    // device storage arrays
    int 		 *d_flag_array;	 	//!< flag (1 indicate start of segment, 0 otherwise)
    SizeT		 *d_keys_array; 	//!< keys array (inclusive scan of the flag array)
 		Value		 *d_reduced_vals; //!< store reduced minimum edge values (weights)
    SizeT 	 *d_reduced_keys; //!< reduced keys array
    VertexId *d_successors;	 	//!< destination vertices that have min edge_values
    int			 *d_mst_output; 	//!< mark selected edges (1 indicate selected edges)
    Value 	 *d_edge_weights;	//!< store values per edge (a.k.a. edge weights)
    Value    *d_temp_storage;	//!< used for storing temporary arrays
    int			 *d_vertex_flag;  //!< finish flag for per-vertex kernels in algorithm
    int      *d_super_flag;		//!< used for mark the boundaries of representatives
    VertexId *d_origin_nodes;	//!< used for keeping track of origin vertex ids
    VertexId *d_origin_edges;	//!< origin edge list keep track of edge ids
    SizeT		 *d_super_vertex;	//!< super vertex ids scaned from super flag
    VertexId *d_super_edges;	//!< super edge list for next iteration
    VertexId *d_edgeId_list;  //!< storing inital column indices

    SizeT	*d_edgeFlag;        /* Used for removing edges between supervertices */
    SizeT	*d_edgeKeys;        /* Used for removing edges between supervertices */
    Value	*d_oriWeights;      /* Original weight list used for total weight calculate */ // Do we need that?
    VertexId	*d_superVertex;         /* Used for storing supervertex in order */
    SizeT	*d_row_offsets;		/* Used for accessing row_offsets */    // Try to use graph_slice
    VertexId	*d_edgeId;		/* Used for storing vid of edges have min_weights */

    VertexId *d_representatives; //!< representative vertices for each successors // TODO
    VertexId	*d_edge_offsets;    /* Used for removing edges between supervertices */


  };

  // Members

  // Number of GPUs to be sliced over
  int                 num_gpus;

  // Size of the graph
  SizeT               nodes;
  SizeT               edges;

  // Selector
  SizeT               selector;

  // Set of data slices (one for each GPU)
  DataSlice           **data_slices;

  // Nasty method for putting struct on device
  // while keeping the SoA structure
  DataSlice           **d_data_slices;

  // Device indices for each data slice
  int                 *gpu_idx;

  // Methods

  /**
   * @brief MSTProblem default constructor
   */

  MSTProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

  /**
   * @brief MSTProblem constructor
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on.
   * @param[in] num_gpus Number of the GPUs used.
   */
  MSTProblem(
		bool  stream_from_host,
	  const Csr<VertexId, Value, SizeT> &graph,
	  int   num_gpus) :
    num_gpus(num_gpus)
  {
    Init(stream_from_host,
				 graph,
				 num_gpus);
  }

  /**
   * @brief MSTProblem default destructor
   */
  ~MSTProblem()
  {
    for (int i = 0; i < num_gpus; ++i)
    {
			if (util::GRError(cudaSetDevice(gpu_idx[i]),
				"~MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

			if (data_slices[i]->d_edgeId_list)
			  util::GRError(cudaFree(data_slices[i]->d_edgeId_list),
					"GpuSlice cudaFree d_edgeId_list failed", __FILE__, __LINE__);
			if (data_slices[i]->d_edge_weights)
			  util::GRError(cudaFree(data_slices[i]->d_edge_weights),
				  "GpuSlice cudaFree d_edge_weights failed", __FILE__, __LINE__);
			if (data_slices[i]->d_oriWeights)
			  util::GRError(cudaFree(data_slices[i]->d_oriWeights),
					"GpuSlice cudaFree d_oriWeights failed", __FILE__, __LINE__);
		  if (data_slices[i]->d_reduced_vals)
			  util::GRError(cudaFree(data_slices[i]->d_reduced_vals),
					"GpuSlice cudaFree d_reduced_vals failed", __FILE__, __LINE__);
			if (data_slices[i]->d_flag_array)
			  util::GRError(cudaFree(data_slices[i]->d_flag_array),
					"GpuSlice cudaFree d_flag_array failed", __FILE__, __LINE__);
			if (data_slices[i]->d_keys_array)
			  util::GRError(cudaFree(data_slices[i]->d_keys_array),
					"GpuSlice cudaFree d_keys_array failed", __FILE__, __LINE__);
			if (data_slices[i]->d_temp_storage)
			  util::GRError(cudaFree(data_slices[i]->d_temp_storage),
					"GpuSlice cudaFree d_temp_storage failed", __FILE__, __LINE__);
			if (data_slices[i]->d_reduced_keys)
			  util::GRError(cudaFree(data_slices[i]->d_reduced_keys),
					"GpuSlice cudaFree d_reduced_keys failed", __FILE__, __LINE__);
			if (data_slices[i]->d_successors)
			  util::GRError(cudaFree(data_slices[i]->d_successors),
					"GpuSlice cudaFree d_successors failed", __FILE__, __LINE__);
			if (data_slices[i]->d_representatives)
			  util::GRError(cudaFree(data_slices[i]->d_representatives),
					"GpuSlice cudaFree d_representatives failed", __FILE__, __LINE__);
			if (data_slices[i]->d_superVertex)
			  util::GRError(cudaFree(data_slices[i]->d_superVertex),
					"GpuSlice cudaFree d_superVertex failed", __FILE__, __LINE__);
			if (data_slices[i]->d_row_offsets)
			  util::GRError(cudaFree(data_slices[i]->d_row_offsets),
					"GpuSlice cudaFree d_row_offsets failed", __FILE__, __LINE__);
			if (data_slices[i]->d_edgeId)
			  util::GRError(cudaFree(data_slices[i]->d_edgeId),
					"GpuSlice cudaFree d_edgeId failed", __FILE__, __LINE__);
			if (data_slices[i]->d_vertex_flag)
			  util::GRError(cudaFree(data_slices[i]->d_vertex_flag),
					"GpuSlice cudaFree d_vertex_flag failed", __FILE__, __LINE__);
			if (data_slices[i]->d_origin_nodes)
			  util::GRError(cudaFree(data_slices[i]->d_origin_nodes),
					"GpuSlice cudaFree d_origin_nodes failed", __FILE__, __LINE__);
			if (data_slices[i]->d_super_flag)
			  util::GRError(cudaFree(data_slices[i]->d_super_flag),
					"GpuSlice cudaFree d_super_flag failed", __FILE__, __LINE__);
			if (data_slices[i]->d_super_vertex)
			  util::GRError(cudaFree(data_slices[i]->d_super_vertex),
					"GpuSlice cudaFree d_super_vertex failed", __FILE__, __LINE__);
			if (data_slices[i]->d_origin_edges)
			  util::GRError(cudaFree(data_slices[i]->d_origin_edges),
					"GpuSlice cudaFree d_origin_edges failed", __FILE__, __LINE__);
			if (data_slices[i]->d_mst_output)
			  util::GRError(cudaFree(data_slices[i]->d_mst_output),
					"GpuSlice cudaFree d_mst_output failed", __FILE__, __LINE__);
			if (data_slices[i]->d_edge_offsets)
			  util::GRError(cudaFree(data_slices[i]->d_edge_offsets),
					"GpuSlice cudaFree d_edge_offsets failed", __FILE__, __LINE__);
			if (data_slices[i]->d_edgeFlag)
			  util::GRError(cudaFree(data_slices[i]->d_edgeFlag),
					"GpuSlice cudaFree d_edgeFlag failed", __FILE__, __LINE__);
			if (data_slices[i]->d_edgeKeys)
			  util::GRError(cudaFree(data_slices[i]->d_edgeKeys),
					"GpuSlice cudaFree d_edgeKeys failed", __FILE__, __LINE__);
			if (data_slices[i]->d_super_edges)
			  util::GRError(cudaFree(data_slices[i]->d_super_edges),
					"GpuSlice cudaFree d_super_edges failed", __FILE__, __LINE__);

			if (d_data_slices[i])
			  util::GRError(cudaFree(d_data_slices[i]),
					"GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
      }

    if (d_data_slices) delete[] d_data_slices;
    if (data_slices)   delete[] data_slices;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Copy result labels and/or predecessors computed
   * on the GPU back to host-side vectors.
   *
   * @param[out] h_selector host-side vector to store mst results.
   *
   *\return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  //TODO: write extract function
  cudaError_t Extract(SizeT *h_selector)
  {
    cudaError_t retval = cudaSuccess;
    do
    {
      if (num_gpus == 1)
      {
				// Set device
				if (util::GRError(cudaSetDevice(gpu_idx[0]),
				  "MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

				if (retval = util::GRError(cudaMemcpy(
					h_selector,
					data_slices[0]->d_mst_output,
					sizeof(SizeT) * edges,
					cudaMemcpyDeviceToHost),
					"MSTProblem cudaMemcpy selector failed", __FILE__, __LINE__)) break;
      }
			else
			{
				// TODO: multi-GPU extract result
      } //end if (data_slices.size() ==1)
    } while(0);
    return cudaSuccess;
  }

  /**
   * @brief MSTProblem initialization
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on. @see Csr
   * @param[in] _num_gpus Number of the GPUs used.
   *
   * \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  cudaError_t Init(
		bool  stream_from_host,
		const Csr<VertexId, Value, SizeT> &graph,
		int   _num_gpus)
  {
    num_gpus = _num_gpus;
    nodes = graph.nodes;
    edges = graph.edges;
    VertexId *h_row_offsets = graph.row_offsets;
    VertexId *h_col_indices = graph.column_indices;
    ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER>::Init(
			stream_from_host,
			nodes,
			edges,
			h_row_offsets,
			h_col_indices,
			NULL,
			NULL,
			num_gpus);

    // No data in DataSlice needs to be copied from host

    /**
     * Allocate output labels/preds
     */
    cudaError_t retval = cudaSuccess;
    data_slices   = new DataSlice*[num_gpus];
    d_data_slices = new DataSlice*[num_gpus];

    do
    {
      if (num_gpus <= 1)
      {
			  gpu_idx = (int*)malloc(sizeof(int));
			  // Create a single data slice for the currently-set gpu
			  int gpu;
			  if (retval = util::GRError(cudaGetDevice(&gpu),
					"MSTProblem cudaGetDevice failed", 
					__FILE__, __LINE__)) break;
			  gpu_idx[0] = gpu;

			  data_slices[0] = new DataSlice;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_data_slices[0],
					sizeof(DataSlice)),
					"MSTProblem cudaMalloc d_data_slices failed",
					__FILE__, __LINE__)) return retval;

			  // Create SoA on device
			  VertexId	*d_edgeId_list;
			  if (retval = util::GRError(cudaMalloc(
				(void**)&d_edgeId_list,
					edges * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_edgeId_list failed",
					__FILE__, __LINE__)) return retval;
			  if (retval = util::GRError(cudaMemcpy(
					d_edgeId_list,
					graph.column_indices,
					edges * sizeof(VertexId),
					cudaMemcpyHostToDevice),
					"ProblemBase cudaMemcpy d_edgeId_list failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_edgeId_list = d_edgeId_list;

				VertexId *d_super_edges;
			  if (retval = util::GRError(cudaMalloc(
				(void**)&d_super_edges,
					edges * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_super_edges failed",
					__FILE__, __LINE__)) return retval;
			  if (retval = util::GRError(cudaMemcpy(
					d_super_edges,
					graph.column_indices,
					edges * sizeof(VertexId),
					cudaMemcpyHostToDevice),
					"ProblemBase cudaMemcpy d_super_edges failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_super_edges = d_super_edges;

			  Value	*d_edge_weights;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_edge_weights,
					edges * sizeof(Value)),
					"MSTProblem cudaMalloc d_edge_weights failed",
					__FILE__, __LINE__)) return retval;
			  if (retval = util::GRError(cudaMemcpy(
					d_edge_weights,
					graph.edge_values,
					edges * sizeof(Value),
					cudaMemcpyHostToDevice),
					"ProblemBase cudaMemcpy d_edge_weights failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_edge_weights = d_edge_weights;

			  Value   *d_oriWeights;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_oriWeights,
					edges * sizeof(Value)),
					"MSTProblem cudaMalloc d_oriWeights failed",
					__FILE__, __LINE__)) return retval;
			  if (retval = util::GRError(cudaMemcpy(
					d_oriWeights,
					graph.edge_values,
					edges * sizeof(Value),
					cudaMemcpyHostToDevice),
					"ProblemBase cudaMemcpy d_oriWeights failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_oriWeights = d_oriWeights;

			  Value	*d_reduced_vals;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_reduced_vals,
					nodes * sizeof(Value)),
					"MSTProblem cudaMalloc d_reduced_vals failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_reduced_vals = d_reduced_vals;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_reduced_vals, 0, nodes);

			  int	*d_flag_array;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_flag_array,
					edges * sizeof(int)),
					"MSTProblem cudaMalloc d_flag_array Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_flag_array = d_flag_array;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_flag_array, 0, edges);

			  SizeT *d_keys_array;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_keys_array,
					edges * sizeof(SizeT)),
					"MSTProblem cudaMalloc d_keys_array Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_keys_array = d_keys_array;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_keys_array, 0, edges);

			  SizeT   *d_temp_storage;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_temp_storage,
					edges * sizeof(SizeT)),
					"MSTProblem cudaMalloc d_temp_storage Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_temp_storage = d_temp_storage;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_temp_storage, 0, edges);

			  VertexId	*d_reduced_keys;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_reduced_keys,
					nodes * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_reduced_keys Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_reduced_keys = d_reduced_keys;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_reduced_keys, 0, nodes);

			  VertexId	*d_successors;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_successors,
					nodes * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_successors Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_successors = d_successors;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_successors, 0, nodes);

			  VertexId *d_representatives;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_representatives,
					nodes * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_representatives Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_representatives = d_representatives;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_representatives, 0, nodes);

			  VertexId *d_superVertex;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_superVertex,
					nodes * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_superVertex Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_superVertex = d_superVertex;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_superVertex, 0, nodes);

			  VertexId        *d_row_offsets;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_row_offsets,
					nodes * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_row_offsets Failed",
					__FILE__, __LINE__)) return retval;
			  if (retval = util::GRError(cudaMemcpy(
					d_row_offsets,
					graph.row_offsets,
					nodes * sizeof(VertexId),
					cudaMemcpyHostToDevice),
					"ProblemBase cudaMemcpy d_row_offsets failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_row_offsets = d_row_offsets;

			  VertexId        *d_edgeId;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_edgeId,
					nodes * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_edgeId Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_edgeId = d_edgeId;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_edgeId, 0, nodes);

			  int *d_vertex_flag;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_vertex_flag,
					sizeof(int)),
					"MSTProblem cudaMalloc d_vertex_flag failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_vertex_flag = d_vertex_flag;

			  VertexId	*d_origin_nodes;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_origin_nodes,
					nodes * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_origin_nodes failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_origin_nodes = d_origin_nodes;
			  util::MemsetIdxKernel<<<128, 128>>>(
          data_slices[0]->d_origin_nodes, nodes);

			  int *d_super_flag;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_super_flag,
					nodes * sizeof(int)),
					"MSTProblem cudaMalloc d_super_flag Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_super_flag = d_super_flag;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_super_flag, 0, nodes);

			  SizeT *d_super_vertex;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_super_vertex,
					nodes * sizeof(SizeT)),
					"MSTProblem cudaMalloc d_super_vertex Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_super_vertex = d_super_vertex;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_super_vertex, 0, nodes);

			  VertexId *d_origin_edges;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_origin_edges,
					edges * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_origin_edges Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_origin_edges = d_origin_edges;
			  util::MemsetIdxKernel<<<128, 128>>>(
			  	data_slices[0]->d_origin_edges, edges);

			  int *d_mst_output;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_mst_output,
					edges * sizeof(SizeT)),
					"MSTProblem cudaMalloc d_mst_output Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_mst_output = d_mst_output;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_mst_output, 0, edges);

			  VertexId *d_edge_offsets;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_edge_offsets,
					edges * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_edge_offsets failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_edge_offsets = d_edge_offsets;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_edge_offsets, 0, edges);

			  SizeT *d_edgeFlag;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_edgeFlag,
					edges * sizeof(SizeT)),
					"MSTProblem cudaMalloc d_edgeFlag Failed",
					__FILE__, __LINE__)) return retval;
			  data_slices[0]->d_edgeFlag = d_edgeFlag;
			  util::MemsetKernel<<<128, 128>>>(
			  	data_slices[0]->d_edgeFlag, 0, edges);

			  SizeT *d_edgeKeys;
			  if (retval = util::GRError(cudaMalloc(
					(void**)&d_edgeKeys,
					edges * sizeof(SizeT)),
					"MSTProblem cudaMalloc d_edgeKeys Failed",
					__FILE__, __LINE__)) return retval;
				data_slices[0]->d_edgeKeys = d_edgeKeys;
				util::MemsetKernel<<<128, 128>>>(
					data_slices[0]->d_edgeKeys, 0, edges);

				data_slices[0]->d_labels = NULL;

			}
      //TODO: add multi-GPU allocation code
    } while (0);

    return retval;
  }

  /**
   * @brief Performs any initialization work needed for MST problem type.
   * Must be called prior to each MST iteration.
   *
   * @param[in] src Source node for one MST computing pass.
   * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
   *
   *  \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  cudaError_t Reset(FrontierType frontier_type)
  {
    typedef ProblemBase<
    	VertexId, SizeT,
    	_USE_DOUBLE_BUFFER> BaseProblem;

    //load ProblemBase Reset
    BaseProblem::Reset(frontier_type, 1.0f); // Default queue sizing is 1.0

    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < num_gpus; ++gpu)
    {
      // Set device
      if (retval = util::GRError(
      	cudaSetDevice(gpu_idx[gpu]),
				"MSTProblem cudaSetDevice failed",
				__FILE__, __LINE__)) return retval;

      // Allocate output if necessary
      if (!data_slices[gpu]->d_edgeId_list)
      {
				VertexId *d_edgeId_list;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_edgeId_list,
					edges * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_edgeId_list failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_edgeId_list = d_edgeId_list;
			}

			if (!data_slices[gpu]->d_super_edges)
      {
				VertexId *d_super_edges;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_super_edges,
					edges * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_super_edges failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_super_edges = d_super_edges;
			}

      if (!data_slices[gpu]->d_edge_weights)
      {
				Value *d_edge_weights;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_edge_weights,
					edges * sizeof(Value)),
				  "MSTProblem cudaMalloc d_edge_weights failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_edge_weights = d_edge_weights;
			}

      if (!data_slices[gpu]->d_oriWeights)
      {
				Value    *d_oriWeights;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_oriWeights,
					edges * sizeof(Value)),
				  "MSTProblem cudaMalloc d_oriWeights failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_oriWeights = d_oriWeights;
      }

      if(!data_slices[gpu]->d_reduced_vals)
      {
				Value *d_reduced_vals;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_reduced_vals,
					nodes * sizeof(Value)),
				  "MSTProblem cudaMalloc d_reduced_vals failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_reduced_vals = d_reduced_vals;
      }

      if (!data_slices[gpu]->d_flag_array)
      {
				int	*d_flag_array;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_flag_array,
					edges * sizeof(int)),
				  "MSTProblem cudaMalloc d_flag_array Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_flag_array = d_flag_array;
			}

      if (!data_slices[gpu]->d_keys_array)
      {
				int *d_keys_array;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_keys_array,
					edges * sizeof(int)),
				  "MSTProblem cudaMalloc d_keys_array Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_keys_array = d_keys_array;
      }

      if (!data_slices[gpu]->d_temp_storage)
      {
				SizeT *d_temp_storage;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_temp_storage,
					edges * sizeof(SizeT)),
				  "MSTProblem cudaMalloc d_temp_storage Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_temp_storage = d_temp_storage;
      }

      if (!data_slices[gpu]->d_successors)
      {
				VertexId *d_successors;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_successors,
					nodes * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_successors Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_successors = d_successors;
			}

      if (!data_slices[gpu]->d_representatives)
      {
				VertexId *d_representatives;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_representatives,
					nodes * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_representatives Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_representatives = d_representatives;
      }

      if (!data_slices[gpu]->d_superVertex)
      {
				VertexId        *d_superVertex;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_superVertex,
					nodes * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_superVertex Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_superVertex = d_superVertex;
      }

      if (!data_slices[gpu]->d_reduced_keys)
      {
				VertexId *d_reduced_keys;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_reduced_keys,
					nodes * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_reduced_keys Failed",
				  __FILE__, __LINE__)) return retval;
		    data_slices[gpu]->d_reduced_keys = d_reduced_keys;
      }

      if (!data_slices[gpu]->d_row_offsets)
      {
				SizeT *d_row_offsets;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_row_offsets,
					nodes * sizeof(SizeT)),
				  "MSTProblem cudaMalloc d_row_offsets Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_row_offsets = d_row_offsets;
      }

      if (!data_slices[gpu]->d_edgeId)
      {
				VertexId *d_edgeId;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_edgeId,
					nodes * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_edgeId Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_edgeId = d_edgeId;
      }

      int *vertex_flag = new int;
      // Allocate vertex_flag if necessary
      if (!data_slices[gpu]->d_vertex_flag)
      {
				int *d_vertex_flag;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_vertex_flag,
					sizeof(int)),
				  "MSTProblem cudaMalloc d_vertex_flag failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_vertex_flag = d_vertex_flag;
      }
      vertex_flag[0] = 1;
      if (retval = util::GRError(cudaMemcpy(
			  data_slices[gpu]->d_vertex_flag,
			  vertex_flag,
			  sizeof(int),
			  cudaMemcpyHostToDevice),
				"MSTProblem cudaMemcpy vertex_flag to d_vertex_flag failed",
				__FILE__, __LINE__)) return retval;
      delete vertex_flag;

      if (!data_slices[gpu]->d_origin_nodes)
      {
				VertexId *d_origin_nodes;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_origin_nodes,
					nodes * sizeof(VertexId)),
					"MSTProblem cudaMalloc d_origin_nodes Failed",
					__FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_origin_nodes = d_origin_nodes;
			}

      if (!data_slices[gpu]->d_super_flag)
			{
				int *d_super_flag;
				if(retval = util::GRError(cudaMalloc(
				  (void**)&d_super_flag,
					nodes * sizeof(int)),
				  "MSTProblem cudaMalloc d_super_flag Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_super_flag = d_super_flag;
			}

      if (!data_slices[gpu]->d_super_vertex)
			{
				VertexId *d_super_vertex;
				if(retval = util::GRError(cudaMalloc(
				  (void**)&d_super_vertex,
					nodes * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_super_vertex Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_super_vertex = d_super_vertex;
			}

      if (!data_slices[gpu]->d_origin_edges)
			{
				VertexId *d_origin_edges;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_origin_edges,
					edges * sizeof(VertexId)),
				  "MSTProblem cudaMalloc d_origin_edges Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_origin_edges = d_origin_edges;
			}

      if (!data_slices[gpu]->d_mst_output)
			{
				int *d_mst_output;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_mst_output,
					edges * sizeof(int)),
				  "MSTProblem cudaMalloc d_mst_output Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_mst_output = d_mst_output;
			}

      if (!data_slices[gpu]->d_edge_offsets)
			{
				SizeT *d_edge_offsets;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_edge_offsets,
					edges * sizeof(SizeT)),
				  "MSTProblem cudaMalloc d_edge_offsets Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_edge_offsets = d_edge_offsets;
			}

      if (!data_slices[gpu]->d_edgeFlag)
			{
				SizeT *d_edgeFlag;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_edgeFlag,
					edges * sizeof(SizeT)),
				  "MSTProblem cudaMalloc d_edgeFlag Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_edgeFlag = d_edgeFlag;
			}

      if (!data_slices[gpu]->d_edgeKeys)
			{
				SizeT   *d_edgeKeys;
				if (retval = util::GRError(cudaMalloc(
					(void**)&d_edgeKeys,
					edges * sizeof(SizeT)),
				  "MSTProblem cudaMalloc d_edgeKeys Failed",
				  __FILE__, __LINE__)) return retval;
				data_slices[gpu]->d_edgeKeys = d_edgeKeys;
      }

      data_slices[0]->d_labels = NULL;

      if (retval = util::GRError(cudaMemcpy(
			  d_data_slices[gpu],
			  data_slices[gpu],
			  sizeof(DataSlice),
			  cudaMemcpyHostToDevice),
				"MSTProblem cudaMemcpy data_slices to d_data_slices failed",
				__FILE__, __LINE__)) return retval;

    }

    // Fillin the initial input_queue for MST problem
    // in multi-GPU scene

    // Put every vertex in frontier queue
    util::MemsetIdxKernel<<<128, 128>>>(
			BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);

	  util::MemsetIdxKernel<<<128, 128>>>(
    	BaseProblem::graph_slices[0]->frontier_queues.d_values[0], edges);

    return retval;
  }

  /** @} */

};

} //namespace mst
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: