// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * dc_problem.cuh
 *
 * @brief GPU Storage management Structure for Degree Centrality Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace dc {

/**
 * @brief DC Problem structure stores device-side vectors for doing DC on the GPU.
 *
 * @tparam _VertexId    Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT       Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value       Type of float or double to use for computing degree centrality value.
 */
template <
  typename    _VertexId,
  typename    _SizeT,
  typename    _Value>
struct DCProblem : ProblemBase<_VertexId, _SizeT, false> // USE_DOUBLE_BUFFER = false
{
  typedef _VertexId   VertexId;
  typedef _SizeT      SizeT;
  typedef _Value      Value;
  
  static const bool MARK_PREDECESSORS     = true;
  static const bool ENABLE_IDEMPOTENCE    = false;

  //Helper structures
  
  /**
   * @brief Data slice structure which contains DC problem specific data.
   */
  struct DataSlice
  {

    // device storage arrays
    SizeT       *d_labels;
    VertexId    *d_node_id;         
    Value       *d_degrees_tot;     
    Value       *d_degrees_inv;     

  };
  
  // Members
  
  // Number of GPUs to be sliced over
  int                 num_gpus;
  
  // Size of the graph
  SizeT               nodes;
  SizeT               edges;
  
  // Selector, which d_rank array stores the final page rank?
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
   * @brief DCProblem default constructor
   */
  
  DCProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}
  
  /**
   * @brief DCProblem constructor
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on.
   * @param[in] num_gpus Number of the GPUs used.
   */
  DCProblem(bool        		        stream_from_host,       // Only meaningful for single-GPU
	    const Csr<VertexId, Value, SizeT> 	&graph,
	    const Csr<VertexId, Value, SizeT>   &graph_inv,
	    int         			num_gpus) :
    num_gpus(num_gpus)
  {
    Init(stream_from_host, graph, graph_inv, num_gpus);
  }
  
  /**
   * @brief DCProblem default destructor
   */
  ~DCProblem()
  {
    for (int i = 0; i < num_gpus; ++i)
      {
	if (util::GRError(cudaSetDevice(gpu_idx[i]),
	  "~DCProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
	
	if (data_slices[i]->d_node_id) util::GRError(cudaFree(data_slices[i]->d_node_id),
          "GpuSlice cudaFree d_node_id failed", __FILE__, __LINE__);
	if (data_slices[i]->d_degrees_tot) util::GRError(cudaFree(data_slices[i]->d_degrees_tot), 
          "GpuSlice cudaFree d_degrees_tot failed", __FILE__, __LINE__);
	if (data_slices[i]->d_degrees_inv) util::GRError(cudaFree(data_slices[i]->d_degrees_inv),
	  "GpuSlice cudaFree d_degrees_inv failed", __FILE__, __LINE__);

	if (d_data_slices[i])   util::GRError(cudaFree(d_data_slices[i]), 
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
   * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
   *
   * @param[out] h_rank host-side vector to store page rank values.
   *
   *\return cudaError_t object which indicates the success of all CUDA function calls.
   */
  cudaError_t Extract(VertexId *h_node_id, 
		      Value    *h_degrees, 
		      SizeT    nodes)
  {
    
    cudaError_t retval = cudaSuccess;
    
    do 
      {
	if (num_gpus == 1)
	  {
	    // Set device
	    if (util::GRError(cudaSetDevice(gpu_idx[0]),
			      "DCProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
	    
	    if (retval = util::GRError(cudaMemcpy(h_node_id,
						  data_slices[0]->d_node_id,
						  sizeof(VertexId) * nodes,
						  cudaMemcpyDeviceToHost),
				       "DCProblem cudaMemcpy d_node_id failed", __FILE__, __LINE__)) break;
	    
	    if (retval = util::GRError(cudaMemcpy(h_degrees,
						  data_slices[0]->d_degrees_tot,
						  sizeof(Value) * nodes,
						  cudaMemcpyDeviceToHost),
				       "DCProblem cudaMemcpy d_degrees_tot failed", __FILE__, __LINE__)) break;
	  } else {
	  // TODO: multi-GPU extract result
	} //end if (data_slices.size() ==1)
      } while(0);
    
    return retval;
  }
  
  /**
   * @brief DCProblem initialization
   *
   * @param[in] stream_from_host Whether to stream data from host.
   * @param[in] graph Reference to the CSR graph object we process on. @see Csr
   * @param[in] _num_gpus Number of the GPUs used.
   *
   * \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  cudaError_t Init(bool                              stream_from_host, // Only meaningful for single-GPU
		   const Csr<VertexId, Value, SizeT> &graph_i,
		   const Csr<VertexId, Value, SizeT> &graph_o,
		   int                               _num_gpus)
  {
    // non-inversed graph
    num_gpus = _num_gpus;
    nodes    = graph_i.nodes;
    edges    = graph_i.edges;
    VertexId *h_row_offsets_i = graph_i.row_offsets;
    VertexId *h_col_indices_i = graph_i.column_indices;
    VertexId *h_row_offsets_o = graph_o.row_offsets;
    VertexId *h_col_indices_o = graph_o.column_indices;
    ProblemBase<VertexId, SizeT, false>::Init(stream_from_host,
					      nodes,
					      edges,                
					      h_row_offsets_i,
					      h_col_indices_i,
					      h_row_offsets_o, // d_column_offsets
					      h_col_indices_o, // d_row_indices
					      num_gpus);

    // No data in DataSlice needs to be copied from host
    
    /**
     * Allocate output labels/preds
     */
    cudaError_t retval = cudaSuccess;

    data_slices   = new DataSlice*[num_gpus];
    d_data_slices = new DataSlice*[num_gpus];
    
    do {
      if (num_gpus <= 1) 
	{
	  gpu_idx = (int*)malloc(sizeof(int));
	  // Create a single data slice for the currently-set gpu
	  int gpu;
	  if (retval = util::GRError(cudaGetDevice(&gpu), 
	    "DCProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
	  gpu_idx[0] = gpu;
	  
	  data_slices[0] = new DataSlice;
	  if (retval = util::GRError(cudaMalloc((void**)&d_data_slices[0],
						sizeof(DataSlice)),
				     "DCProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;
	  
	  // Create SoA on device
	  VertexId    *d_node_id;
	  if (retval = util::GRError(cudaMalloc((void**)&d_node_id,
						nodes * sizeof(VertexId)),
				     "DCProblem cudaMalloc d_node_id failed", __FILE__, __LINE__)) return retval;
	  data_slices[0]->d_node_id = d_node_id;
	  
	  Value *d_degrees_tot;
	  if (retval = util::GRError(cudaMalloc((void**)&d_degrees_tot,
						nodes * sizeof(Value)),
				     "DCProblem cudaMalloc d_degrees_tot failed", __FILE__, __LINE__)) return retval;
	  data_slices[0]->d_degrees_tot = d_degrees_tot;				
	  
	  Value *d_degrees_inv;
	  if (retval = util::GRError(cudaMalloc((void**)&d_degrees_inv,
						nodes * sizeof(Value)),
				      "DCProblem cudaMalloc d_degrees_inv failed", __FILE__, __LINE__)) return retval;
	  data_slices[0]->d_degrees_inv = d_degrees_inv;
	  
	  data_slices[0]->d_labels  = NULL;
	  
	}
      //TODO: add multi-GPU allocation code
    } while (0);
    
    return retval;
  }
  
  /**
   *  @brief Performs any initialization work needed for DC problem type. 
   *	Must be called prior to each DC iteration.
   *
   *  @param[in] src Source node for one DC computing pass.
   *  @param[in] frontier_type The f rontier type (i.e., edge/vertex/mixed)
   * 
   *  \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  cudaError_t Reset(FrontierType frontier_type)             // The frontier type (i.e., edge/vertex/mixed)
  {
    typedef ProblemBase<VertexId, SizeT, false> BaseProblem;
    //load ProblemBase Reset
    BaseProblem::Reset(frontier_type, 1.0f); // Default queue sizing is 1.0
    
    cudaError_t retval = cudaSuccess;
    
    for (int gpu = 0; gpu < num_gpus; ++gpu) 
      {
	// Set device
	if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
				   "DCProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
	
	// Allocate output if necessary
	if (!data_slices[gpu]->d_node_id)
	{
	  VertexId    *d_node_id;
	  if (retval = util::GRError(cudaMalloc((void**)&d_node_id,
						nodes * sizeof(VertexId)),
				     "DCProblem cudaMalloc d_node_id failed", __FILE__, __LINE__)) return retval;
	  data_slices[gpu]->d_node_id = d_node_id;
	}
	
	if (!data_slices[gpu]->d_degrees_tot)
	{
	  Value *d_degrees_tot;
	  if (retval = util::GRError(cudaMalloc((void**)&d_degrees_tot,
						nodes * sizeof(Value)),
				     "DCProblem cudaMalloc d_degrees_tot failed", __FILE__, __LINE__)) return retval;
	  data_slices[gpu]->d_degrees_tot = d_degrees_tot;
	}
	
	if (!data_slices[gpu]->d_degrees_inv)
	{
	  Value *d_degrees_inv;
	  if (retval = util::GRError(cudaMalloc((void**)&d_degrees_inv,
						nodes * sizeof(Value)),
				     "DCProblem cudaMalloc d_degrees_inv failed", __FILE__, __LINE__)) return retval;
	  data_slices[gpu]->d_degrees_inv = d_degrees_inv;
	}

	data_slices[gpu]->d_labels = NULL;
	
	if (retval = util::GRError(cudaMemcpy(d_data_slices[gpu],
					      data_slices[gpu],
					      sizeof(DataSlice),
					      cudaMemcpyHostToDevice),
				   "DCProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;
      }
 
    // Fillin the initial input_queue for DC problem, this needs to be modified
    // in multi-GPU scene
        
    // set node ids
    util::MemsetIdxKernel<<<128, 128>>>(data_slices[0]->d_node_id, nodes);
    
    // count number of out-going degrees for each node
    util::MemsetMadVectorKernel<<<128, 128>>>(data_slices[0]->d_degrees_tot,
					      BaseProblem::graph_slices[0]->d_row_offsets, 
					      &BaseProblem::graph_slices[0]->d_row_offsets[1], -1, nodes);

    // count number of in-going degrees for each node
    util::MemsetMadVectorKernel<<<128, 128>>>(data_slices[0]->d_degrees_inv,
					      BaseProblem::graph_slices[0]->d_column_offsets,
					      &BaseProblem::graph_slices[0]->d_column_offsets[1], -1, nodes);
    
    return retval;
  }
  
  /** @} */
  
};
  
  
} //namespace dc
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
