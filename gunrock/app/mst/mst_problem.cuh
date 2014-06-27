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
 * @brief GPU Storage management Structure for MST Problem Specific Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace mst {

    /**
     * @brief Minimum Spanning Tree Problem structure stores device-side
     * vectors for doing Miminum Spanning Tree calculations on the GPU.
     *
     * @tparam _VertexId          Type of signed integer used as vertex id
     * @tparam _SizeT             Type of unsigned integer for array index
     * @tparam _Value             Type of float or double  for MST values
     * @tparam _USE_DOUBLE_BUFFER Defines whether to enable double buffer
     *
     */
    template <
	typename _VertexId,
	typename _SizeT,
	typename _Value,
	bool     _USE_DOUBLE_BUFFER>
    struct MSTProblem : ProblemBase<_VertexId, _SizeT, _USE_DOUBLE_BUFFER>
    {
	typedef _VertexIdVertexId;
	typedef _SizeTSizeT;
	typedef _ValueValue;

	static const bool MARK_PREDECESSORS= true;
	static const bool ENABLE_IDEMPOTENCE= false;

	/**
	 * @brief Data slice structure which contains MST problem specific data.
	 */
	struct DataSlice
	{

	};
	// Members

	// Number of GPUs to be sliced over
	int                 num_gpus;

	// Size of the graph
	SizeT               nodes;
	SizeT               edges;
	unsigned int        num_components;

	// Set of data slices (one for each GPU)
	DataSlice           **data_slices;

	// Nasty method for putting struct on device
	// while keeping the SoA structure
	DataSlice           **d_data_slices;

	// Device indices for each data slice
	int                 *gpu_idx;

	// Methods

	/**
	 * @brief CCProblem default constructor
	 */

	MSTProblem():
	    nodes(0),
	    edges(0),
	    num_gpus(0),
	    num_components(0) {}

	/**
	 * @brief MSTProblem constructor
	 *
	 * @param[in] stream_from_host Whether to stream data from host.
	 * @param[in] graph Reference to the CSR graph object we process on.
	 * @param[in] num_gpus Number of the GPUs used.
	 */
	MSTProblem(
	    bool      stream_from_host, // Only meaningful for single-GPU
	    const Csr<VertexId, Value, SizeT> &graph,
	    int         num_gpus) :
	    num_gpus(num_gpus)
	    {
		Init(stream_from_host,
		     graph,
		     num_gpus);
	    }

	/**
	 * @brief CCProblem default destructor
	 */
	~MSTProblem()
	{
	    for (int i = 0; i < num_gpus; ++i)
	    {
		if (util::GRError(cudaSetDevice(gpu_idx[i]),
				  "~MSTProblem cudaSetDevice failed",
				  __FILE__, __LINE__)) break;

		if (d_data_slices[i])
		{
		    util::GRError(cudaFree(d_data_slices[i]),
				  "GpuSlice cudaFree data_slices failed",
				  __FILE__, __LINE__);
		}
	    }
	    if (d_data_slices) delete[] d_data_slices;
	    if (data_slices)   delete[] data_slices;
	}

	/**
	 * \addtogroup PublicInterface
	 * @{
	 */

	/**
	 * @brief Copy MST result on the GPU back to a host vector.
	 *
	 * @param[out] Host-side vector to store computed MST results.
	 *
	 *\return cudaError_t
	 * object which indicates the success of all CUDA function calls.
	 */
	cudaError_t Extract()
	{
	    cudaError_t retval = cudaSuccess;

	    do
	    {
		if (num_gpus == 1)
		{
		    // Set device
		    if (util::GRError(cudaSetDevice(gpu_idx[0]),
				      "MSTProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
		    /*
		      if (retval = util::GRError(cudaMemcpy(
		      h_component_ids,
		      data_slices[0]->d_component_ids,
		      sizeof(VertexId) * nodes,
		      cudaMemcpyDeviceToHost),
		      "MSTProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
		    */
		}
		else
		{
		    // TODO: multi-GPU extract result
		} //end if (data_slices.size() ==1)
		for (int i = 0; i < nodes; ++i)
		{
		    if (h_component_ids[i] == i)
		    {
			++num_components;
		    }
		}

	    } while(0);

	    return retval;
	}

	/**
	 * @brief MSTProblem initialization
	 *
	 * @param[in] stream_from_host Whether to stream data from host.
	 * @param[in] graph Reference to the CSR graph object we process on.
	 * @see Csr
	 * @param[in] _num_gpus Number of the GPUs used.
	 *
	 * \return cudaError_t object
	 * which indicates the success of all CUDA function calls.
	 */
	cudaError_t Init(
	    bool stream_from_host, // Only meaningful for single-GPU
	    const Csr<VertexId, Value, SizeT> &graph,
	    int  _num_gpus)
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
	    data_slices   = new DataSlice*[num_gpus];
	    d_data_slices = new DataSlice*[num_gpus];


	    cudaError_t retval = cudaSuccess;

	    do {
		if (num_gpus <= 1)
		{
		    gpu_idx = (int*)malloc(sizeof(int));

		    // Create a single data slice for the currently-set gpu
		    int gpu;
		    if (retval = util::GRError(cudaGetDevice(&gpu),
					       "MSTProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
		    gpu_idx[0] = gpu;

		    data_slices[0] = new DataSlice;

		    // Create SoA on device
		    /*
		      VertexId    *d_component_ids;
		      if (retval = util::GRError(cudaMalloc(
		      (void**)&d_component_ids,
		      nodes * sizeof(VertexId)),
		      "MSTProblem cudaMalloc d_component_ids failed", __FILE__, __LINE__))
		      return retval;
		      data_slices[0]->d_component_ids = d_component_ids;
		    */

		    //TODO: add multi-GPU allocation code
		} while (0);

		return retval;
	    }

	    /**
	     * @brief Performs any initialization work needed for MST problem type.
	     * Must be called prior to each MST run.
	     *
	     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
	     *
	     *  \return cudaError_t object
	     * which indicates the success of all CUDA function calls.
	     */
	    cudaError_t Reset(FrontierType frontier_type)
	    {
		typedef ProblemBase<
		    VertexId, SizeT,
		    _USE_DOUBLE_BUFFER> BaseProblem;

		//load ProblemBase Reset
		BaseProblem::Reset(frontier_type, 1.0);

		cudaError_t retval = cudaSuccess;

		for (int gpu = 0; gpu < num_gpus; ++gpu)
		{
		    // Set device
		    if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
		        "MSTProblem cudaSetDevice failed", __FILE__, __LINE__))
			return retval;

		    // Allocate output component_ids if necessary
		    /*
		      if (!data_slices[gpu]->d_component_ids)
		      {
		      VertexId    *d_component_ids;
		      if (retval = util::GRError(cudaMalloc(
		      (void**)&d_component_ids,
		      nodes * sizeof(VertexId)),
		      "MSTProblem cudaMalloc d_component_ids failed",
		      __FILE__, __LINE__)) return retval;
		      data_slices[gpu]->d_component_ids = d_component_ids;
		      }
		    */

		    // Initialize edge frontier_queue
		    util::MemsetIdxKernel<<<128, 128>>>(
			BaseProblem::graph_slices[0]->frontier_queues.d_keys[0],
			edges);

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
