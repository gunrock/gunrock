// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * topk_enactor.cuh
 *
 * @brief TOPK Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/sort_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/topk/topk_problem.cuh>
#include <gunrock/app/topk/topk_functor.cuh>

#include <cub/cub.cuh>
#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace topk {

/**
 * @brief TOPK problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<typename _Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK>
class TOPKEnactor : public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;   
    typedef typename Problem::Value    Value   ;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;

  // Members
protected:
  
  /**
   * CTA duty kernel stats
   */
    
  unsigned long long total_runtimes;  // Total working time by each CTA
  unsigned long long total_lifetimes; // Total life time of each CTA
  unsigned long long total_queued;
  
  /**
   * A pinned, mapped word that the traversal kernels will signal when done
   */
  //volatile int        *done;
  //int                 *d_done;
  //cudaEvent_t         throttle_event;
  
  /**
   * Current iteration, also used to get the final search depth of the TOPK search
   */
  long long           iteration;
  
  // Methods
protected:
  
  /**
   * @brief Prepare the enactor for TOPK kernel call. Must be called prior to each TOPK iteration.
   *
   * @param[in] problem TOPK Problem object which holds the graph data and TOPK problem data to compute.
   * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
   * @param[in] filter_grid_size CTA occupancy for filter kernel call.
   *
   * \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  template <typename ProblemData>
  cudaError_t Setup(ProblemData *problem)
  {
    typedef typename ProblemData::SizeT     SizeT;
    typedef typename ProblemData::VertexId  VertexId;
    
    cudaError_t retval = cudaSuccess;
    
    //initialize the host-mapped "done"
    //if (!done) {
    {
      //int flags = cudaHostAllocMapped;
      
      // Allocate pinned memory for done
      //if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
	  //"TOPKEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) return retval;
      
      // Map done into GPU space
      //if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
	  //"TOPKEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) return retval;
      
      // Create throttle event
      //if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
	  //"TOPKEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) return retval;
    }
    
    //graph slice
    //typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];
    //typename ProblemData::DataSlice  *data_slice  = problem->data_slices[0];
  
    do {
      // Bind row-offsets and bitmask texture
      /*cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
      if (retval = util::GRError(cudaBindTexture(0,
	 gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
	 graph_slice->d_row_offsets,
	 row_offsets_desc,
	 (graph_slice->nodes + 1) * sizeof(SizeT)),
	 "TOPKEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;*/
      
      
      /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
	if (retval = util::GRError(cudaBindTexture(
	0,
	gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
	graph_slice->d_column_indices,
	column_indices_desc,
	graph_slice->edges * sizeof(VertexId)),
	"TOPKEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
    } while (0);
    
    return retval;
  }
  
public:
  
  /**
   * @brief TOPKEnactor constructor
   */
  TOPKEnactor(int *gpu_idx) :
    EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>(EDGE_FRONTIERS, 1, gpu_idx),
    iteration(0),
    total_queued(0)
    //done(NULL),
    //d_done(NULL)
  {}
  
  /**
   * @brief TOPKEnactor destructor
   */
  virtual ~TOPKEnactor()
  {
    //if (done) 
    //{
    //  util::GRError(cudaFreeHost((void*)done),
	//    "TOPKEnactor cudaFreeHost done failed", __FILE__, __LINE__);
      
    //  util::GRError(cudaEventDestroy(throttle_event),
	//    "TOPKEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
    //}
  }
  
  /**
   * \addtogroup PublicInterface
   * @{
   */
  
  /**
   * @brief Obtain statistics about the last TOPK search enacted.
   *
   * @param[out] total_queued Total queued elements in TOPK kernel running.
   * @param[out] search_depth Search depth of TOPK algorithm.
   * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
   */
template <typename VertexId>
void GetStatistics(long long   &total_queued,
		   VertexId    &search_depth,
		   double      &avg_duty)
  {
    cudaThreadSynchronize();
    
    total_queued = this->total_queued;
    search_depth = this->iteration;
    
    avg_duty = (total_lifetimes > 0) ?
      double(total_runtimes) / total_lifetimes : 0.0;
  }
  
  /** @} */
  
  /**
   * @brief Enacts a degree centrality on the specified graph.
   *
   * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
   * @tparam FilterKernelPolicy Kernel policy for filtering.
   * @tparam TOPKProblem TOPK Problem type.
   *
   * @param[in] problem TOPKProblem object.
   * @param[in] max_grid_size Max grid size for TOPK kernel calls.
   *
   * \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename TOPKProblem>
  cudaError_t EnactTOPK(
    ContextPtr context,
		      TOPKProblem   *problem,
		      int         top_nodes,
		      float         max_grid_size = 0)
  {
    typedef typename TOPKProblem::SizeT      SizeT;
    typedef typename TOPKProblem::Value      Value;
    typedef typename TOPKProblem::VertexId   VertexId;

    typedef TOPKFunctor<VertexId, SizeT, Value, TOPKProblem> TopkFunctor;

    cudaError_t retval = cudaSuccess;
   
do
    {
      // initialization
      if (retval = Setup(problem)) break;
      if (retval = EnactorBase<SizeT, _DEBUG, _SIZE_CHECK>::Setup(
        problem,
        max_grid_size,
        AdvanceKernelPolicy::CTA_OCCUPANCY,
        FilterKernelPolicy::CTA_OCCUPANCY)) break;

      // single gpu graph slice
      GraphSlice<SizeT, VertexId, Value> *graph_slice = problem->graph_slices[0];
      //typename TOPKProblem::DataSlice *d_data_slice = problem->d_data_slices[0];
      typename TOPKProblem::DataSlice *data_slice = problem->data_slices[0];
      util::CtaWorkProgressLifetime
                     *work_progress      = &this->work_progress     [0];

      // add out-going and in-going degrees -> sum stored in d_degrees_s
      util::MemsetCopyVectorKernel<<<128, 128>>>(
        data_slice ->d_degrees_s,
        data_slice ->d_degrees_o,
        graph_slice->nodes);
      util::MemsetAddVectorKernel<<<128, 128>>>(
        data_slice ->d_degrees_s,
        data_slice ->d_degrees_i,
        graph_slice->nodes);

      // sort node_ids by degree centralities
      util::MemsetCopyVectorKernel<<<128, 128>>>(
        problem->data_slices[0]->d_temp_i,
        problem->data_slices[0]->d_degrees_s,
        graph_slice->nodes);
      util::MemsetCopyVectorKernel<<<128, 128>>>(
        problem->data_slices[0]->d_temp_o,
        problem->data_slices[0]->d_degrees_s,
        graph_slice->nodes);
      util::CUBRadixSort<Value, VertexId>(
        false, graph_slice->nodes,
        problem->data_slices[0]->d_degrees_s,
        problem->data_slices[0]->d_node_id);
      util::CUBRadixSort<Value, VertexId>(
        false, graph_slice->nodes,
        problem->data_slices[0]->d_temp_i,
        problem->data_slices[0]->d_degrees_i);
      util::CUBRadixSort<Value, VertexId>(
        false, graph_slice->nodes,
        problem->data_slices[0]->d_temp_o,
        problem->data_slices[0]->d_degrees_o);

      // check if any of the frontiers overflowed due to redundant expansion
      bool overflowed = false;
      if (retval = work_progress->CheckOverflow<SizeT>(overflowed)) break;
      if (overflowed)
      {
        retval = util::GRError(
          cudaErrorInvalidConfiguration,
          "Frontier queue overflow. Please increase queus size factor.",
          __FILE__, __LINE__); break;
      }
    } while(0);
    if (DEBUG) printf("==> GPU Top K Degree Centrality Complete.\n");
    return retval;
 
  }
  
  /**
   * \addtogroup PublicInterface
   * @{
   */
  
  /**
   * @brief TOPK Enact kernel entry.
   *
   * @tparam TOPKProblem TOPK Problem type. @see TOPKProblem
   *
   * @param[in] problem Pointer to TOPKProblem object.
   * @param[in] src Source node for TOPK.
   * @param[in] max_grid_size Max grid size for TOPK kernel calls.
   *
   * \return cudaError_t object which indicates the success of all CUDA function calls.
   */
  template <typename TOPKProblem>
  cudaError_t Enact(ContextPtr context,
		    TOPKProblem   *problem,
		    int         top_nodes,
		    int	        max_grid_size = 0)
  {
    int min_sm_version = -1;
    for (int i=0;i<this->num_gpus;i++)
        if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
            min_sm_version = this->cuda_props[i].device_sm_version;

    if (min_sm_version >= 300) 
    {
      typedef gunrock::oprtr::filter::KernelPolicy<
	TOPKProblem,                          // Problem data type
	300,                                // CUDA_ARCH
	INSTRUMENT,                         // INSTRUMENT
	0,                                  // SATURATION QUIT
	true,                               // DEQUEUE_PROBLEM_SIZE
	8,                                  // MIN_CTA_OCCUPANCY
	8,                                  // LOG_THREADS
	1,                                  // LOG_LOAD_VEC_SIZE
	0,                                  // LOG_LOADS_PER_TILE
	5,                                  // LOG_RAKING_THREADS
	5,                                  // END_BITMASK_CULL
	8>                                  // LOG_SCHEDULE_GRANULARITY
	FilterKernelPolicy;
      
      typedef gunrock::oprtr::advance::KernelPolicy<
	TOPKProblem,                          // Problem data type
	300,                                // CUDA_ARCH
	INSTRUMENT,                         // INSTRUMENT
	8,                                  // MIN_CTA_OCCUPANCY
	7,                                  // LOG_THREADS
	8,                                  // LOG_BLOCKS
	32 * 128,                           // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
	1,                                  // LOG_LOAD_VEC_SIZE
	0,                                  // LOG_LOADS_PER_TILE
	5,                                  // LOG_RAKING_THREADS
	32,                                 // WARP_GATHER_THRESHOLD
	128 * 4,                            // CTA_GATHER_THRESHOLD
	7,                                  // LOG_SCHEDULE_GRANULARITY
	gunrock::oprtr::advance::TWC_FORWARD>        
	AdvanceKernelPolicy;
      
      return  EnactTOPK<AdvanceKernelPolicy, FilterKernelPolicy, TOPKProblem>(context,
									  problem,
									  top_nodes,
									  max_grid_size);
    }
    
    //to reduce compile time, get rid of other architecture for now
    //TODO: add all the kernelpolicy settings for all archs
    
    printf("Not yet tuned for this architecture\n");
    return cudaErrorInvalidDeviceFunction;
  
  }
  
  /** @} */
  
};
  
} // namespace topk
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
