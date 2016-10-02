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
 * @tparam _Problem
 */
template <
    typename _Problem> 
class TOPKEnactor : public EnactorBase<typename _Problem::SizeT>
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;   
    typedef typename Problem::Value    Value   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    Problem    *problem;
    ContextPtr *context;

public:
  
    /**
     * @brief TOPKEnactor constructor
     */
    TOPKEnactor(
        int   num_gpus   = 1,  
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true) :
        BaseEnactor(
            EDGE_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        problem (NULL),
        context (NULL)
    {
    }
  
    /**
     * @brief TOPKEnactor destructor
     */
    virtual ~TOPKEnactor()
    {
    }
  
    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitTOPK(
        ContextPtr *context,
        Problem    *problem,
        int         max_grid_size = 0)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = BaseEnactor::Init(
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
            return retval;

        this -> problem = problem;
        this -> context = context;

        return retval;
    }
 
    /**
     * @brief Enacts a degree centrality on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for forward edge mapping.
     * @tparam FilterKernelPolicy Kernel policy for filtering.
     *
     * @param[in] top_nodes Number of top nodes to process.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactTOPK(
        SizeT         top_nodes)
  {
        typedef TOPKFunctor<VertexId, SizeT, Value, Problem> TopkFunctor;
        typedef typename Problem::DataSlice DataSlice;

        // single gpu graph slice
        GraphSlice<VertexId, SizeT, Value> 
                  *graph_slice   =  problem -> graph_slices[0];
        DataSlice *data_slice    =  problem -> data_slices [0].GetPointer(util::HOST);
        util::CtaWorkProgressLifetime<SizeT>
                  *work_progress = &this->work_progress    [0];
        SizeT      nodes         = graph_slice -> nodes;
        cudaError_t retval       = cudaSuccess;

        // add out-going and in-going degrees -> sum stored in d_degrees_s
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice -> degrees_s.GetPointer(util::DEVICE),
            data_slice -> degrees_o.GetPointer(util::DEVICE),
            nodes);
        util::MemsetAddVectorKernel<<<128, 128>>>(
            data_slice -> degrees_s.GetPointer(util::DEVICE),
            data_slice -> degrees_i.GetPointer(util::DEVICE),
            nodes);

        // sort node_ids by degree centralities
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice -> temp_i   .GetPointer(util::DEVICE),
            data_slice -> degrees_s.GetPointer(util::DEVICE),
            nodes);
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice -> temp_o   .GetPointer(util::DEVICE),
            data_slice -> degrees_s.GetPointer(util::DEVICE),
            nodes);

        util::CUBRadixSort<SizeT, VertexId>(
            false, nodes,
            data_slice -> degrees_s.GetPointer(util::DEVICE),
            data_slice -> node_id  .GetPointer(util::DEVICE));
        util::CUBRadixSort<SizeT, SizeT>(
            false, nodes,
            data_slice -> temp_i   .GetPointer(util::DEVICE),
            data_slice -> degrees_i.GetPointer(util::DEVICE));
        util::CUBRadixSort<SizeT, SizeT>(
            false, nodes,
            data_slice -> temp_o   .GetPointer(util::DEVICE),
            data_slice -> degrees_o.GetPointer(util::DEVICE));

        // check if any of the frontiers overflowed due to redundant expansion
        bool overflowed = false;
        if (retval = work_progress -> CheckOverflow(overflowed)) return retval;
        if (overflowed)
        {
            retval = util::GRError(
                cudaErrorInvalidConfiguration,
                "Frontier queue overflow. Please increase queus size factor.",
                __FILE__, __LINE__); 
            return retval;
        }

        if (this -> debug) printf("==> GPU Top K Degree Centrality Complete.\n");
        return retval;
    }
 
    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
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
	    Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
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
 
    /**
     * \addtogroup PublicInterface
     * @{
     */

    cudaError_t Reset()
    {
        return BaseEnactor::Reset();
    }
 
    /**
     * @brief TOPK Init kernel entry.
     *
     * @tparam TOPKProblem TOPK Problem type. @see TOPKProblem
     *
     * @param[in] context CUDA context pointer.
     * @param[in] problem Pointer to TOPKProblem object.
     * @param[in] max_grid_size Max grid size for TOPK kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
    */
    cudaError_t Init(
        ContextPtr *context,
        Problem    *problem,
        int	       max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i=0; i< this->num_gpus; i++)
        if (min_sm_version == -1 || 
            this->cuda_props[i].device_sm_version < min_sm_version)
        {
            min_sm_version = this->cuda_props[i].device_sm_version;
        }

        if (min_sm_version >= 300) 
        { 
            return  InitTOPK<AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs
    
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }
 
    /**
     * @brief TOPK Enact kernel entry.
     *
     * @tparam TOPKProblem TOPK Problem type. @see TOPKProblem
     *
     * @param[in] top_nodes Top nodes to process.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
    */
    cudaError_t Enact(
        SizeT         top_nodes)
    {
        int min_sm_version = -1;
        for (int i=0; i< this->num_gpus; i++)
        if (min_sm_version == -1 || 
            this->cuda_props[i].device_sm_version < min_sm_version)
        {
            min_sm_version = this->cuda_props[i].device_sm_version;
        }

        if (min_sm_version >= 300) 
        { 
            return  EnactTOPK<AdvanceKernelPolicy, FilterKernelPolicy>(top_nodes);
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
