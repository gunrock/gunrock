// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_base.cuh
 *
 * @brief Base Graph Problem Enactor
 */

#pragma once
#include <time.h>

#include <boost/predef.h>

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/util/sharedmem.cuh>
#include <gunrock/util/info.cuh>
#include <gunrock/app/problem_base.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_kernel.cuh>
#include <gunrock/app/enactor_types.cuh>
#include <gunrock/app/enactor_helper.cuh>
#include <gunrock/app/enactor_loop.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

/**
 * @brief Base class for graph problem enactor.
 *
 * @tparam SizeT
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template <
    typename SizeT> //,
    //bool     _DEBUG,  // if DEBUG is set, print details to STDOUT
    //bool     _SIZE_CHECK>
class EnactorBase
{
public:
    //static const bool DEBUG = _DEBUG;
    //static const bool SIZE_CHECK = _SIZE_CHECK;
    int           num_gpus;
    int          *gpu_idx;
    FrontierType  frontier_type;
    bool          instrument;
    bool          debug     ;
    bool          size_check;

    //Device properties
    util::Array1D<SizeT, util::CudaProperties>          cuda_props        ;

    // Queue size counters and accompanying functionality
    util::Array1D<SizeT, util::CtaWorkProgressLifetime<SizeT> > work_progress     ;
    util::Array1D<SizeT, EnactorStats<SizeT> >                  enactor_stats     ;
    util::Array1D<SizeT, FrontierAttribute<SizeT> >     frontier_attribute;

    FrontierType GetFrontierType() {return frontier_type;}

protected:

    /**
     * @brief Constructor
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] num_gpus
     * @param[in] gpu_idx
     */
    EnactorBase(
        FrontierType  _frontier_type,
        int           _num_gpus,
        int          *_gpu_idx,
        bool          _instrument,
        bool          _debug,
        bool          _size_check) :
        frontier_type (_frontier_type),
        num_gpus      (_num_gpus     ),
        gpu_idx       (_gpu_idx      ),
        instrument    (_instrument   ),
        debug         (_debug        ),
        size_check    (_size_check   )
    {
        cuda_props        .SetName("cuda_props"        );
        work_progress     .SetName("work_progress"     );
        enactor_stats     .SetName("enactor_stats"     );
        frontier_attribute.SetName("frontier_attribute");
        
        if (cuda_props        .Init(
            num_gpus         , util::HOST, true, 
            cudaHostAllocMapped | cudaHostAllocPortable)) 
            return;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            // Setup work progress (only needs doing once since we maintain
            // it in our kernel code)
            cuda_props   [gpu].Setup(gpu_idx[gpu]);
        }      
    }

    /**
     * @brief EnactorBase destructor
     */
    virtual ~EnactorBase()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval;
        if (enactor_stats.GetPointer() != NULL)
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            for (int peer=0;peer<num_gpus;peer++)
            {
                if (retval = enactor_stats [gpu*num_gpus+peer].Release())
                    return retval;
                if (retval = work_progress [gpu*num_gpus+peer].Release()) 
                    return retval;
                if (retval = frontier_attribute[gpu*num_gpus + peer].Release())
                    return retval;
            }
        }
        if (retval = work_progress     .Release()) return retval;
        if (retval = cuda_props        .Release()) return retval;
        if (retval = enactor_stats     .Release()) return retval;
        if (retval = frontier_attribute.Release()) return retval;
        return retval;
    }

   /**
     * @brief Init function for enactor base class.
     *
     * @tparam Problem
     *
     * @param[in] problem The problem object for the graph primitive
     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
     * @param[in] advance_occupancy CTA Occupancy for Advance operator
     * @param[in] filter_occupancy CTA Occupancy for Filter operator
     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 256 by default.
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    //template <typename Problem>
    cudaError_t Init(
        //Problem *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = work_progress     .Init(
            num_gpus*num_gpus, util::HOST, true, 
            cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;
        if (retval = enactor_stats     .Init(
            num_gpus*num_gpus, util::HOST, true, 
            cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;
        if (retval = frontier_attribute.Init(
            num_gpus*num_gpus, util::HOST, true, 
            cudaHostAllocMapped | cudaHostAllocPortable))
            return retval;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            // Setup work progress (only needs doing once since we maintain
            // it in our kernel code)
            for (int peer=0;peer<num_gpus;peer++)
            {
                if (retval = work_progress     [gpu*num_gpus + peer].Init())
                    return retval;
                if (retval = frontier_attribute[gpu*num_gpus + peer].Init())
                    return retval;
                EnactorStats<SizeT> *enactor_stats_ = enactor_stats + gpu*num_gpus + peer;
                //initialize runtime stats
                enactor_stats_ -> advance_grid_size = MaxGridSize(
                    gpu, advance_occupancy, max_grid_size);
                enactor_stats_ -> filter_grid_size  = MaxGridSize(
                    gpu, filter_occupancy , max_grid_size);
                if (retval = enactor_stats_ -> Init(node_lock_size))
                    return retval;
            }
        }
        return retval;
    }

    /*
     * @brief Reset function.
     */
    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            for (int peer=0; peer<num_gpus; peer++)
            {
                if (retval = enactor_stats     [gpu * num_gpus + peer]
                    .Reset())
                    return retval;
                if (retval = work_progress     [gpu * num_gpus + peer]
                    .Reset_())
                    return retval;
                if (retval = frontier_attribute[gpu * num_gpus + peer]
                    .Reset())
                    return retval;
            }
        }
        return retval;
    }

    /**
     * @brief Setup function for enactor base class.
     *
     * @tparam Problem
     *
     * @param[in] problem The problem object for the graph primitive
     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
     * @param[in] advance_occupancy CTA Occupancy for Advance operator
     * @param[in] filter_occupancy CTA Occupancy for Filter operator
     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 256 by default.
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    //template <typename Problem>
    cudaError_t Setup(
        //Problem *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = Init(/*problem,*/ max_grid_size, 
            advance_occupancy, filter_occupancy, node_lock_size)) return retval;
        if (retval = Reset()) return retval;
        return retval;
    }

    /**
     * @brief Utility function for getting the max grid size.
     *
     * @param[in] gpu
     * @param[in] cta_occupancy CTA occupancy for current architecture
     * @param[in] max_grid_size Preset max grid size. If less or equal to 0, fully populate all SMs
     *
     * \return The maximum number of thread blocks this enactor class can launch.
     */
    int MaxGridSize(int gpu, int cta_occupancy, int max_grid_size = 0)
    {
        if (max_grid_size <= 0) {
            max_grid_size = cuda_props[gpu].device_props.multiProcessorCount * cta_occupancy;
        }

        return max_grid_size;
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
