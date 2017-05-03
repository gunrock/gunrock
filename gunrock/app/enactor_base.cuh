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
//#include <time.h>

//#include <boost/predef.h>

#include <moderngpu.cuh>

#include <gunrock/util/cuda_properties.cuh>

#include <gunrock/util/error_utils.cuh>
//#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/array_utils.cuh>
//#include <gunrock/util/sharedmem.cuh>
//#include <gunrock/util/info.cuh>
//#include <gunrock/app/problem_base.cuh>

//#include <gunrock/oprtr/advance/kernel.cuh>
//#include <gunrock/oprtr/advance/kernel_policy.cuh>
//#include <gunrock/oprtr/filter/kernel.cuh>
//#include <gunrock/oprtr/filter/kernel_policy.cuh>

//#include <gunrock/app/enactor_kernel.cuh>
#include <gunrock/app/enactor_types.cuh>
//#include <gunrock/app/enactor_helper.cuh>
//#include <gunrock/app/enactor_loop.cuh>


//using namespace mgpu;

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

using Enactor_Flag = unsigned int;

enum : Enactor_Flag
{
    Instrument = 0x01,
    Debug      = 0x02,
    Size_Check = 0x04,
};

cudaError_t UseParameters2(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    if (!parameters.Have("device"))
    {
        retval = parameters.Use<int>(
            "device",
            util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
            0,
            "Set GPU(s) for testing",
            __FILE__, __LINE__);
        if (retval) return retval;
    }

    retval = parameters.Use<int>(
        "communicate-latency",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "additional communication latency",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<float>(
        "communicate-multipy",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        1.0f,
        "communication sizing factor",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<int>(
        "expand-latency",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "additional expand incoming latency",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<int>(
        "subqueue-latency",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "additional subqueue latency",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<int>(
        "fullqueue-latency",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "additional fullqueue latency",
        __FILE__, __LINE__);
    if (retval) return retval;

    retval = parameters.Use<int>(
        "makeout-latency",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "additional make-out latency",
        __FILE__, __LINE__);
    if (retval) return retval;
    return retval;
}

/**
 * @brief Base class for graph problem enactor.
 *
 * @tparam SizeT
 */
template <typename GraphT,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class EnactorBase
{
public:
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef EnactorSlice<GraphT, ARRAY_FLAG, cudaHostRegisterFlag>
                                     EnactorSliceT;
    //typedef Frontier<VertexT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag>
    //                                 FrontierT;
    int           num_gpus;
    std::vector<int> gpu_idx;
    std::string   algo_name;

    int           communicate_latency;
    float         communicate_multipy;
    int           expand_latency;
    int           subqueue_latency;
    int           fullqueue_latency;
    int           makeout_latency;
    int           min_sm_version;

    //Device properties
    util::Array1D<SizeT, util::CudaProperties, ARRAY_FLAG,
        cudaHostRegisterFlag | cudaHostAllocMapped | cudaHostAllocPortable>
        cuda_props;

    //Per-GPU enactor slices
    util::Array1D<int, EnactorSliceT, ARRAY_FLAG,
        cudaHostRegisterFlag>// | cudaHostAllocMapped | cudaHostAllocPortable>
        enactor_slices;

    //Frontiers
    //util::Array1D<int, FrontierT, ARRAY_FLAG, cudaHostRegisterFlag>
    //    frontiers;

#ifdef ENABLE_PERFORMANCE_PROFILING
    util::Array1D<int, std::vector<std::vector<double> > > iter_full_queue_time;
    util::Array1D<int, std::vector<std::vector<double> > > iter_sub_queue_time;
    util::Array1D<int, std::vector<std::vector<double> > > iter_total_time;
    util::Array1D<int, std::vector<std::vector<SizeT > > > iter_full_queue_nodes_queued;
    util::Array1D<int, std::vector<std::vector<SizeT > > > iter_full_queue_edges_queued;
#endif

//protected:

    /**
     * @brief Constructor
     *
     * @param[in] _frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] _num_gpus
     * @param[in] _gpu_idx
     * @param[in] _instrument
     * @param[in] _debug
     * @param[in] _size_check
     */
    EnactorBase(std::string algo_name = "test")
    {
        this -> algo_name = algo_name;
        cuda_props          .SetName("cuda_props"          );
        enactor_slices      .SetName("enactor_slices"      );

#ifdef ENABLE_PERFORMANCE_PROFILING
        iter_full_queue_time.SetName("iter_full_queue_time");
        iter_sub_queue_time .SetName("iter_sub_queue_time" );
        iter_total_time     .SetName("iter_total_time"     );
        iter_full_queue_edges_queued.SetName("iter_full_queue_edges_queued");
        iter_full_queue_nodes_queued.SetName("iter_full_queue_nodes_queued");
#endif
    }

    /**
     * @brief EnactorBase destructor
     */
    virtual ~EnactorBase()
    {
        Release();
    }

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval;
        if (enactor_slices + 0 != NULL)
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            retval = util::SetDevice(gpu_idx[gpu]);
            if (retval) return retval;
            for (int peer = 0; peer < num_gpus; peer++)
            {
                int idx = gpu * num_gpus + peer;
                retval = enactor_slices[idx].Release(target);
                if (retval) return retval;
            }
        }
        if (retval = cuda_props    .Release(target)) return retval;
        if (retval = enactor_slices.Release(target)) return retval;

#ifdef ENABLE_PERFORMANCE_PROFILING
        if (iter_full_queue_time + 0!= NULL && (target & util::HOST) != 0)
        {
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                for (auto it = iter_full_queue_time[gpu].begin();
                    it != iter_full_queue_time[gpu].end(); it++)
                    it -> clear();
                for (auto it = iter_sub_queue_time[gpu].begin();
                    it != iter_sub_queue_time[gpu].end(); it++)
                    it -> clear();
                for (auto it = iter_total_time[gpu].begin();
                    it != iter_total_time[gpu].end(); it++)
                    it -> clear();
                for (auto it = iter_full_queue_nodes_queued[gpu].begin();
                    it != iter_full_queue_nodes_queued[gpu].end(); it++)
                    it -> clear();
                for (auto it = iter_full_queue_edges_queued[gpu].begin();
                    it != iter_full_queue_edges_queued[gpu].end(); it++)
                    it -> clear();
                iter_full_queue_time[gpu].clear();
                iter_sub_queue_time [gpu].clear();
                iter_total_time     [gpu].clear();
                iter_full_queue_nodes_queued[gpu].clear();
                iter_full_queue_edges_queued[gpu].clear();
            }
            if (retval = iter_full_queue_time.Release(target)) return retval;
            if (retval = iter_sub_queue_time .Release(target)) return retval;
            if (retval = iter_total_time     .Release(target)) return retval;
            if (retval = iter_full_queue_nodes_queued.Release(target)) return retval;
            if (retval = iter_full_queue_edges_queued.Release(target)) return retval;
        }
#endif
        return retval;
    }

   /**
     * @brief Init function for enactor base class.
     *
     * @tparam Problem
     *
     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
     * @param[in] advance_occupancy CTA Occupancy for Advance operator
     * @param[in] filter_occupancy CTA Occupancy for Filter operator
     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 1024 by default.
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    //template <typename Problem>
    cudaError_t Init(
        util::Parameters &parameters,
        //int max_grid_size,
        //int advance_occupancy,
        //int filter_occupancy,
        unsigned int num_queues = 2,
        FrontierType *frontier_types = NULL,
        int node_lock_size = 1024,
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;

        gpu_idx             = parameters.Get<std::vector<int>>("device");
        num_gpus            = gpu_idx.size();
        communicate_latency = parameters.Get<int  >("communicate-latency");
        communicate_multipy = parameters.Get<float>("communicate-multipy");
        expand_latency      = parameters.Get<int  >("expand-latency");
        subqueue_latency    = parameters.Get<int  >("subqueue-latency");
        fullqueue_latency   = parameters.Get<int  >("fullqueue-latency");
        makeout_latency     = parameters.Get<int  >("makeout-latency");
        min_sm_version      = -1;

        retval = cuda_props   .Allocate(num_gpus, util::HOST);
        if (retval) return retval;

        retval = enactor_slices.Allocate(num_gpus * num_gpus, util::HOST);
        if (retval) return retval;

#ifdef ENABLE_PERFORMANCE_PROFILING
        if (retval = iter_full_queue_time.Allocate(num_gpus, util::HOST)) return retval;
        if (retval = iter_sub_queue_time .Allocate(num_gpus, util::HOST)) return retval;
        if (retval = iter_total_time     .Allocate(num_gpus, util::HOST)) return retval;
        if (retval = iter_full_queue_nodes_queued.Allocate(num_gpus, util::HOST)) return retval;
        if (retval = iter_full_queue_edges_queued.Allocate(num_gpus, util::HOST)) return retval;
#endif

        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            if (target & util::DEVICE)
            {
                retval = util::SetDevice(gpu_idx[gpu]);
                if (retval) return retval;

                // Setup work progress (only needs doing once since we maintain
                // it in our kernel code)
                cuda_props   [gpu].Setup(gpu_idx[gpu]);
                if (min_sm_version == -1 ||
                    cuda_props[gpu].device_sm_version < min_sm_version)
                    min_sm_version = cuda_props[gpu].device_sm_version;
            }

            for (int peer = 0; peer < num_gpus; peer++)
            {
                auto &enactor_slice = enactor_slices[gpu*num_gpus + peer];

                retval = enactor_slice.Init(num_queues, frontier_types,
                    algo_name + "::frontier[" + std::to_string(gpu) + "," +
                    std::to_string(peer) + "]", node_lock_size, target);
                if (retval) return retval;

                //initialize runtime stats
                //enactor_slice.enactor_stats -> advance_grid_size = MaxGridSize(
                //    gpu, advance_occupancy, max_grid_size);
                //enactor_slice.enactor_stats -> filter_grid_size  = MaxGridSize(
                //    gpu, filter_occupancy , max_grid_size);

                if (gpu != peer && (target & util::DEVICE) != 0)
                {
                    int peer_access_avail;
                    if (retval = util::GRError(cudaDeviceCanAccessPeer(
                        &peer_access_avail, gpu_idx[gpu], gpu_idx[peer]),
                        "cudaDeviceCanAccess failed", __FILE__, __LINE__))
                        return retval;
                    if (peer_access_avail)
                    {
                        if (retval = util::GRError(cudaDeviceEnablePeerAccess(gpu_idx[peer],0),
                            "cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__))
                            return retval;
                    }
                }
            }

#ifdef ENABLE_PERFORMANCE_PROFILING
            iter_sub_queue_time [gpu].clear();
            iter_full_queue_time[gpu].clear();
            iter_total_time     [gpu].clear();
            iter_full_queue_nodes_queued[gpu].clear();
            iter_full_queue_edges_queued[gpu].clear();
#endif
        }
        return retval;
    }

    /*
     * @brief Reset function.
     */
    cudaError_t Reset(util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            if (target & util::DEVICE)
            {
                retval = util::SetDevice(gpu_idx[gpu]);
                if (retval) return retval;
            }
            for (int peer = 0; peer < num_gpus; peer++)
            {
                if (retval = enactor_slices[gpu * num_gpus + peer]
                    .Reset(target))
                    return retval;
                //if (retval = work_progress     [gpu * num_gpus + peer]
                //    .Reset_())
                //    return retval;
                //if (retval = frontier_attribute[gpu * num_gpus + peer]
                //    .Reset())
                //    return retval;
            }

#ifdef ENABLE_PERFORMANCE_PROFILING
            iter_sub_queue_time [gpu].push_back(std::vector<double>());
            iter_full_queue_time[gpu].push_back(std::vector<double>());
            iter_total_time     [gpu].push_back(std::vector<double>());
            iter_full_queue_nodes_queued[gpu].push_back(std::vector<SizeT>());
            iter_full_queue_edges_queued[gpu].push_back(std::vector<SizeT>());
#endif
        }
        return retval;
    }

    /**
     * @brief Setup function for enactor base class.
     *
     * @tparam Problem
     *
     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
     * @param[in] advance_occupancy CTA Occupancy for Advance operator
     * @param[in] filter_occupancy CTA Occupancy for Filter operator
     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 256 by default.
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    //template <typename Problem>
    //cudaError_t Setup(
    //    int max_grid_size,
    //    int advance_occupancy,
    //    int filter_occupancy,
    //    int node_lock_size = 1024)
    //{
    //    cudaError_t retval = cudaSuccess;

    //    if (retval = Init(/*problem,*/ max_grid_size,
    //        advance_occupancy, filter_occupancy, node_lock_size)) return retval;
    //    if (retval = Reset()) return retval;
    //    return retval;
    //}

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
