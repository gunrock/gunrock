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

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/app/problem_base.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {

/**
 * @brief Structure for auxiliary variables used in enactor.
 */
struct EnactorStats
{
    long long           iteration;
    unsigned int        num_gpus;
    unsigned int        gpu_id;

    unsigned long long  total_lifetimes;
    unsigned long long  total_runtimes;
    unsigned long long  total_queued;

    unsigned int        advance_grid_size;
    unsigned int        filter_grid_size;

    util::KernelRuntimeStatsLifetime advance_kernel_stats;
    util::KernelRuntimeStatsLifetime filter_kernel_stats;

    unsigned int        *d_node_locks;
    unsigned int        *d_node_locks_out;

};

/**
 * @brief Structure for auxiliary variables used in frontier operations.
 */
struct FrontierAttribute
{
    unsigned int        queue_length;
    unsigned int        queue_index;
    int                 selector;
    bool                queue_reset;
    int                 current_label;
    gunrock::oprtr::advance::TYPE   advance_type;
};

/**
 * @brief Base class for graph problem enactors.
 */
class EnactorBase
{
protected:  

    //Device properties
    util::CudaProperties            cuda_props;
    
    // Queue size counters and accompanying functionality
    util::CtaWorkProgressLifetime   work_progress;

    FrontierType                    frontier_type;

    EnactorStats                    enactor_stats;

    FrontierAttribute               frontier_attribute;

public:

    // if DEBUG is set, print details to stdout
    bool DEBUG;

    FrontierType GetFrontierType() { return frontier_type;}

protected:  

    /**
     * @brief Constructor
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] DEBUG If set, will collect kernel running stats and display the running info.
     */
    EnactorBase(FrontierType frontier_type, bool DEBUG) :
        frontier_type(frontier_type),
        DEBUG(DEBUG)
    {
        // Setup work progress (only needs doing once since we maintain
        // it in our kernel code)
        work_progress.Setup();
        enactor_stats.d_node_locks = NULL;
        enactor_stats.d_node_locks_out = NULL;
    }

    /**
     * @brief Destructor
     */
    virtual ~EnactorBase()
    {
        if (enactor_stats.d_node_locks) util::GRError(cudaFree(enactor_stats.d_node_locks), "EnactorBase cudaFree d_node_locks failed", __FILE__, __LINE__);
        if (enactor_stats.d_node_locks_out) util::GRError(cudaFree(enactor_stats.d_node_locks_out), "EnactorBase cudaFree d_node_locks_out failed", __FILE__, __LINE__);
    }

    /**
     * @brief Setup function for enactor base class
     *
     * @param[in] problem The problem object for the graph primitive
     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
     * @param[in] advance_occupancy CTA Occupancy for Advance operator
     * @param[in] filter_occupancy CTA Occupancy for Filter operator
     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 256 by default.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    {
        cudaError_t retval = cudaSuccess;

        //initialize runtime stats
        enactor_stats.advance_grid_size = MaxGridSize(advance_occupancy, max_grid_size);
        enactor_stats.filter_grid_size  = MaxGridSize(filter_occupancy, max_grid_size);

        if (retval = enactor_stats.advance_kernel_stats.Setup(enactor_stats.advance_grid_size)) return retval;
        if (retval = enactor_stats.filter_kernel_stats.Setup(enactor_stats.filter_grid_size)) return retval;

        enactor_stats.iteration             = 0;
        enactor_stats.total_runtimes        = 0;
        enactor_stats.total_lifetimes       = 0;
        enactor_stats.total_queued          = 0;

        enactor_stats.num_gpus              = 1;
        enactor_stats.gpu_id                = 0;

        if (retval = util::GRError(cudaMalloc(
                            (void**)&enactor_stats.d_node_locks,
                            node_lock_size * sizeof(unsigned int)),
                        "EnactorBase cudaMalloc d_node_locks failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMalloc(
                            (void**)&enactor_stats.d_node_locks_out,
                            node_lock_size * sizeof(unsigned int)),
                        "EnactorBase cudaMalloc d_node_locks_out failed", __FILE__, __LINE__)) return retval;

        return retval;
    }

    /**
     * @brief Utility function for getting the max grid size.
     *
     * @param[in] cta_occupancy CTA occupancy for current architecture
     * @param[in] max_grid_size Preset max grid size. If less or equal to 0, fully populate all SMs
     *
     * \return The maximum number of threadblocks this enactor class can launch.
     */
    int MaxGridSize(int cta_occupancy, int max_grid_size = 0)
    {
        if (max_grid_size <= 0) {
            max_grid_size = this->cuda_props.device_props.multiProcessorCount * cta_occupancy;
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
