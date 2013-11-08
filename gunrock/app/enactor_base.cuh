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

namespace gunrock {
namespace app {

/**
 * Base class for graph problem enactors.
 * 
 */
class EnactorBase
{
protected:  

    //Device properties
    util::CudaProperties cuda_props;
    
    // Queue size counters and accompanying functionality
    util::CtaWorkProgressLifetime work_progress;

    FrontierType frontier_type;

public:

    // if DEBUG is set, print details to stdout
    bool DEBUG;

    FrontierType GetFrontierType() { return frontier_type;}

protected:  

    /**
     * Constructor.
     */
    EnactorBase(FrontierType frontier_type, bool DEBUG) :
        frontier_type(frontier_type),
        DEBUG(DEBUG)
    {
        // Setup work progress (only needs doing once since we maintain
        // it in our kernel code)
        work_progress.Setup();
    }

    /**
     * Utility function: Returns the default maximum number of threadblocks
     * this enactor class can launch.
     */
    int MaxGridSize(int cta_occupancy, int max_grid_size = 0)
    {
        if (max_grid_size <= 0) {
            // No override: Fully populate all SMs
            max_grid_size = this->cuda_props.device_props.multiProcessorCount * cta_occupancy;
        }

        return max_grid_size;
    }

    /**
     * Utility method to display the contents of a device array
     */
    template <typename T>
    void DisplayDeviceResults(
        T *d_data,
        size_t num_elements)
    {
        // Allocate array on host and copy back
        T *h_data = (T*) malloc(num_elements * sizeof(T));
        cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

        // Display data
        for (int i = 0; i < num_elements; i++) {
            util::PrintValue(h_data[i]);
            printf(", ");
        }
        printf("\n\n");

        // Cleanup
        if (h_data) free(h_data);
    }
};


} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
