// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * oprtr_parameters.cuh
 *
 * @brief Data structure for operator parameters
 */

#pragma once

#include <gunrock/app/frontier.cuh>
#include <gunrock/oprtr/oprtr_base.cuh>
#include <moderngpu/context.hxx>

namespace gunrock {
namespace oprtr {
    template <typename GraphT, typename FrontierT, typename _LabelT>
    struct OprtrParameters {
        typedef typename GraphT::VertexT VertexT;
        typedef typename GraphT::SizeT SizeT;
        typedef typename GraphT::ValueT ValueT;
        typedef _LabelT LabelT;

        FrontierT *frontier;
        util::Array1D<SizeT, ValueT> *values_in;
        util::Array1D<SizeT, ValueT> *values_out;
        util::Array1D<SizeT, ValueT> *reduce_values_temp;
        util::Array1D<SizeT, ValueT> *reduce_values_temp2;
        util::Array1D<SizeT, ValueT> *reduce_values_out;
        util::Array1D<SizeT, SizeT> *vertex_markers;
        util::Array1D<SizeT, unsigned char> *visited_masks;
        util::Array1D<SizeT, LabelT> *labels;
        util::CudaProperties *cuda_props;

        mgpu::standard_context_t* context = nullptr;
        cudaStream_t stream;

        bool get_output_length;
        bool reduce_reset;
        std::string advance_mode;
        std::string filter_mode;
        LabelT label;
        int max_grid_size;

        // Set print_prop to false to hide output
        OprtrParameters(cudaStream_t stream = 0) { Init(); }

        ~OprtrParameters() { Release(); }

        cudaError_t Init(cudaStream_t stream = 0) {
            this->stream = stream;

            if (context != nullptr) {
                Release();
            }
            context = new mgpu::standard_context_t(false, stream);
            if (context == nullptr) {
                return cudaErrorMemoryAllocation;
            }

            frontier = NULL;
            values_in = NULL;
            values_out = NULL;
            reduce_values_temp = NULL;
            reduce_values_temp2 = NULL;
            reduce_values_out = NULL;
            vertex_markers = NULL;
            visited_masks = NULL;
            labels = NULL;
            cuda_props = NULL;
            get_output_length = true;
            reduce_reset = true;
            advance_mode = "";
            filter_mode = "";
            max_grid_size = 0;
            return cudaSuccess;
        }

        cudaError_t Release() {
            delete context;
            context = nullptr;

            return cudaSuccess;
        }
    };
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
