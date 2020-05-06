/*
 * XXX: error handling within methods,
 * remove return error_t it adds unnecessary
 * work for users to handle every return
 * statement by each of these functions.
 */

#pragma once
#include <cstddef>
#include <cub/cub.cuh>

namespace gunrock {
namespace algo {
namespace select {

// XXX: condense code using enum
enum select_t
{
  flagged,
  if,
  unique
}

namespace device
{

  // XXX: Possible way to condense everything in one function:
  //   template<select_t select_type = select_t::unique,
  //            typename error_t,
  //            typename storage_t,
  //            typename input_t,
  //            typename output_t,
  //            typename flag_t,
  //            typename op_t,
  //            typename select_out_t,
  //            typename int_t,
  //            typename stream_t>
  //   error_t select(input_t * input,
  //                  output_t * output,
  //                  select_out_t * selected,
  //                  int_t count,
  //                  storage_t * temp_storage = NULL,
  //                  size_t & temp_storage_bytes = 0,
  //                  flag_t * flags = std::nullptr,
  //                  op_t op = std::nullptr;
  //                  stream_t stream = 0, // XXX: generalize
  //                  bool sync = false)
  //   {}

  template<typename error_t,
           typename storage_t,
           typename input_t,
           typename output_t,
           typename flag_t,
           typename select_out_t,
           typename int_t,
           typename stream_t>
  error_t select_flagged(input_t * input,
                         flag_t * flags,
                         output_t * output,
                         select_out_t * selected,
                         int_t count,
                         storage_t * temp_storage = NULL,
                         size_t & temp_storage_bytes = 0,
                         stream_t stream = 0, // XXX: generalize
                         bool sync = false)
  {
    error_t retval = cudaSuccess;

    // Determine temporary device storage requirements
    cub::DeviceSelect::Flagged(temp_storage,
                               temp_storage_bytes,
                               input,
                               flags,
                               output,
                               selected,
                               count,
                               stream,
                               sync);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run selection
    cub::DeviceSelect::Flagged(temp_storage,
                               temp_storage_bytes,
                               input,
                               flags,
                               output,
                               selected,
                               count,
                               stream,
                               sync);

    return retval;
  }

  template<typename error_t,
           typename storage_t,
           typename input_t,
           typename output_t,
           typename op_t,
           typename select_out_t,
           typename int_t,
           typename stream_t>
  error_t select_if(input_t * input,
                    output_t * output,
                    select_out_t * selected,
                    int_t count,
                    op_t op,
                    storage_t * temp_storage = NULL,
                    size_t & temp_storage_bytes = 0,
                    stream_t stream = 0, // XXX: generalize
                    bool sync = false)
  {
    error_t retval = cudaSuccess;

    // Determine temporary device storage requirements
    cub::DeviceSelect::If(temp_storage,
                          temp_storage_bytes,
                          input,
                          output,
                          selected,
                          count,
                          op,
                          stream,
                          sync);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run selection
    cub::DeviceSelect::If(temp_storage,
                          temp_storage_bytes,
                          input,
                          output,
                          selected,
                          count,
                          op,
                          stream,
                          sync);

    return retval;
  }

  template<typename error_t,
           typename storage_t,
           typename input_t,
           typename output_t,
           typename select_out_t,
           typename int_t,
           typename stream_t>
  error_t select_unique(input_t * input,
                        output_t * output,
                        select_out_t * selected,
                        int_t count,
                        storage_t * temp_storage = NULL,
                        size_t & temp_storage_bytes = 0,
                        stream_t stream = 0, // XXX: generalize
                        bool sync = false)
  {
    error_t retval = cudaSuccess;

    // Determine temporary device storage requirements
    cub::DeviceSelect::Unique(temp_storage,
                              temp_storage_bytes,
                              input,
                              output,
                              selected,
                              count,
                              stream,
                              sync);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run selection
    cub::DeviceSelect::Unique(temp_storage,
                              temp_storage_bytes,
                              input,
                              output,
                              selected,
                              count,
                              stream,
                              sync);

    return retval;
  }

} // namespace: device

} // namespace: select
} // namespace: algo
} // namespace: gunrock