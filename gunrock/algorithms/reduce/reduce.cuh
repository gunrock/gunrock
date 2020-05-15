/**
 * @file reduce.cuh
 *
 * @brief
 *
 * @todo error handling within methods, remove return error_t it adds
 * unnecessary work for users to handle every return statement by each of these
 * functions.
 *
 * Maybe the file-ext needs to be .hxx instead of .cuh for faster compilation.
 *
 */

#pragma once
#include <cub/cub.cuh>
#include <moderngpu/kernel_reduce.hxx>

namespace gunrock {

/**
 * @namespace algo
 * A collection of parallel algorithms/primitives, often used as building blocks
 * for much more sophisticated problems. These algorithms include reduce, sort,
 * select, scan, etc.
 */
namespace algo {

/**
 * @namespace reduce
 * Namespace for reduce algorithm, includes reduction on host and device, and on
 * different hierarchy of the device (blocks, warps, threads).
 */
namespace reduce {

/**
 * @namespace host
 * Namespace for host (CPU, system) related functions. These maybe serial or
 * parallel using openmp.
 */
namespace host {
// XXX: CPU reduce
} // namespace: host

/**
 * @namespace device Namespace for device (GPU) related functions.
 */
namespace device {

template<typename error_t,
         typename input_t,
         typename output_t,
         typename int_t,
         typename op_t>
error_t
reduce(input_t* input,
       output_t* output,
       int_t count,
       mgpu::context_t& context,          // XXX: generalize
       op_t op = mgpu::plus_t<input_t>(), // XXX: generalize
       bool sync = false)
{
  error_t status = util::error::success;

  // XXX: Experiment with these values and choose the best ones.
  typedef mgpu::launch_box_t<mgpu::arch_30_cta<256, 8>,
                             mgpu::arch_35_cta<256, 8>,
                             mgpu::arch_50_cta<256, 8>,
                             mgpu::arch_60_cta<256, 8>,
                             mgpu::arch_70_cta<256, 8>,
                             mgpu::arch_75_cta<256, 8>>
    launch_t;

  mgpu::reduce<launch_t>(input, count, output, op, context);

  if (sync) {
    // XXX: cuda sync
  }

  return status;
}

template<typename error_t,
         typename input_t,
         typename output_t,
         typename offsets_t,
         typename type_t,
         typename int_t,
         typename op_t>
error_t
reduce_segments(input_t input,
                output_t output,
                int_t count,
                offsets_t* segments,
                int_t num_segments,
                type_t init,
                mgpu::context_t& context,          // XXX: generalize
                op_t op = mgpu::plus_t<input_t>(), // XXX: generalize
                bool sync = false)
{

  // XXX: Experiment with these values and choose the best ones.
  typedef mgpu::launch_box_t<mgpu::arch_30_cta<128, 11, 8>,
                             mgpu::arch_35_cta<128, 7, 5>,
                             mgpu::arch_50_cta<128, 11, 8>,
                             mgpu::arch_60_cta<128, 11, 8>,
                             mgpu::arch_70_cta<128, 11, 8>,
                             mgpu::arch_75_cta<128, 11, 8>>
    launch_t;

  error_t status = util::error::success;
  mgpu::segreduce<launch_t>(
    input, count, segments, num_segments, output, op, init, context);

  if (sync) {
    // XXX: cuda sync
  }

  return status;
}

namespace block {
// XXX: block-wide reduce
namespace warp {
// XXX: warp-wide reduce
namespace thread {
// XXX: thread-wide reduce
} // namespace: thread
} // namespace: warp
} // namespace: block
} // namespace device

} // namespace: reduce
} // namespace: algo
} // namespace: gunrock