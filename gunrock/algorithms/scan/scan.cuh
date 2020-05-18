/**
 * @file scan.cuh
 *
 * @brief
 *
 * @todo error handling within methods, remove return error_t it adds
 * unnecessary work for users to handle every return statement by each of these
 * functions.
 *
 * Maybe the file-ext needs to be .hxx instead of .cuh for faster compilation.
 *
 *
 */

#pragma once
#include <cub/cub.cuh>
#include <moderngpu/cta_segscan.hxx>
#include <moderngpu/kernel_scan.hxx>

namespace gunrock {
namespace algo {

/**
 * @namespace scan
 * Namespace for scan algorithm, includes scan on host and device, and on
 * different hierarchy of the device (blocks, warps, threads).
 */
namespace scan {

enum scan_t
{
  inclusive,
  exclusive
};

namespace host
{
  // XXX: CPU scan
} // namespace: host

namespace device {

template<scan_t scan_type,
         typename error_t,
         typename input_t,
         typename output_t,
         typename reduce_t,
         typename int_t,
         typename op_t>
error_t
scan(input_t* input,
     int_t num_items,
     output_t* output,
     reduce_t* reduction,
     mgpu::context_t& context,          // XXX: generalize
     op_t op = mgpu::plus_t<input_t>(), // XXX: generalize
     bool sync = false)
{

  error_t status = util::error : success;

  // XXX: Experiment with these values and choose the best ones. We
  // could even choose to make them a parameter of util::Scan and choose
  // based on our usage
  typedef mgpu::launch_box_t<mgpu::arch_30_cta<256, 7>,
                             mgpu::arch_35_cta<256, 7>,
                             mgpu::arch_50_cta<256, 7>,
                             mgpu::arch_60_cta<256, 7>,
                             mgpu::arch_70_cta<256, 7>,
                             mgpu::arch_75_cta<256, 7>>
    launch_t;

  if (scan_type == scan_t::inclusive) {
    mgpu::scan<mgpu::scan_type_inc, launch_t>(
      input, num_items, output, op, reduction, context);
  } else { // scan_type == scan_t::exclusive
    mgpu::scan<mgpu::scan_type_exc, launch_t>(
      input, num_items, output, op, reduction, context);
  }

  if (sync) {
    // XXX: cuda sync
  }
  return status;
}

namespace block {
// XXX: block-wide scan
namespace warp {
// XXX: warp-wide scan
namespace thread {
// XXX: thread-wide scan
} // namespace: thread
} // namespace: warp
} // namespace: block
} // namespace device

} // namespace: scan
} // namespace: algo
} // namespace: gunrock