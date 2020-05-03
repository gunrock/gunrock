#pragma once
#include <moderngpu/kernel_sortedsearch.hxx>

namespace gunrock {
namespace algo {
namespace search {

enum bound_t
{
  upper,
  lower
}

namespace sorted
{
  namespace device {

  template<bound_t bounds = lower,
           typename needle_t,
           typename haystack_t,
           typename pos_t,
           typename int_t>
  cudaError_t SortedSearch(needle_t* needles,
                           int_t num_needles,
                           haystack_t* haystack,
                           int_t num_haystack,
                           pos_t* indices,
                           mgpu::context_t& context,
                           bool sync = false)
  {

    cudaError_t retval = cudaSuccess;

    // XXX: Experiment with these values and choose the best ones. We
    // could even choose to make them a parameter of util::SortedSearch
    // and choose based on our usage
    typedef mgpu::launch_box_t<mgpu::arch_30_cta<256, 7>,
                               mgpu::arch_35_cta<256, 7>,
                               mgpu::arch_50_cta<256, 7>,
                               mgpu::arch_60_cta<256, 7>,
                               mgpu::arch_70_cta<256, 7>,
                               mgpu::arch_75_cta<256, 7>>
      launch_t;

    if (bounds == bound_t::lower) {
      mgpu::sorted_search<mgpu::bounds_lower, launch_t>(
        needles,
        num_needles,
        haystack,
        num_haystack,
        indices,
        mgpu::less_t<needle_t>(),   // XXX: generalize
        context);

    } else { // bounds == bound_t::upper
      mgpu::sorted_search<mgpu::bounds_upper, launch_t>(
        needles,
        num_needles,
        haystack,
        num_haystack,
        indices,
        mgpu::less_t<needle_t>(),   // XXX: generalize
        context);
    }

    if (sync) {
      // XXX: implement sync
    }

    return retval;
  }

  } // namespace: device
} // namespace: binary
} // namespace: search
} // namespace: algo
} // namespace: gunrock
