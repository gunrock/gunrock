#pragma once
#include <cub/cub.cuh>
#include <moderngpu/kernel_scan.hxx>

namespace gunrock {
namespace algo {
namespace scan {

enum scan_t
{
  inclusive,
  exclusive
}

namespace device
{

  namespace block {
  namespace warp {}
  }
}

} // namespace: scan
} // namespace: algo
} // namespace: gunrock