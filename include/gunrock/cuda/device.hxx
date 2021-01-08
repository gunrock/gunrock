/**
 * @file device.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-06
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once
namespace gunrock {
namespace cuda {

typedef int device_id_t;

namespace device {

void set(cuda::device_id_t device) {
  cudaSetDevice(device);
}

}  // namespace device
}  // namespace cuda
}  // namespace gunrock