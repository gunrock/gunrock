/**
 * @file device.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @date 2020-10-06
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once
namespace gunrock {
namespace gcuda {

typedef int device_id_t;

namespace device {

void set(gcuda::device_id_t device) {
  hipSetDevice(device);
}

}  // namespace device
}  // namespace gcuda
}  // namespace gunrock