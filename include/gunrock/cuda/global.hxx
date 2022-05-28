/**
 * @file global.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-04-05
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once
namespace gunrock {
namespace gcuda {

typedef int thread_idx_t;

namespace thread {
namespace global {
namespace id {
__device__ __forceinline__ int x() {
  return threadIdx.x + (blockDim.x * blockIdx.x);
}

__device__ __forceinline__ int y() {
  return threadIdx.y + (blockDim.y * blockIdx.y);
}

__device__ __forceinline__ int z() {
  return threadIdx.z + (blockDim.z * blockIdx.z);
}
}  // namespace id
}  // namespace global

namespace local {
namespace id {
__device__ __forceinline__ int x() {
  return threadIdx.x;
}

__device__ __forceinline__ int y() {
  return threadIdx.y;
}

__device__ __forceinline__ int z() {
  return threadIdx.z;
}
}  // namespace id
}  // namespace local
}  // namespace thread

namespace block {
namespace id {
__device__ __forceinline__ int x() {
  return blockIdx.x;
}

__device__ __forceinline__ int y() {
  return blockIdx.y;
}

__device__ __forceinline__ int z() {
  return blockIdx.z;
}
}  // namespace id

namespace size {
__device__ __forceinline__ int x() {
  return blockDim.x;
}

__device__ __forceinline__ int y() {
  return blockDim.y;
}

__device__ __forceinline__ int z() {
  return blockDim.z;
}

__device__ __forceinline__ int total() {
  return x() * y() * z();
}
}  // namespace size
}  // namespace block

namespace grid {
namespace size {
__device__ __forceinline__ int x() {
  return gridDim.x;
}

__device__ __forceinline__ int y() {
  return gridDim.y;
}

__device__ __forceinline__ int z() {
  return gridDim.z;
}

__device__ __forceinline__ int total() {
  return x() * y() * z();
}
}  // namespace size
}  // namespace grid

}  // namespace gcuda
}  // namespace gunrock