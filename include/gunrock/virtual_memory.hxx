/**
 * @file virtual_memory.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-02-23
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <iostream>
#include <memory>
#include <cuda.h>
#include <gunrock/error.hxx>

namespace gunrock {
namespace memory {

/**
 * @brief
 *
 */
enum access_flags_t {
  none,        // Make the address range not accessible
  read_only,   // Make the address range read accessible
  read_write,  // Default: Make the address range read-write accessible
  MAX          // No documentation.
};

/**
 * @brief
 *
 * @tparam type_t
 */
template <typename type_t>
struct physical_memory_t {
  using allocation_handle_t = CUmemGenericAllocationHandle;
  using allocation_properties_t = CUmemAllocationProp;

  std::vector<allocation_handle_t> alloc_handle;
  allocation_properties_t prop = {};
  std::size_t granularity;
  std::size_t padded_size;
  std::size_t stripe_size;
  std::size_t size;
  unsigned long long flags;  // Not currently used in CUDA.

  std::vector<int> resident_devices;

  physical_memory_t(std::size_t _size, const std::vector<int> _resident_devices)
      : size(_size),
        resident_devices(_resident_devices),
        flags(0),
        granularity(0) {
    // Set properties of the allocation to create. The following properties will
    // create a pinned memory, local to the device (GPU).
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    // Find the minimum granularity needed for the resident devices.
    for (std::size_t idx = 0; idx < resident_devices.size(); idx++) {
      std::size_t _granularity = 0;
      // get the minnimum granularity for residentDevices[idx]
      prop.location.id = resident_devices[idx];
      cuMemGetAllocationGranularity(&_granularity, &prop,
                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM);
      if (granularity < _granularity)
        granularity = _granularity;
    }

    // Round up the size such that it can evenly split into a stripe size that
    // meets the granularity requirement. padded_size = N * GPUs * granularity,
    // since each of the piece of the allocation will be N * granularity and the
    // granularity applies to each stripe_size piece of the allocation.
    padded_size = round_up(size, resident_devices.size() * granularity);
    stripe_size = padded_size / resident_devices.size();

    // Create the backings on each GPU.
    alloc_handle.resize(resident_devices.size());
    for (std::size_t idx = 0; idx < resident_devices.size(); idx++) {
      prop.location.id = resident_devices[idx];
      cuMemCreate(&alloc_handle[idx], stripe_size, &prop, flags);
    }
  }

  ~physical_memory_t() {
    for (std::size_t idx = 0; idx < resident_devices.size(); idx++)
      cuMemRelease(alloc_handle[idx]);
  }

  static std::size_t round_up(std::size_t x, std::size_t y) {
    return ((x + y - 1) / y) * y;
  }
};

/**
 * @brief
 *
 * @tparam type_t
 */
template <typename type_t>
struct virtual_memory_t {
  type_t* ptr;               // pointer
  std::size_t size;          // padded size
  std::size_t alignment;     // alignment of reserved range
  type_t* addr;              // Fixed starting address range requested
  unsigned long long flags;  // not used within CUDA

  virtual_memory_t(std::size_t padded_size)
      : size(padded_size), alignment(0), addr(0), flags(0) {
    cuMemAddressReserve((CUdeviceptr*)&ptr, size, alignment, (CUdeviceptr)addr,
                        flags);
  }

  ~virtual_memory_t() { cuMemAddressFree((CUdeviceptr)ptr, size); }
};

/**
 * @brief striped memory mapper for multiple GPU virtual memory range. The idea
 * is to equally map the physical memory handles per GPU to equal chunks of the
 * virtual memory range. Illustration below visualizes the implementation. Note
 * that you can pass in resident and remote access flags, such that the user
 * gets control over what read-only accesses are and what can be written to
 * (defaults to read-write for all devices).
 *
 *     v-stripeSize-v                v-rounding -v
 *     +-----------------------------------------+
 *     |      D1     |      D2     |      D3     |
 *     +-----------------------------------------+
 *     ^-- dptr                      ^-- dptr + size
 *
 * @tparam type_t
 */
template <typename type_t>
class striped_memory_mapper_t {
  const virtual_memory_t<type_t>& virt;
  const physical_memory_t<type_t>& phys;

 public:
  /**
   * @brief Construct a new striped memory mapper object.
   *
   * @param virt_arg virtual memory argument; contiguous memory array.
   * @param phys_arg physical memory argument must contain a handle for each
   * device, therefore, this is really a multi-gpu physical memory struct than a
   * single physical memory handle.
   * @param mapping_devices devices to map virtual memory to.
   * @param resident_access access type for the chunk of the memory for the
   * device where the physical chunk exists (local accesses).
   * @param remote_access access type for the chunk of the memory for the remote
   * devices (remote access type).
   */
  striped_memory_mapper_t(
      const virtual_memory_t<type_t>& virt_arg,
      const physical_memory_t<type_t>& phys_arg,
      const std::vector<int>& mapping_devices,
      const access_flags_t resident_access = access_flags_t::read_write,
      const access_flags_t remote_access = access_flags_t::read_write)
      : virt(virt_arg), phys(phys_arg) {
    const size_t stripe_size = phys.stripe_size;

    for (auto& device : phys.resident_devices)
      cuMemMap((CUdeviceptr)virt.ptr + (stripe_size * device), stripe_size, 0,
               phys.alloc_handle[device], 0);

    std::vector<CUmemAccessDesc> access_descriptors(mapping_devices.size());

    for (auto& local : phys.resident_devices) {
      for (auto& remote : mapping_devices) {
        access_flags_t access;
        access_descriptors[remote].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_descriptors[remote].location.id = remote;

        // If the device being mapped to is where the physical memory resides
        // use the resident_access access flag, otherwise, use remote_access.
        if (remote == local)
          access = resident_access;
        else
          access = remote_access;

        // Set the access_descriptors flag.
        if (access == access_flags_t::none)
          access_descriptors[remote].flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
        else if (access == access_flags_t::read_only)
          access_descriptors[remote].flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        else if (access == access_flags_t::read_write)
          access_descriptors[remote].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        else
          access_descriptors[remote].flags = CU_MEM_ACCESS_FLAGS_PROT_MAX;
      }

      cuMemSetAccess((CUdeviceptr)virt.ptr + (stripe_size * local), stripe_size,
                     access_descriptors.data(), access_descriptors.size());
    }
  }

  ~striped_memory_mapper_t() { cuMemUnmap((CUdeviceptr)virt.ptr, virt.size); }

  type_t* data() { return virt.ptr; }
  std::size_t elements_per_partition() {
    return phys.stripe_size / sizeof(type_t);
  }
  std::size_t number_of_elements() { return phys.size / sizeof(type_t); }
  std::size_t padded_size() { return phys.padded_size; }
  std::size_t requested_size() { return phys.size; }

  static std::size_t round_up(std::size_t x, std::size_t y) {
    return ((x + y - 1) / y) * y;
  }
};

}  // namespace memory
}  // namespace gunrock