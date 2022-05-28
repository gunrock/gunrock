/**
 * @file context.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/cuda/device.hxx>
#include <gunrock/cuda/device_properties.hxx>
#include <gunrock/cuda/event_management.hxx>
#include <gunrock/cuda/stream_management.hxx>
#include <gunrock/cuda/function.hxx>

#include <gunrock/error.hxx>
#include <gunrock/util/timer.hxx>

#include <gunrock/container/array.hxx>
#include <gunrock/container/vector.hxx>

#include <moderngpu/context.hxx>
#include <thrust/execution_policy.h>

namespace gunrock {
namespace gcuda {

template <int dummy_arg>
__global__ void dummy_k() {}

struct context_t {
  context_t() = default;

  // Disable copy ctor and assignment operator. We don't want to let the
  // user copy only a slice.
  context_t(const context_t& rhs) = delete;
  context_t& operator=(const context_t& rhs) = delete;

  virtual const gcuda::device_properties_t& props() const = 0;
  virtual void print_properties() = 0;
  virtual gcuda::compute_capability_t ptx_version() const = 0;
  virtual gcuda::stream_t stream() = 0;
  virtual mgpu::standard_context_t* mgpu() = 0;

  // cudaStreamSynchronize or cudaDeviceSynchronize for stream 0.
  virtual void synchronize() = 0;
  virtual gcuda::event_t event() = 0;
  virtual util::timer_t& timer() = 0;
};  // struct context_t

class standard_context_t : public context_t {
 protected:
  gcuda::device_properties_t _props;
  gcuda::compute_capability_t _ptx_version;

  gcuda::device_id_t _ordinal;
  gcuda::stream_t _stream;
  gcuda::event_t _event;

  /**
   * @todo Find out how to use a shared_ptr<> without printing the GPU debug
   * information. Currently, we are not releasing this pointer, which causes a
   * memory leak. Fix this later.
   */
  mgpu::standard_context_t* _mgpu_context;

  util::timer_t _timer;

  // Making this a template argument means we won't generate an instance
  // of dummy_k for each translation unit.
  template <int dummy_arg = 0>
  void init() {
    gcuda::function_attributes_t attr;
    error::error_t status = cudaFuncGetAttributes(&attr, dummy_k<0>);
    error::throw_if_exception(status);
    _ptx_version = gcuda::make_compute_capability(attr.ptxVersion);

    cudaSetDevice(_ordinal);
    cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&_event, cudaEventDisableTiming);
    cudaGetDeviceProperties(&_props, _ordinal);

    _mgpu_context = new mgpu::standard_context_t(false, _stream);
  }

 public:
  standard_context_t(gcuda::device_id_t device = 0)
      : context_t(), _ordinal(device), _mgpu_context(nullptr) {
    init();
  }

  standard_context_t(cudaStream_t stream, gcuda::device_id_t device = 0)
      : context_t(), _ordinal(device), _mgpu_context(nullptr), _stream(stream) {
    init();
  }

  ~standard_context_t() { cudaEventDestroy(_event); }

  virtual const gcuda::device_properties_t& props() const override {
    return _props;
  }

  virtual void print_properties() override {
    gcuda::device::set(_ordinal);
    gcuda::properties::print(_props);
  }

  virtual gcuda::compute_capability_t ptx_version() const override {
    return _ptx_version;
  }

  virtual gcuda::stream_t stream() override { return _stream; }
  virtual mgpu::standard_context_t* mgpu() override { return _mgpu_context; }

  virtual void synchronize() override {
    error::error_t status =
        _stream ? cudaStreamSynchronize(_stream) : cudaDeviceSynchronize();
    error::throw_if_exception(status);
  }

  virtual gcuda::event_t event() override { return _event; }

  virtual util::timer_t& timer() override { return _timer; }

  virtual gcuda::device_id_t ordinal() { return _ordinal; }

  auto execution_policy() {
    return thrust::cuda::par_nosync.on(this->stream());
  }

};  // class standard_context_t

class multi_context_t {
 public:
  thrust::host_vector<standard_context_t*> contexts;
  thrust::host_vector<gcuda::device_id_t> devices;
  static constexpr std::size_t MAX_NUMBER_OF_GPUS = 1024;

  // Multiple devices.
  multi_context_t(thrust::host_vector<gcuda::device_id_t> _devices)
      : devices(_devices) {
    for (auto& device : devices) {
      standard_context_t* device_context = new standard_context_t(device);
      contexts.push_back(device_context);
    }
  }

  // Multiple devices with a user-provided stream
  multi_context_t(thrust::host_vector<gcuda::device_id_t> _devices,
                  cudaStream_t _stream)
      : devices(_devices) {
    for (auto& device : devices) {
      standard_context_t* device_context =
          new standard_context_t(_stream, device);
      contexts.push_back(device_context);
    }
  }

  // Single device.
  multi_context_t(gcuda::device_id_t _device) : devices(1, _device) {
    for (auto& device : devices) {
      standard_context_t* device_context = new standard_context_t(device);
      contexts.push_back(device_context);
    }
  }

  // Single device with a user-provided stream
  multi_context_t(gcuda::device_id_t _device, cudaStream_t _stream)
      : devices(1, _device) {
    for (auto& device : devices) {
      standard_context_t* device_context =
          new standard_context_t(_stream, device);
      contexts.push_back(device_context);
    }
  }
  ~multi_context_t() {}

  auto get_context(gcuda::device_id_t device) {
    auto contexts_ptr = contexts.data();
    return contexts_ptr[device];
  }

  auto size() { return contexts.size(); }

  void enable_peer_access() {
    int num_gpus = size();
    for (int i = 0; i < num_gpus; i++) {
      auto ctx = get_context(i);
      cudaSetDevice(ctx->ordinal());

      for (int j = 0; j < num_gpus; j++) {
        if (i == j)
          continue;

        auto ctx_peer = get_context(j);
        cudaDeviceEnablePeerAccess(ctx_peer->ordinal(), 0);
      }
    }

    auto ctx0 = get_context(0);
    cudaSetDevice(ctx0->ordinal());
  }
};  // class multi_context_t

}  // namespace gcuda
}  // namespace gunrock
