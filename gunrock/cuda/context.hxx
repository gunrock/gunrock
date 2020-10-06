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

namespace gunrock {
namespace cuda {

template <int dummy_arg>
__global__ void dummy_k() {}

struct context_t {
  context_t() = default;

  // Disable copy ctor and assignment operator. We don't want to let the
  // user copy only a slice.
  context_t(const context_t& rhs) = delete;
  context_t& operator=(const context_t& rhs) = delete;

  virtual const cuda::device_properties_t& props() const = 0;
  virtual int ptx_version() const = 0;
  virtual cuda::stream_t stream() = 0;

  // cudaStreamSynchronize or cudaDeviceSynchronize for stream 0.
  virtual void synchronize() = 0;
  virtual cuda::event_t event() = 0;
};  // struct context_t

class standard_context_t : public context_t {
 protected:
  cuda::device_properties_t _props;
  cuda::architecture_t _ptx_version;

  cuda::device_id_t _ordinal;
  cuda::stream_t _stream;
  cuda::event_t _event;

  // Making this a template argument means we won't generate an instance
  // of dummy_k for each translation unit.
  template <int dummy_arg = 0>
  void init() {
    cuda::function_attributes_t attr;
    cuda::error_t status = cudaFuncGetAttributes(&attr, dummy_k<0>);
    error::throw_if_exception(status);
    _ptx_version = attr.ptxVersion;

    cudaSetDevice(_ordinal);
    cudaStreamCreate(&_stream);
    cudaGetDeviceProperties(&_props, _ordinal);
    cudaEventCreate(&_event);
  }

 public:
  standard_context_t(cuda::device_id_t device = 0)
      : context_t(), _ordinal(device) {
    init();
  }

  ~standard_context_t() { cudaEventDestroy(_event); }

  virtual const cuda::device_properties_t& props() const override {
    return _props;
  }
  virtual cuda::architecture_t ptx_version() const override {
    return _ptx_version;
  }
  virtual cuda::stream_t stream() override { return _stream; }

  virtual void synchronize() override {
    cuda::error_t status =
        _stream ? cudaStreamSynchronize(_stream) : cudaDeviceSynchronize();
    error::throw_if_exception(status);
  }

  virtual cuda::event_t event() { return _event; }
};  // class standard_context_t

class multi_context_t {};  // class multi_context_t

}  // namespace cuda
}  // namespace gunrock