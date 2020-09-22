#pragma once

#include <exception>

namespace gunrock {

/**
 * @namespace error
 * Error utilities for exception handling within device and host code.
 */
namespace error {

typedef cudaError_t error_t;

struct exception_t : std::exception {
  error_t status;

  exception_t(error_t status_) : status(status_) { }
  virtual const char* what() const noexcept { 
    return cudaGetErrorString(status); 
  }
};

} // namespace error
} // namespace gunrock