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
  // error_t status;
  std::string report;

  exception_t(error_t _status, std::string _message = "") {
    report = cudaGetErrorString(_status) + std::string("\t: ") + _message;
  }
  virtual const char* what() const noexcept { return report.c_str(); }
};

// wrapper to reduce lines of code
void throw_if_exception(error_t status, std::string message = "") {
  if (status != cudaSuccess)
    throw exception_t(status, message);
}

}  // namespace error
}  // namespace gunrock