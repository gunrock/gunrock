#pragma once

#include <exception>
#include <string>

namespace gunrock {

/**
 * @namespace error
 * Error utilities for exception handling within device and host code.
 */
namespace error {

typedef cudaError_t error_t;

/**
 * @brief Exception class for errors in device code.
 *
 */
struct exception_t : std::exception {
  std::string report;

  exception_t(error_t _status, std::string _message = "") {
    report = cudaGetErrorString(_status) + std::string("\t: ") + _message;
  }
  virtual const char* what() const noexcept { return report.c_str(); }
};

/**
 * @brief Throw an exception if the given error code is not cudaSuccess.
 *
 * @param status error_t error code (equivalent to cudaError_t).
 * @param message custom message to be appended to the error message.
 */
void throw_if_exception(error_t status, std::string message = "") {
  if (status != cudaSuccess)
    throw exception_t(status, message);
}

}  // namespace error
}  // namespace gunrock