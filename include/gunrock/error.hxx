#pragma once

#include <exception>
#include <string>

namespace gunrock {

/**
 * @namespace error
 * Error utilities for exception handling within device and host code.
 */
namespace error {

typedef hipError_t error_t;

/**
 * @brief Exception class for errors in device code.
 *
 */
struct exception_t : std::exception {
  std::string report;

  exception_t(error_t _status, std::string _message = "") {
    report = hipGetErrorString(_status) + std::string("\t: ") + _message;
  }

  exception_t(std::string _message = "") { report = _message; }
  virtual const char* what() const noexcept { return report.c_str(); }
};

/**
 * @brief Throw an exception if the given error code is not hipSuccess.
 *
 * @param status error_t error code (equivalent to hipError_t).
 * @param message custom message to be appended to the error message.
 */
inline void throw_if_exception(error_t status, std::string message = "") {
  if (status != hipSuccess)
    throw exception_t(status, message);
}

inline void throw_if_exception(bool is_exception, std::string message = "") {
  if (is_exception)
    throw exception_t(message);
}

}  // namespace error
}  // namespace gunrock