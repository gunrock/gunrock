#pragma once

// includes: cuda-api-wrappers
#include <cuda/api_wrappers.hpp>

namespace gunrock {

// gunrock::error_t
typedef cuda::status_t error_t;

// XXX: Should this be under util or just gunrock::error?
namespace util {

/**
 * @namespace error
 * Error utilities for exception handling within device and host code. Wrapper
 * around some useful error handling features within cuda-api-wrappers @ref
 * cuda.
 */
namespace error {

typedef cuda::status::success success;

constexpr inline bool
is_success(error_t status)
{
  return cuda::is_success(status);
}

constexpr inline bool
is_failure(error_t status)
{
  return cuda::is_failure(status);
}

inline void
_throw(error_t status)
{
  return cuda::throw_if_error(status);
}

// throw() with custom message
inline void
_throw(error_t status, std::string message)
{
  return cuda::throw_if_error(status, message);
}

} // namespace error
} // namespace util
} // namespace gunrock