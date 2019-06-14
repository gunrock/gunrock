// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * error_utils.cuh
 *
 * @brief Error handling utility routines
 */

#pragma once

#include <string>

namespace gunrock {
namespace util {

enum gunrockError {
  GR_SUCCESS = 0,
  GR_UNSUPPORTED_INPUT_DATA = 1,
};

typedef enum gunrockError gunrockError_t;

void PrintMsg(const char *msg, bool to_print = true, bool new_line = true);
void PrintMsg(std::string msg, bool to_print = true, bool new_line = true);
void PrintMsg(const char *msg, int gpu_num, long long iteration, int peer,
              bool to_print = true, bool new_line = true);
void PrintMsg(std::string msg, int gpu_num, long long iteration, int peer,
              bool to_print = true, bool new_line = true);

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t GRError(cudaError_t error, const char *message,
                    const char *filename, int line, bool print = true);

cudaError_t GRError(cudaError_t error, std::string message,
                    const char *filename, int line, bool print = true);

/**
 * Checks and resets last CUDA error.  If set, displays last error message in
 * accordance with debug mode.
 */
cudaError_t GRError(const char *message, const char *filename, int line,
                    bool print = true);

cudaError_t GRError(std::string message, const char *filename, int line,
                    bool print = true);

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t GRError(cudaError_t error, bool print = true);

/**
 * Checks and resets last CUDA error.  If set, displays last error message in
 * accordance with debug mode.
 */
cudaError_t GRError(bool print = true);

std::string GetErrorString(gunrockError_t error);

/**
 * Displays Gunrock specific error message in accordance with debug mode
 */
gunrockError_t GRError(gunrockError_t error, std::string message,
                       const char *filename, int line, bool print = true);

}  // namespace util
}  // namespace gunrock

#define GUARD_CU(cuda_call)                                                   \
  {                                                                           \
    retval = gunrock::util::GRError(cuda_call, "error encountered", __FILE__, \
                                    __LINE__);                                \
    if (retval) return retval;                                                \
  }

#define GUARD_CU2(cuda_call, message)                                        \
  {                                                                          \
    retval = gunrock::util::GRError(cuda_call, message, __FILE__, __LINE__); \
    if (retval) return retval;                                               \
  }
