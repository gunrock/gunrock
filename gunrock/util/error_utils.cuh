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


/**
 * Displays error message in accordance with debug mode
 */
cudaError_t GRError(
    cudaError_t error,
    const char *message,
    const char *filename,
    int line,
    bool print = true);

cudaError_t GRError(
    cudaError_t error,
    std::string message,
    const char *filename,
    int line,
    bool print = true);

/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
cudaError_t GRError(
    const char *message,
    const char *filename,
    int line,
    bool print = true);

cudaError_t GRError(
    std::string message,
    const char *filename,
    int line,
    bool print = true);

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t GRError(
    cudaError_t error,
    bool print = true);


/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
cudaError_t GRError(
    bool print = true);
	

} // namespace util
} // namespace gunrock

