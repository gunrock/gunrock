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

#include <stdio.h>
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
    bool print = true)
{
    if (error && print) {
        fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

cudaError_t GRError(
    cudaError_t error,
    std::string message,
    const char *filename,
    int line,
    bool print = true)
{
    if (error && print) {
        fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message.c_str(), error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
cudaError_t GRError(
    const char *message,
    const char *filename,
    int line,
    bool print = true)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {

        fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

cudaError_t GRError(
    std::string message,
    const char *filename,
    int line,
    bool print = true)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {

        fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message.c_str(), error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}


/**
 * Displays error message in accordance with debug mode
 */
cudaError_t GRError(
    cudaError_t error,
    bool print = true)
{
    if (error && print) {
        fprintf(stderr, "(CUDA error %d: %s)\n", error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}


/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
cudaError_t GRError(
    bool print = true)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {
        fprintf(stderr, "(CUDA error %d: %s)\n", error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}


} // namespace util
} // namespace gunrock

