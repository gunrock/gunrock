// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * error_utils.cu
 *
 * @brief Error handling utility routines
 */

#include <stdio.h>
#include <gunrock/util/error_utils.cuh> 

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t gunrock::util::GRError(
    cudaError_t error,
    const char *message,
    const char *filename,
    int line,
    bool print)
{
    if (error && print) {
        fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
cudaError_t gunrock::util::GRError(
    const char *message,
    const char *filename,
    int line,
    bool print)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {

        fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t gunrock::util::GRError(
    cudaError_t error,
    bool print)
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
cudaError_t gunrock::util::GRError(
    bool print)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {
        fprintf(stderr, "(CUDA error %d: %s)\n", error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}