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
        int gpu;
        cudaGetDevice(&gpu);
        fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename, line, gpu, message, error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

cudaError_t gunrock::util::GRError(
    cudaError_t error,
    std::string message,
    const char *filename,
    int line,
    bool print)
{
    if (error && print) {
        int gpu;
        cudaGetDevice(&gpu);
        fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename, line, gpu, message.c_str(), error, cudaGetErrorString(error));
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
        int gpu;
        cudaGetDevice(&gpu);
        fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename, line, gpu, message, error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

cudaError_t gunrock::util::GRError(
    std::string message,
    const char *filename,
    int line,
    bool print)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {
        int gpu;
        cudaGetDevice(&gpu);
        fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename, line, gpu, message.c_str(), error, cudaGetErrorString(error));
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
        int gpu;
        cudaGetDevice(&gpu);
        fprintf(stderr, "[@ gpu %d] (CUDA error %d: %s)\n", gpu, error, cudaGetErrorString(error));
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
        int gpu;
        cudaGetDevice(&gpu);
        fprintf(stderr, "[@ gpu %d] (CUDA error %d: %s)\n", gpu, error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

std::string gunrock::util::GetErrorString(gunrock::util::gunrockError_t error)
{
    switch (error) {
    case gunrock::util::GR_UNSUPPORTED_INPUT_DATA:
        return "unsupported input data";
        default:
        return "unknown error";
    }
}
gunrock::util::gunrockError_t gunrock::util::GRError(
    gunrock::util::gunrockError_t error,
    std::string message,
    const char *filename,
    int line,
    bool print)
{
    if (error && print) {
        int gpu;
        cudaGetDevice(&gpu);
        fprintf(stderr, "[%s, %d @ gpu %d] %s Gunrock error: %s.\n", filename, line, gpu, message.c_str(), GetErrorString(error).c_str());
        fflush(stderr);
    }
    return error;
}
