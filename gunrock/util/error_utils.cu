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

namespace gunrock {
namespace util {

void PrintMsg(const char *msg, bool to_print, bool new_line) {
  if (!to_print) return;
  printf("%s%s", msg, new_line ? "\n" : "");
  if (new_line) fflush(stdout);
}

void PrintMsg(std::string msg, bool to_print, bool new_line) {
  if (!to_print) return;
  PrintMsg(msg.c_str(), to_print, new_line);
}

void PrintMsg(const char *msg, int gpu_num, long long iteration, int peer,
              bool to_print, bool new_line) {
  if (!to_print) return;
  PrintMsg(std::to_string(gpu_num) + "\t " + std::to_string(iteration) + "\t " +
               std::to_string(peer) + "\t " + std::string(msg),
           true, new_line);
}

void PrintMsg(std::string msg, int gpu_num, long long iteration, int peer,
              bool to_print, bool new_line) {
  if (!to_print) return;
  PrintMsg(std::to_string(gpu_num) + "\t " + std::to_string(iteration) + "\t " +
               std::to_string(peer) + "\t " + msg,
           true, new_line);
}

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t GRError(cudaError_t error, const char *message,
                    const char *filename, int line, bool print) {
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename,
            line, gpu, message, error, cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

cudaError_t GRError(cudaError_t error, std::string message,
                    const char *filename, int line, bool print) {
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename,
            line, gpu, message.c_str(), error, cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

/**
 * Checks and resets last CUDA error.  If set, displays last error message in
 * accordance with debug mode.
 */
cudaError_t GRError(const char *message, const char *filename, int line,
                    bool print) {
  cudaError_t error = cudaGetLastError();
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename,
            line, gpu, message, error, cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

cudaError_t GRError(std::string message, const char *filename, int line,
                    bool print) {
  cudaError_t error = cudaGetLastError();
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename,
            line, gpu, message.c_str(), error, cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t GRError(cudaError_t error, bool print) {
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[@ gpu %d] (CUDA error %d: %s)\n", gpu, error,
            cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

/**
 * Checks and resets last CUDA error.  If set, displays last error message in
 * accordance with debug mode.
 */
cudaError_t GRError(bool print) {
  cudaError_t error = cudaGetLastError();
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[@ gpu %d] (CUDA error %d: %s)\n", gpu, error,
            cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

std::string GetErrorString(gunrock::util::gunrockError_t error) {
  switch (error) {
    case gunrock::util::GR_UNSUPPORTED_INPUT_DATA:
      return "unsupported input data";
    default:
      return "unknown error";
  }
}
gunrockError_t GRError(gunrock::util::gunrockError_t error, std::string message,
                       const char *filename, int line, bool print) {
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[%s, %d @ gpu %d] %s Gunrock error: %s.\n", filename, line,
            gpu, message.c_str(), GetErrorString(error).c_str());
    fflush(stderr);
  }
  return error;
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
