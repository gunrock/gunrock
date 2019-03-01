// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * multi_scan.cuh
 *
 * @brief Multi Scan that splict and scan on array
 */

#pragma once

#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace util {
namespace scan {

template <typename _SizeT, int Block_N>
__device__ __forceinline__ void ScanLoop(_SizeT* s_Buffer, _SizeT* Sum,
                                         _SizeT Sum_Offset) {
  _SizeT Step = 1;
#pragma unrool
  for (int i = 0; i < Block_N; i++) {
    _SizeT k = threadIdx.x * Step * 2 + Step - 1;
    if (k + Step < blockDim.x * 2) s_Buffer[k + Step] += s_Buffer[k];
    Step *= 2;
    __syncthreads();
  }  // for i
  if (threadIdx.x == blockDim.x - 1) {
    if (Sum_Offset != -1) Sum[Sum_Offset] = s_Buffer[blockDim.x * 2 - 1];
    s_Buffer[blockDim.x * 2 - 1] = 0;
  }  // if
  __syncthreads();

  Step /= 2;
#pragma unrool
  for (int i = Block_N - 1; i >= 0; i--) {
    _SizeT k = threadIdx.x * Step * 2 + Step - 1;
    if (k + Step < blockDim.x * 2) {
      _SizeT t = s_Buffer[k];
      s_Buffer[k] = s_Buffer[k + Step];
      s_Buffer[k + Step] += t;
    }
    Step /= 2;
    __syncthreads();
  }  // for i
}

template <typename _VertexId, typename _SizeT, int Block_N>
__global__ void Step0(const _SizeT N, const _SizeT M, const _SizeT N_Next,
                      const _VertexId* const Select, const int* const Splict,
                      _SizeT* Buffer, _SizeT* Sum) {
  extern __shared__ _SizeT s_Buffer[];
  int Splict0 = -1, Splict1 = -1;
  _SizeT x =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * 2 + threadIdx.x;

  if (x - threadIdx.x >= N) return;
  if (x < N)
    if (Select[x] != 0) Splict0 = Splict[x];
  if (x + blockDim.x < N)
    if (Select[x + blockDim.x] != 0) Splict1 = Splict[x + blockDim.x];

  for (int y = 0; y < M; y++) {
    if (y == Splict0)
      s_Buffer[threadIdx.x] = 1;
    else
      s_Buffer[threadIdx.x] = 0;
    if (y == Splict1)
      s_Buffer[threadIdx.x + blockDim.x] = 1;
    else
      s_Buffer[threadIdx.x + blockDim.x] = 0;
    __syncthreads();

    if (x / blockDim.x / 2 < N_Next)
      ScanLoop<_SizeT, Block_N>(s_Buffer, Sum, y * N_Next + x / blockDim.x / 2);
    else
      ScanLoop<_SizeT, Block_N>(s_Buffer, Sum, -1);

    if (y == Splict0) Buffer[x] = s_Buffer[threadIdx.x];
    if (y == Splict1)
      Buffer[x + blockDim.x] = s_Buffer[threadIdx.x + blockDim.x];
  }  // for y
}  // Step0

template <typename _VertexId, typename _SizeT, int Block_N>
__global__ void Step0b(const _SizeT N, const _SizeT M, const _SizeT N_Next,
                       const _VertexId* const Keys, const int* const Splict,
                       _SizeT* Buffer, _SizeT* Sum) {
  extern __shared__ _SizeT s_Buffer[];
  int Splict0 = -1, Splict1 = -1;
  _SizeT x =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * 2 + threadIdx.x;

  if (x - threadIdx.x >= N) return;
  if (x < N) Splict0 = Splict[Keys[x]];
  if (x + blockDim.x < N) Splict1 = Splict[Keys[x + blockDim.x]];

  for (int y = 0; y < M; y++) {
    if (y == Splict0)
      s_Buffer[threadIdx.x] = 1;
    else
      s_Buffer[threadIdx.x] = 0;
    if (y == Splict1)
      s_Buffer[threadIdx.x + blockDim.x] = 1;
    else
      s_Buffer[threadIdx.x + blockDim.x] = 0;
    __syncthreads();

    if (x / blockDim.x / 2 < N_Next)
      ScanLoop<_SizeT, Block_N>(s_Buffer, Sum, y * N_Next + x / blockDim.x / 2);
    else
      ScanLoop<_SizeT, Block_N>(s_Buffer, Sum, -1);

    if (y == Splict0) Buffer[x] = s_Buffer[threadIdx.x];
    if (y == Splict1)
      Buffer[x + blockDim.x] = s_Buffer[threadIdx.x + blockDim.x];
  }
}

template <typename _VertexId, typename _SizeT, int Block_N>
__global__ void Step0d(const _SizeT N, const _SizeT M, const _SizeT N_Next,
                       const _VertexId* const Keys, const _SizeT* const Offsets,
                       const int* const Splict, _SizeT* Buffer, _SizeT* Sum) {
  extern __shared__ _SizeT s_Buffer[];
  _VertexId K0 = 0, K1 = 1;
  bool mark0 = false, mark1 = false;
  _SizeT x =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * 2 + threadIdx.x;

  if (x - threadIdx.x >= N) return;
  if (x < N) K0 = Keys[x];
  if (x + blockDim.x < N) K1 = Keys[x + blockDim.x];

  for (int y = 0; y < M; y++) {
    s_Buffer[threadIdx.x] = 0;
    mark0 = false;
    s_Buffer[threadIdx.x + blockDim.x] = 0;
    mark1 = false;
    if (x < N) {
      for (_SizeT i = Offsets[K0]; i < Offsets[K0 + 1]; i++)
        if (y == Splict[i]) {
          s_Buffer[threadIdx.x] = 1;
          mark0 = true;
          break;
        }
    }
    if (x + blockDim.x < N) {
      for (_SizeT i = Offsets[K1]; i < Offsets[K1 + 1]; i++)
        if (y == Splict[i]) {
          s_Buffer[threadIdx.x + blockDim.x] = 1;
          mark1 = true;
          break;
        }
    }
    __syncthreads();

    if (x / blockDim.x / 2 < N_Next)
      ScanLoop<_SizeT, Block_N>(s_Buffer, Sum, y * N_Next + x / blockDim.x / 2);
    else
      ScanLoop<_SizeT, Block_N>(s_Buffer, Sum, -1);

    if (mark0) Buffer[x + y * N] = s_Buffer[threadIdx.x];
    if (mark1)
      Buffer[x + blockDim.x + y * N] = s_Buffer[threadIdx.x + blockDim.x];
  }
}

template <typename _SizeT, int Block_N>
__global__ void Step1(const _SizeT N, _SizeT* Buffer, _SizeT* Sum) {
  extern __shared__ _SizeT s_Buffer[];
  _SizeT y = blockIdx.y * blockDim.y + threadIdx.y;
  _SizeT x = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  if (x >= N)
    s_Buffer[threadIdx.x] = 0;
  else
    s_Buffer[threadIdx.x] = Buffer[y * N + x];
  if (x + blockDim.x >= N)
    s_Buffer[threadIdx.x + blockDim.x] = 0;
  else
    s_Buffer[threadIdx.x + blockDim.x] = Buffer[y * N + x + blockDim.x];
  __syncthreads();

  ScanLoop<_SizeT, Block_N>(s_Buffer, Sum, y * gridDim.x + blockIdx.x);

  if (x < N) Buffer[y * N + x] = s_Buffer[threadIdx.x];
  if (x + blockDim.x < N)
    Buffer[y * N + x + blockDim.x] = s_Buffer[threadIdx.x + blockDim.x];
}  // Step1

template <typename _SizeT>
__global__ void Step2(const _SizeT N, const _SizeT* Sum, _SizeT* Buffer) {
  _SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
  _SizeT y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < N) Buffer[y * N + x] += Sum[y * gridDim.x + blockIdx.x];
}  // Step2

template <typename _SizeT>
__global__ void Step2_5(_SizeT* Length, _SizeT* Offset, _SizeT Num_Rows) {
  Offset[0] = 0;
  for (int i = 0; i < Num_Rows; i++) Offset[i + 1] = Offset[i] + Length[i];
}

template <typename _VertexId, typename _SizeT, bool EXCLUSIVE>
__global__ void Step3(const _SizeT N, const _SizeT N_Next,
                      const _VertexId* const Select, const int* const Splict,
                      const _SizeT* const Offset, const _SizeT* const Sum,
                      const _SizeT* const Buffer, _SizeT* Result) {
  _SizeT x_Next = blockIdx.x + blockIdx.y * gridDim.x;
  _SizeT x = x_Next * blockDim.x + threadIdx.x;

  if (x >= N) return;
  if (Select[x] == 0) {
    Result[x] = -1;
    return;
  }

  _SizeT r = Buffer[x] + Offset[Splict[x]];
  if (x_Next > 0) r += Sum[Splict[x] * N_Next + x_Next];
  if (!EXCLUSIVE) r += 1;
  Result[x] = r;
}  // Step3

template <typename _VertexId, typename _SizeT, bool EXCLUSIVE>
__global__ void Step3b(const _SizeT N, const _SizeT N_Next,
                       const _SizeT num_associate, const _VertexId* const Key,
                       const int* const Splict,
                       const _VertexId* const Convertion,
                       const _SizeT* const Offset, const _SizeT* const Sum,
                       const _SizeT* const Buffer, _VertexId* Result,
                       _VertexId** associate_in, _VertexId** associate_out) {
  _SizeT x_Next = blockIdx.x + blockIdx.y * gridDim.x;
  _SizeT x = x_Next * blockDim.x + threadIdx.x;

  if (x >= N) return;
  _VertexId key = Key[x];
  _SizeT tOffset = Offset[1];
  _SizeT splict = Splict[key];
  _SizeT r = Buffer[x] + Offset[splict];
  if (x_Next > 0) r += Sum[splict * N_Next + x_Next];
  if (!EXCLUSIVE) r += 1;
  Result[r] = Convertion[key];

  if (splict > 0)
    for (int i = 0; i < num_associate; i++) {
      associate_out[i][r - tOffset] = associate_in[i][key];
    }
}

template <typename _VertexId, typename _SizeT, typename _Value, bool EXCLUSIVE>
__global__ void Step3c(
    const _SizeT N, const _SizeT N_Next, const _SizeT num_vertex_associate,
    const _SizeT num_value__associate, const _VertexId* const Key,
    const int* const Splict, const _VertexId* const Convertion,
    const _SizeT* const Offset, const _SizeT* const Sum,
    const _SizeT* const Buffer, _VertexId* Result,
    _VertexId** vertex_associate_in, _VertexId** vertex_associate_out,
    _Value** value__associate_in, _Value** value__associate_out) {
  _SizeT x_Next = blockIdx.x + blockIdx.y * gridDim.x;
  _SizeT x = x_Next * blockDim.x + threadIdx.x;

  if (x >= N) return;
  _VertexId key = Key[x];
  _SizeT tOffset = Offset[1];
  _SizeT splict = Splict[key];
  _SizeT r = Buffer[x] + Offset[splict];
  if (x_Next > 0) r += Sum[splict * N_Next + x_Next];
  if (!EXCLUSIVE) r += 1;
  Result[r] = Convertion[key];

  if (splict > 0) {
    for (int i = 0; i < num_vertex_associate; i++)
      vertex_associate_out[i][r - tOffset] = vertex_associate_in[i][key];
    for (int i = 0; i < num_value__associate; i++)
      value__associate_out[i][r - tOffset] = value__associate_in[i][key];
  }
}

template <typename _VertexId, typename _SizeT, typename _Value, bool EXCLUSIVE,
          _SizeT num_vertex_associate, _SizeT num_value__associate>
__global__ void Step3c(const _SizeT N, const _SizeT N_Next,
                       // const _SizeT            num_vertex_associate,
                       // const _SizeT            num_value__associate,
                       const _VertexId* const Key, const int* const Splict,
                       const _VertexId* const Convertion,
                       const _SizeT* const Offset, const _SizeT* const Sum,
                       const _SizeT* const Buffer, _VertexId* Result,
                       _VertexId** vertex_associate_in,
                       _VertexId** vertex_associate_out,
                       _Value** value__associate_in,
                       _Value** value__associate_out) {
  _SizeT x_Next = blockIdx.x + blockIdx.y * gridDim.x;
  _SizeT x = x_Next * blockDim.x + threadIdx.x;

  if (x >= N) return;
  _VertexId key = Key[x];
  _SizeT tOffset = Offset[1];
  _SizeT splict = Splict[key];
  _SizeT r = Buffer[x] + Offset[splict];
  if (x_Next > 0) r += Sum[splict * N_Next + x_Next];
  if (!EXCLUSIVE) r += 1;
  Result[r] = Convertion[key];

  if (splict > 0) {
#pragma unroll
    for (int i = 0; i < num_vertex_associate; i++)
      vertex_associate_out[i][r - tOffset] = vertex_associate_in[i][key];
#pragma unroll
    for (int i = 0; i < num_value__associate; i++)
      value__associate_out[i][r - tOffset] = value__associate_in[i][key];
  }
}

template <typename _VertexId, typename _SizeT, typename _Value, bool EXCLUSIVE,
          _SizeT num_vertex_associate, _SizeT num_value__associate>
__global__ void Step3d(const _SizeT N, const _SizeT N_Next,
                       // const _SizeT            num_vertex_associate,
                       // const _SizeT            num_value__associate,
                       const _VertexId* const Key, const int* const Splict,
                       const _VertexId* const Convertion,
                       const _SizeT* const In_Offset,
                       const _SizeT* const Out_Offset, const _SizeT* const Sum,
                       const _SizeT* const Buffer, _VertexId* Result,
                       _VertexId** vertex_associate_in,
                       _VertexId** vertex_associate_out,
                       _Value** value__associate_in,
                       _Value** value__associate_out) {
  _SizeT x_Next = blockIdx.x + blockIdx.y * gridDim.x;
  _SizeT x = x_Next * blockDim.x + threadIdx.x;

  if (x >= N) return;
  _VertexId key = Key[x];
  _SizeT tOffset = Out_Offset[1];

  for (_SizeT i = In_Offset[key]; i < In_Offset[key + 1]; i++) {
    _SizeT splict = Splict[i];
    _SizeT r = Buffer[x + splict * N] + Out_Offset[splict];
    if (x_Next > 0) r += Sum[splict * N_Next + x_Next];
    if (!EXCLUSIVE) r += 1;
    Result[r] = Convertion[i];

    if (splict > 0) {
#pragma unroll
      for (int j = 0; j < num_vertex_associate; j++)
        vertex_associate_out[j][r - tOffset] = vertex_associate_in[j][key];
#pragma unroll
      for (int j = 0; j < num_value__associate; j++)
        value__associate_out[j][r - tOffset] = value__associate_in[j][key];
    }
  }
}

template <typename VertexId,       // Type of the select array
          typename SizeT,          // Type of counters
          bool EXCLUSIVE = true,   // Whether or not this is an exclusive scan
          SizeT BLOCK_SIZE = 256,  // Size of element to process by a block
          SizeT BLOCK_N = 8,       // log2(BLOCKSIZE)
          typename Value = VertexId>  // Type of values
struct MultiScan {
  SizeT* History_Size;
  SizeT** d_Buffer;
  SizeT* h_Offset1;
  SizeT* d_Offset1;
  SizeT Max_Elements, Max_Rows;

  MultiScan() {
    Max_Elements = 0;
    Max_Rows = 0;
    History_Size = NULL;
    d_Buffer = NULL;
    h_Offset1 = NULL;
    d_Offset1 = NULL;
  }

  __host__ void Scan(
      const SizeT Num_Elements, const SizeT Num_Rows,
      const VertexId* const d_Select,  // Selection flag, 1 = Selected
      const int* const d_Splict,       // Spliction mark
      const SizeT* const d_Offset,     // Offset of each sub-array
      SizeT* d_Length,                 // Length of each sub-array
      SizeT* d_Result)                 // The scan result
  {
    SizeT* History_Size = new SizeT[40];
    SizeT** d_Buffer = new SizeT*[40];
    SizeT Current_Size = Num_Elements;
    int Current_Level = 0;
    dim3 Block_Size, Grid_Size;

    for (int i = 0; i < 40; i++) d_Buffer[i] = NULL;
    d_Buffer[0] = d_Result;
    History_Size[0] = Current_Size;
    History_Size[1] = Current_Size / BLOCK_SIZE;
    if ((History_Size[0] % BLOCK_SIZE) != 0) History_Size[1]++;
    util::GRError(
        cudaMalloc(&(d_Buffer[1]), sizeof(SizeT) * History_Size[1] * Num_Rows),
        "cudaMalloc d_Buffer[1] failed", __FILE__, __LINE__);

    while (Current_Size > 1) {
      Block_Size = dim3(BLOCK_SIZE / 2, 1, 1);
      if (Current_Level == 0) {
        Grid_Size = dim3(History_Size[1] / 32, 32, 1);
        if ((History_Size[1] % 32) != 0) Grid_Size.x++;
        Step0<VertexId, SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE>>>(
                Current_Size, Num_Rows, History_Size[1], d_Select, d_Splict,
                d_Buffer[0], d_Buffer[1]);
        cudaDeviceSynchronize();
        util::GRError("Step0 failed", __FILE__, __LINE__);
      } else {
        Grid_Size = dim3(History_Size[Current_Level + 1], Num_Rows, 1);
        Step1<SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE>>>(
                Current_Size, d_Buffer[Current_Level],
                d_Buffer[Current_Level + 1]);
        cudaDeviceSynchronize();
        util::GRError("Step1 failed", __FILE__, __LINE__);
      }

      Current_Level++;
      Current_Size = History_Size[Current_Level];
      if (Current_Size > 1) {
        History_Size[Current_Level + 1] = Current_Size / BLOCK_SIZE;
        if ((Current_Size % BLOCK_SIZE) != 0) History_Size[Current_Level + 1]++;
        util::GRError(
            cudaMalloc(
                &(d_Buffer[Current_Level + 1]),
                sizeof(SizeT) * History_Size[Current_Level + 1] * Num_Rows),
            "cudaMalloc d_Buffer failed", __FILE__, __LINE__);
      }
    }  // while Current_Size>1

    util::GRError(
        cudaMemcpy(d_Length, d_Buffer[Current_Level], sizeof(SizeT) * Num_Rows,
                   cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Length failed", __FILE__, __LINE__);
    Current_Level--;
    while (Current_Level > 1) {
      Block_Size = dim3(BLOCK_SIZE, 1, 1);
      Grid_Size = dim3(History_Size[Current_Level], Num_Rows, 1);
      Step2<SizeT><<<Grid_Size, Block_Size>>>(History_Size[Current_Level - 1],
                                              d_Buffer[Current_Level],
                                              d_Buffer[Current_Level - 1]);
      cudaDeviceSynchronize();
      util::GRError("Step2 failed", __FILE__, __LINE__);
      Current_Level--;
    }  // while Current_Level>1

    Block_Size = dim3(BLOCK_SIZE, 1, 1);
    Grid_Size = dim3(History_Size[1] / 32, 32, 1);
    if ((History_Size[1] % 32) != 0) Grid_Size.x++;
    Step3<VertexId, SizeT, EXCLUSIVE><<<Grid_Size, Block_Size>>>(
        Num_Elements, History_Size[1], d_Select, d_Splict, d_Offset,
        d_Buffer[1], d_Buffer[0], d_Result);
    cudaDeviceSynchronize();
    util::GRError("Step3 failed", __FILE__, __LINE__);

    for (int i = 1; i < 40; i++)
      if (d_Buffer[i] != NULL) {
        util::GRError(cudaFree(d_Buffer[i]), "cudaFree d_Buffer failed",
                      __FILE__, __LINE__);
        d_Buffer[i] = NULL;
      }
    delete[] d_Buffer;
    d_Buffer = NULL;
    delete[] History_Size;
    History_Size = NULL;
  }  // Scan

  __host__ void Scan_with_Keys(const SizeT Num_Elements, const SizeT Num_Rows,
                               const SizeT Num_Associate,
                               const VertexId* const d_Keys, VertexId* d_Result,
                               const int* const d_Splict,  // Spliction mark
                               const VertexId* const d_Convertion,
                               SizeT* d_Length,  // Length of each sub-array
                               VertexId** d_Associate_in,
                               VertexId** d_Associate_out)  // The scan result
  {
    if (Num_Elements <= 0) return;
    SizeT* History_Size = new SizeT[40];
    SizeT** d_Buffer = new SizeT*[40];
    SizeT Current_Size = Num_Elements;
    int Current_Level = 0;
    dim3 Block_Size, Grid_Size;
    SizeT* h_Offset1 = new SizeT[Num_Rows + 1];
    SizeT* d_Offset1;

    util::GRError(
        cudaMalloc((void**)&d_Offset1, sizeof(SizeT) * (Num_Rows + 1)),
        "cudaMalloc d_Offset1 failed", __FILE__, __LINE__);

    for (int i = 0; i < 40; i++) d_Buffer[i] = NULL;
    History_Size[0] = Current_Size;
    History_Size[1] = Current_Size / BLOCK_SIZE;
    if ((History_Size[0] % BLOCK_SIZE) != 0) History_Size[1]++;
    util::GRError(cudaMalloc(&(d_Buffer[0]), sizeof(SizeT) * History_Size[0]),
                  "cudaMalloc d_Buffer[0] failed", __FILE__, __LINE__);
    util::GRError(
        cudaMalloc(&(d_Buffer[1]), sizeof(SizeT) * History_Size[1] * Num_Rows),
        "cudaMalloc d_Buffer[1] failed", __FILE__, __LINE__);

    while (Current_Size > 1 || Current_Level == 0) {
      Block_Size = dim3(BLOCK_SIZE / 2, 1, 1);
      if (Current_Level == 0) {
        Grid_Size = dim3(History_Size[1] / 32, 32, 1);
        if ((History_Size[1] % 32) != 0) Grid_Size.x++;
        Step0b<VertexId, SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE>>>(
                History_Size[0], Num_Rows, History_Size[1], d_Keys, d_Splict,
                d_Buffer[0], d_Buffer[1]);
        cudaDeviceSynchronize();
        util::GRError("Step0b failed", __FILE__, __LINE__);
      } else {
        Grid_Size = dim3(History_Size[Current_Level + 1], Num_Rows, 1);
        Step1<SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE>>>(
                Current_Size, d_Buffer[Current_Level],
                d_Buffer[Current_Level + 1]);
        cudaDeviceSynchronize();
        util::GRError("Step1 failed", __FILE__, __LINE__);
      }

      Current_Level++;
      Current_Size = History_Size[Current_Level];
      if (Current_Size > 1) {
        History_Size[Current_Level + 1] = Current_Size / BLOCK_SIZE;
        if ((Current_Size % BLOCK_SIZE) != 0) History_Size[Current_Level + 1]++;
        util::GRError(
            cudaMalloc(
                &(d_Buffer[Current_Level + 1]),
                sizeof(SizeT) * History_Size[Current_Level + 1] * Num_Rows),
            "cudaMalloc d_Buffer failed", __FILE__, __LINE__);
      }
    }  // while Current_Size>1

    util::GRError(
        cudaMemcpy(d_Length, d_Buffer[Current_Level], sizeof(SizeT) * Num_Rows,
                   cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Length failed", __FILE__, __LINE__);
    Current_Level--;
    while (Current_Level > 1) {
      Block_Size = dim3(BLOCK_SIZE, 1, 1);
      Grid_Size = dim3(History_Size[Current_Level], Num_Rows, 1);
      Step2<SizeT><<<Grid_Size, Block_Size>>>(History_Size[Current_Level - 1],
                                              d_Buffer[Current_Level],
                                              d_Buffer[Current_Level - 1]);
      cudaDeviceSynchronize();
      util::GRError("Step2 failed", __FILE__, __LINE__);
      Current_Level--;
    }  // while Current_Level>1

    Block_Size = dim3(BLOCK_SIZE, 1, 1);
    Grid_Size = dim3(History_Size[1] / 32, 32, 1);
    h_Offset1[0] = 0;
    util::GRError(cudaMemcpy(&(h_Offset1[1]), d_Length,
                             sizeof(SizeT) * Num_Rows, cudaMemcpyDeviceToHost),
                  "cudaMemcpy h_Offset1 failed", __FILE__, __LINE__);
    for (int i = 0; i < Num_Rows; i++) h_Offset1[i + 1] += h_Offset1[i];
    util::GRError(
        cudaMemcpy(d_Offset1, h_Offset1, sizeof(SizeT) * (Num_Rows + 1),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy d_Offset1 failed", __FILE__, __LINE__);

    if ((History_Size[1] % 32) != 0) Grid_Size.x++;
    Step3b<VertexId, SizeT, EXCLUSIVE><<<Grid_Size, Block_Size>>>(
        Num_Elements, History_Size[1], Num_Associate, d_Keys, d_Splict,
        d_Convertion, d_Offset1, d_Buffer[1], d_Buffer[0], d_Result,
        d_Associate_in, d_Associate_out);
    cudaDeviceSynchronize();
    util::GRError("Step3b failed", __FILE__, __LINE__);

    for (int i = 0; i < 40; i++)
      if (d_Buffer[i] != NULL) {
        util::GRError(cudaFree(d_Buffer[i]), "cudaFree d_Buffer failed",
                      __FILE__, __LINE__);
        d_Buffer[i] = NULL;
      }
    util::GRError(cudaFree(d_Offset1), "cudaFree d_Offset1 failed", __FILE__,
                  __LINE__);
    d_Offset1 = NULL;
    delete[] h_Offset1;
    h_Offset1 = NULL;
    delete[] d_Buffer;
    d_Buffer = NULL;
    delete[] History_Size;
    History_Size = NULL;
  }  // Scan_with_Keys

  __host__ void Scan_with_dKeys(
      const SizeT Num_Elements, const SizeT Num_Rows,
      const SizeT Num_Vertex_Associate, const SizeT Num_Value__Associate,
      const VertexId* const d_Keys, VertexId* d_Result,
      const int* const d_Splict,  // Spliction mark
      const VertexId* const d_Convertion,
      SizeT* d_Length,  // Length of each sub-array
      VertexId** d_Vertex_Associate_in, VertexId** d_Vertex_Associate_out,
      Value** d_Value__Associate_in,
      Value** d_Value__Associate_out)  // The scan result
  {
    if (Num_Elements <= 0) return;
    SizeT* History_Size = new SizeT[40];
    SizeT** d_Buffer = new SizeT*[40];
    SizeT Current_Size = Num_Elements;
    int Current_Level = 0;
    dim3 Block_Size, Grid_Size;
    SizeT* h_Offset1 = new SizeT[Num_Rows + 1];
    SizeT* d_Offset1;

    util::GRError(
        cudaMalloc((void**)&d_Offset1, sizeof(SizeT) * (Num_Rows + 1)),
        "cudaMalloc d_Offset1 failed", __FILE__, __LINE__);

    for (int i = 0; i < 40; i++) d_Buffer[i] = NULL;
    History_Size[0] = Current_Size;
    History_Size[1] = Current_Size / BLOCK_SIZE;
    if ((History_Size[0] % BLOCK_SIZE) != 0) History_Size[1]++;
    util::GRError(cudaMalloc(&(d_Buffer[0]), sizeof(SizeT) * History_Size[0]),
                  "cudaMalloc d_Buffer[0] failed", __FILE__, __LINE__);
    util::GRError(
        cudaMalloc(&(d_Buffer[1]), sizeof(SizeT) * History_Size[1] * Num_Rows),
        "cudaMalloc d_Buffer[1] failed", __FILE__, __LINE__);

    while (Current_Size > 1 || Current_Level == 0) {
      Block_Size = dim3(BLOCK_SIZE / 2, 1, 1);
      if (Current_Level == 0) {
        Grid_Size = dim3(History_Size[1] / 32, 32, 1);
        if ((History_Size[1] % 32) != 0) Grid_Size.x++;
        Step0b<VertexId, SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE>>>(
                History_Size[0], Num_Rows, History_Size[1], d_Keys, d_Splict,
                d_Buffer[0], d_Buffer[1]);
        cudaDeviceSynchronize();
        util::GRError("Step0b failed", __FILE__, __LINE__);
      } else {
        Grid_Size = dim3(History_Size[Current_Level + 1], Num_Rows, 1);
        Step1<SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE>>>(
                Current_Size, d_Buffer[Current_Level],
                d_Buffer[Current_Level + 1]);
        cudaDeviceSynchronize();
        util::GRError("Step1 failed", __FILE__, __LINE__);
      }

      Current_Level++;
      Current_Size = History_Size[Current_Level];
      if (Current_Size > 1) {
        History_Size[Current_Level + 1] = Current_Size / BLOCK_SIZE;
        if ((Current_Size % BLOCK_SIZE) != 0) History_Size[Current_Level + 1]++;
        util::GRError(
            cudaMalloc(
                &(d_Buffer[Current_Level + 1]),
                sizeof(SizeT) * History_Size[Current_Level + 1] * Num_Rows),
            "cudaMalloc d_Buffer failed", __FILE__, __LINE__);
      }
    }  // while Current_Size>1

    util::GRError(
        cudaMemcpy(d_Length, d_Buffer[Current_Level], sizeof(SizeT) * Num_Rows,
                   cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Length failed", __FILE__, __LINE__);
    Current_Level--;
    while (Current_Level > 1) {
      Block_Size = dim3(BLOCK_SIZE, 1, 1);
      Grid_Size = dim3(History_Size[Current_Level], Num_Rows, 1);
      Step2<SizeT><<<Grid_Size, Block_Size>>>(History_Size[Current_Level - 1],
                                              d_Buffer[Current_Level],
                                              d_Buffer[Current_Level - 1]);
      cudaDeviceSynchronize();
      util::GRError("Step2 failed", __FILE__, __LINE__);
      Current_Level--;
    }  // while Current_Level>1

    Block_Size = dim3(BLOCK_SIZE, 1, 1);
    Grid_Size = dim3(History_Size[1] / 32, 32, 1);
    h_Offset1[0] = 0;
    util::GRError(cudaMemcpy(&(h_Offset1[1]), d_Length,
                             sizeof(SizeT) * Num_Rows, cudaMemcpyDeviceToHost),
                  "cudaMemcpy h_Offset1 failed", __FILE__, __LINE__);
    for (int i = 0; i < Num_Rows; i++) h_Offset1[i + 1] += h_Offset1[i];
    util::GRError(
        cudaMemcpy(d_Offset1, h_Offset1, sizeof(SizeT) * (Num_Rows + 1),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy d_Offset1 failed", __FILE__, __LINE__);

    if ((History_Size[1] % 32) != 0) Grid_Size.x++;
    Step3c<VertexId, SizeT, Value, EXCLUSIVE><<<Grid_Size, Block_Size>>>(
        Num_Elements, History_Size[1], Num_Vertex_Associate,
        Num_Value__Associate, d_Keys, d_Splict, d_Convertion, d_Offset1,
        d_Buffer[1], d_Buffer[0], d_Result, d_Vertex_Associate_in,
        d_Vertex_Associate_out, d_Value__Associate_in, d_Value__Associate_out);
    cudaDeviceSynchronize();
    util::GRError("Step3b failed", __FILE__, __LINE__);

    for (int i = 0; i < 40; i++)
      if (d_Buffer[i] != NULL) {
        util::GRError(cudaFree(d_Buffer[i]), "cudaFree d_Buffer failed",
                      __FILE__, __LINE__);
        d_Buffer[i] = NULL;
      }
    util::GRError(cudaFree(d_Offset1), "cudaFree d_Offset1 failed", __FILE__,
                  __LINE__);
    d_Offset1 = NULL;
    delete[] h_Offset1;
    h_Offset1 = NULL;
    delete[] d_Buffer;
    d_Buffer = NULL;
    delete[] History_Size;
    History_Size = NULL;
  }  // Scan_with_dKeys

  //   template < SizeT            Num_Vertex_Associate,
  //              SizeT            Num_Value__Associate>
  __host__ void Init(const SizeT Max_Elements, const SizeT Max_Rows)
  // const SizeT            Num_Vertex_Associate,
  // const SizeT            Num_Value__Associate,
  // const VertexId*  const d_Keys,
  //      VertexId*        d_Result,
  // const int*       const d_Splict,    // Spliction mark
  // const VertexId*  const d_Convertion,
  //      SizeT*           d_Length,    // Length of each sub-array
  //      VertexId**       d_Vertex_Associate_in,
  //      VertexId**       d_Vertex_Associate_out,
  //      Value**          d_Value__Associate_in,
  //      Value**          d_Value__Associate_out)    // The scan result
  {
    this->Max_Elements = Max_Elements;
    this->Max_Rows = Max_Rows;
    History_Size = new SizeT[40];
    d_Buffer = new SizeT*[40];
    SizeT Current_Size = Max_Elements;
    int Current_Level = 0;
    dim3 Block_Size, Grid_Size;
    h_Offset1 = new SizeT[Max_Rows + 1];

    util::GRError(
        cudaMalloc((void**)&d_Offset1, sizeof(SizeT) * (Max_Rows + 1)),
        "cudaMalloc d_Offset1 failed", __FILE__, __LINE__);

    for (int i = 0; i < 40; i++) d_Buffer[i] = NULL;
    History_Size[0] = Current_Size;
    History_Size[1] = Current_Size / BLOCK_SIZE;
    if ((History_Size[0] % BLOCK_SIZE) != 0) History_Size[1]++;
    util::GRError(cudaMalloc(&(d_Buffer[0]), sizeof(SizeT) * History_Size[0]),
                  "cudaMalloc d_Buffer[0] failed", __FILE__, __LINE__);
    util::GRError(
        cudaMalloc(&(d_Buffer[1]), sizeof(SizeT) * History_Size[1] * Max_Rows),
        "cudaMalloc d_Buffer[1] failed", __FILE__, __LINE__);

    while (Current_Size > 1 || Current_Level == 0) {
      Current_Level++;
      Current_Size = History_Size[Current_Level];
      if (Current_Size > 1) {
        History_Size[Current_Level + 1] = Current_Size / BLOCK_SIZE;
        if ((Current_Size % BLOCK_SIZE) != 0) History_Size[Current_Level + 1]++;
        util::GRError(
            cudaMalloc(
                &(d_Buffer[Current_Level + 1]),
                sizeof(SizeT) * History_Size[Current_Level + 1] * Max_Rows),
            "cudaMalloc d_Buffer failed", __FILE__, __LINE__);
      }
    }  // while Current_Size>1
  }

  void Release() {
    for (int i = 0; i < 40; i++)
      if (d_Buffer[i] != NULL) {
        util::GRError(cudaFree(d_Buffer[i]), "cudaFree d_Buffer failed",
                      __FILE__, __LINE__);
        d_Buffer[i] = NULL;
      }
    util::GRError(cudaFree(d_Offset1), "cudaFree d_Offset1 failed", __FILE__,
                  __LINE__);
    d_Offset1 = NULL;
    delete[] h_Offset1;
    h_Offset1 = NULL;
    delete[] d_Buffer;
    d_Buffer = NULL;
    delete[] History_Size;
    History_Size = NULL;
  }  // Scan_with_dKeys

  template <SizeT Num_Vertex_Associate, SizeT Num_Value__Associate>
  __host__ void Scan_with_dKeys2(const SizeT Num_Elements, const SizeT Num_Rows,
                                 // const SizeT            Num_Vertex_Associate,
                                 // const SizeT            Num_Value__Associate,
                                 const VertexId* const d_Keys,
                                 VertexId* d_Result,
                                 const int* const d_Splict,  // Spliction mark
                                 const VertexId* const d_Convertion,
                                 SizeT* d_Length,  // Length of each sub-array
                                 VertexId** d_Vertex_Associate_in,
                                 VertexId** d_Vertex_Associate_out,
                                 Value** d_Value__Associate_in,
                                 Value** d_Value__Associate_out,
                                 cudaStream_t stream = 0)  // The scan result
  {
    if (Num_Elements <= 0) {
      util::MemsetKernel<<<128, 1, 0, stream>>>(d_Length, 0, Num_Rows);
      cudaStreamSynchronize(stream);
      return;
    }
    // SizeT *History_Size = new SizeT[40];
    // SizeT **d_Buffer    = new SizeT*[40];
    SizeT Current_Size = Num_Elements;
    int Current_Level = 0;
    dim3 Block_Size, Grid_Size;
    // SizeT *h_Offset1    = new SizeT[Num_Rows+1];
    // SizeT *d_Offset1;

    if (Num_Elements > Max_Elements || Num_Rows > Max_Rows) {
      printf("Scanner expended: %d,%d -> %d,%d \n", Max_Elements, Max_Rows,
             Num_Elements, Num_Rows);
      fflush(stdout);
      Release();
      Init(Num_Elements, Num_Rows);
    }
    // util::GRError(cudaMalloc((void**)&d_Offset1, sizeof(SizeT)*(Num_Rows+1)),
    // "cudaMalloc d_Offset1 failed", __FILE__, __LINE__);

    // for (int i=0;i<40;i++) d_Buffer[i]=NULL;
    History_Size[0] = Current_Size;
    History_Size[1] = Current_Size / BLOCK_SIZE;
    if ((History_Size[0] % BLOCK_SIZE) != 0) History_Size[1]++;
    // util::GRError(cudaMalloc(&(d_Buffer[0]), sizeof(SizeT) *
    // History_Size[0]),
    //      "cudaMalloc d_Buffer[0] failed", __FILE__, __LINE__);
    // util::GRError(cudaMalloc(&(d_Buffer[1]), sizeof(SizeT) * History_Size[1]
    // * Num_Rows),
    //      "cudaMalloc d_Buffer[1] failed", __FILE__, __LINE__);

    while (Current_Size > 1 || Current_Level == 0) {
      Block_Size = dim3(BLOCK_SIZE / 2, 1, 1);
      if (Current_Level == 0) {
        Grid_Size = dim3(History_Size[1] / 32, 32, 1);
        if ((History_Size[1] % 32) != 0) Grid_Size.x++;
        Step0b<VertexId, SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE, stream>>>(
                History_Size[0], Num_Rows, History_Size[1], d_Keys, d_Splict,
                d_Buffer[0], d_Buffer[1]);
        // cudaDeviceSynchronize();
        // util::GRError("Step0b failed", __FILE__, __LINE__);
      } else {
        Grid_Size = dim3(History_Size[Current_Level + 1], Num_Rows, 1);
        Step1<SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE, stream>>>(
                Current_Size, d_Buffer[Current_Level],
                d_Buffer[Current_Level + 1]);
        // cudaDeviceSynchronize();
        // util::GRError("Step1 failed", __FILE__, __LINE__);
      }

      Current_Level++;
      Current_Size = History_Size[Current_Level];
      if (Current_Size > 1) {
        History_Size[Current_Level + 1] = Current_Size / BLOCK_SIZE;
        if ((Current_Size % BLOCK_SIZE) != 0) History_Size[Current_Level + 1]++;
        // util::GRError(cudaMalloc(&(d_Buffer[Current_Level+1]),
        //    sizeof(SizeT)*History_Size[Current_Level+1]*Num_Rows),
        //    "cudaMalloc d_Buffer failed", __FILE__, __LINE__);
      }
    }  // while Current_Size>1

    // util::GRError(cudaMemcpy(d_Length, d_Buffer[Current_Level], sizeof(SizeT)
    // * Num_Rows, cudaMemcpyDeviceToDevice),
    //      "cudaMemcpy d_Length failed", __FILE__, __LINE__);
    MemsetCopyVectorKernel<<<128, 1, 0, stream>>>(
        d_Length, d_Buffer[Current_Level], Num_Rows);

    Current_Level--;
    while (Current_Level > 1) {
      Block_Size = dim3(BLOCK_SIZE, 1, 1);
      Grid_Size = dim3(History_Size[Current_Level], Num_Rows, 1);
      Step2<SizeT><<<Grid_Size, Block_Size, 0, stream>>>(
          History_Size[Current_Level - 1], d_Buffer[Current_Level],
          d_Buffer[Current_Level - 1]);
      // cudaDeviceSynchronize();
      // util::GRError("Step2 failed", __FILE__, __LINE__);
      Current_Level--;
    }  // while Current_Level>1

    Step2_5<<<1, 1, 0, stream>>>(d_Length, d_Offset1, Num_Rows);

    Block_Size = dim3(BLOCK_SIZE, 1, 1);
    Grid_Size = dim3(History_Size[1] / 32, 32, 1);
    /*h_Offset1[0]=0;
    util::GRError(cudaMemcpy(&(h_Offset1[1]), d_Length, sizeof(SizeT)*Num_Rows,
    cudaMemcpyDeviceToHost), "cudaMemcpy h_Offset1 failed", __FILE__, __LINE__);
    for (int i=0;i<Num_Rows;i++) h_Offset1[i+1]+=h_Offset1[i];
    util::GRError(cudaMemcpy(d_Offset1, h_Offset1, sizeof(SizeT)*(Num_Rows+1),
    cudaMemcpyHostToDevice), "cudaMemcpy d_Offset1 failed", __FILE__, __LINE__);
    */

    if ((History_Size[1] % 32) != 0) Grid_Size.x++;
    Step3c<VertexId, SizeT, Value, EXCLUSIVE, Num_Vertex_Associate,
           Num_Value__Associate><<<Grid_Size, Block_Size, 0, stream>>>(
        Num_Elements, History_Size[1],
        // Num_Vertex_Associate,
        // Num_Value__Associate,
        d_Keys, d_Splict, d_Convertion, d_Offset1, d_Buffer[1], d_Buffer[0],
        d_Result, d_Vertex_Associate_in, d_Vertex_Associate_out,
        d_Value__Associate_in, d_Value__Associate_out);
    cudaStreamSynchronize(stream);
    // cudaDeviceSynchronize();
    // util::GRError("Step3b failed", __FILE__, __LINE__);

    // for (int i=0;i<40;i++)
    // if (d_Buffer[i]!=NULL)
    //{
    //    util::GRError(cudaFree(d_Buffer[i]),
    //          "cudaFree d_Buffer failed", __FILE__, __LINE__);
    //    d_Buffer[i]=NULL;
    //}
    // util::GRError(cudaFree(d_Offset1),"cudaFree d_Offset1 failed", __FILE__,
    // __LINE__); d_Offset1=NULL; delete[] h_Offset1;    h_Offset1    = NULL;
    // delete[] d_Buffer;     d_Buffer     = NULL;
    // delete[] History_Size; History_Size = NULL;
  }  // Scan_with_dKeys

  template <SizeT Num_Vertex_Associate, SizeT Num_Value__Associate>
  __host__ void Scan_with_dKeys_Backward(
      const SizeT Num_Elements, const SizeT Num_Rows,
      // const SizeT            Num_Vertex_Associate,
      // const SizeT            Num_Value__Associate,
      const VertexId* const d_Keys, const SizeT* const d_Offset,
      VertexId* d_Result,
      const int* const d_Splict,  // Spliction mark
      const VertexId* const d_Convertion,
      SizeT* d_Length,  // Length of each sub-array
      VertexId** d_Vertex_Associate_in, VertexId** d_Vertex_Associate_out,
      Value** d_Value__Associate_in,
      Value** d_Value__Associate_out)  // The scan result
  {
    if (Num_Elements <= 0) return;
    SizeT* History_Size = new SizeT[40];
    SizeT** d_Buffer = new SizeT*[40];
    SizeT Current_Size = Num_Elements;
    int Current_Level = 0;
    dim3 Block_Size, Grid_Size;
    SizeT* h_Offset1 = new SizeT[Num_Rows + 1];
    SizeT* d_Offset1;

    util::GRError(
        cudaMalloc((void**)&d_Offset1, sizeof(SizeT) * (Num_Rows + 1)),
        "cudaMalloc d_Offset1 failed", __FILE__, __LINE__);

    for (int i = 0; i < 40; i++) d_Buffer[i] = NULL;
    History_Size[0] = Current_Size;
    History_Size[1] = Current_Size / BLOCK_SIZE;
    if ((History_Size[0] % BLOCK_SIZE) != 0) History_Size[1]++;
    util::GRError(
        cudaMalloc(&(d_Buffer[0]), sizeof(SizeT) * History_Size[0] * Num_Rows),
        "cudaMalloc d_Buffer[0] failed", __FILE__, __LINE__);
    util::GRError(
        cudaMalloc(&(d_Buffer[1]), sizeof(SizeT) * History_Size[1] * Num_Rows),
        "cudaMalloc d_Buffer[1] failed", __FILE__, __LINE__);

    while (Current_Size > 1 || Current_Level == 0) {
      Block_Size = dim3(BLOCK_SIZE / 2, 1, 1);
      if (Current_Level == 0) {
        Grid_Size = dim3(History_Size[1] / 32, 32, 1);
        if ((History_Size[1] % 32) != 0) Grid_Size.x++;
        Step0d<VertexId, SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE>>>(
                History_Size[0], Num_Rows, History_Size[1], d_Keys, d_Offset,
                d_Splict, d_Buffer[0], d_Buffer[1]);
        cudaDeviceSynchronize();
        util::GRError("Step0b failed", __FILE__, __LINE__);
      } else {
        Grid_Size = dim3(History_Size[Current_Level + 1], Num_Rows, 1);
        Step1<SizeT, BLOCK_N>
            <<<Grid_Size, Block_Size, sizeof(SizeT) * BLOCK_SIZE>>>(
                Current_Size, d_Buffer[Current_Level],
                d_Buffer[Current_Level + 1]);
        cudaDeviceSynchronize();
        util::GRError("Step1 failed", __FILE__, __LINE__);
      }

      Current_Level++;
      Current_Size = History_Size[Current_Level];
      if (Current_Size > 1) {
        History_Size[Current_Level + 1] = Current_Size / BLOCK_SIZE;
        if ((Current_Size % BLOCK_SIZE) != 0) History_Size[Current_Level + 1]++;
        util::GRError(
            cudaMalloc(
                &(d_Buffer[Current_Level + 1]),
                sizeof(SizeT) * History_Size[Current_Level + 1] * Num_Rows),
            "cudaMalloc d_Buffer failed", __FILE__, __LINE__);
      }
    }  // while Current_Size>1

    util::GRError(
        cudaMemcpy(d_Length, d_Buffer[Current_Level], sizeof(SizeT) * Num_Rows,
                   cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Length failed", __FILE__, __LINE__);
    Current_Level--;
    while (Current_Level > 1) {
      Block_Size = dim3(BLOCK_SIZE, 1, 1);
      Grid_Size = dim3(History_Size[Current_Level], Num_Rows, 1);
      Step2<SizeT><<<Grid_Size, Block_Size>>>(History_Size[Current_Level - 1],
                                              d_Buffer[Current_Level],
                                              d_Buffer[Current_Level - 1]);
      cudaDeviceSynchronize();
      util::GRError("Step2 failed", __FILE__, __LINE__);
      Current_Level--;
    }  // while Current_Level>1

    Block_Size = dim3(BLOCK_SIZE, 1, 1);
    Grid_Size = dim3(History_Size[1] / 32, 32, 1);
    h_Offset1[0] = 0;
    util::GRError(cudaMemcpy(&(h_Offset1[1]), d_Length,
                             sizeof(SizeT) * Num_Rows, cudaMemcpyDeviceToHost),
                  "cudaMemcpy h_Offset1 failed", __FILE__, __LINE__);
    for (int i = 0; i < Num_Rows; i++) h_Offset1[i + 1] += h_Offset1[i];
    util::GRError(
        cudaMemcpy(d_Offset1, h_Offset1, sizeof(SizeT) * (Num_Rows + 1),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy d_Offset1 failed", __FILE__, __LINE__);

    // for (int k=0;k<Num_Rows;k++)
    //    util::cpu_mt::PrintGPUArray<SizeT,
    //    SizeT>("Buffer1",d_Buffer[1]+k*History_Size[1],History_Size[1]);
    // util::cpu_mt::PrintGPUArray<SizeT,
    // SizeT>("Buffer0",d_Buffer[0],History_Size[0]);
    if ((History_Size[1] % 32) != 0) Grid_Size.x++;
    Step3d<VertexId, SizeT, Value, EXCLUSIVE, Num_Vertex_Associate,
           Num_Value__Associate><<<Grid_Size, Block_Size>>>(
        Num_Elements, History_Size[1],
        // Num_Vertex_Associate,
        // Num_Value__Associate,
        d_Keys, d_Splict, d_Convertion, d_Offset, d_Offset1, d_Buffer[1],
        d_Buffer[0], d_Result, d_Vertex_Associate_in, d_Vertex_Associate_out,
        d_Value__Associate_in, d_Value__Associate_out);
    cudaDeviceSynchronize();
    util::GRError("Step3b failed", __FILE__, __LINE__);

    for (int i = 0; i < 40; i++)
      if (d_Buffer[i] != NULL) {
        util::GRError(cudaFree(d_Buffer[i]), "cudaFree d_Buffer failed",
                      __FILE__, __LINE__);
        d_Buffer[i] = NULL;
      }
    util::GRError(cudaFree(d_Offset1), "cudaFree d_Offset1 failed", __FILE__,
                  __LINE__);
    d_Offset1 = NULL;
    delete[] h_Offset1;
    h_Offset1 = NULL;
    delete[] d_Buffer;
    d_Buffer = NULL;
    delete[] History_Size;
    History_Size = NULL;
  }  // Scan_with_dKeys

};  // struct MultiScan

}  // namespace scan
}  // namespace util
}  // namespace gunrock
