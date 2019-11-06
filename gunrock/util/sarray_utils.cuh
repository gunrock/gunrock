// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sarray_utils.cuh
 *
 * @brief sparse array utilities for SparseArray1D
 */

#pragma once
#include <stdio.h>

#include <cub/cub.cuh>
#include <gunrock/util/types.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace util {

#define SARRAY_DEBUG true

enum SARRAY_METHOD {
  FLAT,
  ORDERED,
  CUCKOO_HASH,
};

enum SARRAY_STATUS {
  EMPTY,
  INITED,
  BUILT,
};

template <typename _IndexT, typename _SizeT, typename _DataT>
// SARRAY_METHOD _METHOD>
class SparseArray1D_Base {
 public:
  typedef _IndexT IndexT;
  typedef _SizeT SizeT;
  typedef _DataT DataT;
  // const SARRAY_METHOD METHOD = _METHOD;
  Array1D<SizeT, IndexT> Index;
  Array1D<SizeT, DataT> Data;

 protected:
  std::string name;
  SizeT index_range;
  SizeT num_occuplied_items;
  SARRAY_STATUS status;
  // Array1D<SizeT, IndexT> temp_Index;
  Array1D<SizeT, DataT> temp_Data;

 public:
  SparseArray1D_Base()
      : name(""), index_range(0), num_occuplied_items(0), status(EMPTY) {}

  virtual ~SparseArray1D_Base() {
#ifdef __CUDA_ARCH__
#else
    // Release();
#endif
  }

  void SetName(std::string name) {
    Index.SetName(name + "_Index");
    Data.SetName(name + "_Data");
    // temp_Index.SetName(name + "_temp_Index");
    temp_Data.SetName(name + "_temp_Data");
    this->name = name;
  }

  cudaError_t Init(std::string name, SizeT index_range,
                   unsigned int target = HOST) {
    SetName(name);
    cudaError_t retval = cudaSuccess;
    // if (retval = temp_Index.Allocate(index_range, target)) return retval;
    if (retval = temp_Data.Allocate(index_range + 1, target)) return retval;
    if ((target & HOST) == HOST) {
      for (SizeT i = 0; i < index_range + 1; i++)
        temp_Data[i] = InvalidValue<DataT>();
    }
    if ((target & DEVICE) == DEVICE) {
      MemsetKernel<<<128, 128>>>(temp_Data.GetPointer(DEVICE),
                                 InvalidValue<DataT>(), index_range + 1);
    }
    status = INITED;
    return retval;
  }

  cudaError_t Release() {
    cudaError_t retval = cudaSuccess;
    status = EMPTY;
    if (retval = Index.Release()) return retval;
    if (retval = Data.Release()) return retval;
    // if (retval = temp_Index.Release()) return retval;
    if (retval = temp_Data.Release()) return retval;
    return retval;
  }

  // virtual __host__ __device__ __forceinline__ SizeT Get_Pos(IndexT idx)
  // const;
  /*{
      extern __host__ __device__ void UnSupportedMethod();
      UnSupportedMethod();
      return InvalidValue<SizeT>();
  //#ifdef __CUDA_ARCH__
  //#else
  //#endif
  }*/

  /*virtual cudaError_t Build_(unsigned int source, unsigned int processor,
  unsigned int target)
  {
      extern void UnSupportedMethod();
      UnSupportedMethod();
      return cudaSuccess;
  }

  cudaError_t Build(unsigned int source, unsigned int processor, unsigned int
  target, DataT *data_in, SizeT index_range = -1)
  {
      cudaError_t retval = cudaSuccess;
      if (index_range == -1) index_range = this -> index_range;
      else this -> index_range = index_range;

      if (retval = temp_Data.Release()) return retval;
      if (retval = temp_Data.SetPointer(data_in, index_range, source)) return
  retval; return Build_(source, processor, target);
  }*/

  /*__host__ __device__ __forceinline__ DataT& operator[](IndexT idx)
  {
      if (status == BUILT)
      {
          SizeT pos = Get_Pos(idx);
          return Data[pos];
      }
      else //if (status == INITED)
          return temp_Data[idx];
  }

  __host__ __device__ const __forceinline__ DataT& operator[](IndexT idx) const
  {
      if (status == BUILT)
      {
          SizeT pos = Get_Pos(idx);
          return const_cast<DataT&>(Data[pos]);
      } else //if (status == INITED)
          return const_cast<DataT&>(temp_Data[idx]);
  }

  __host__ __device__ __forceinline__ DataT& operator()(SizeT pos)
  {
      return Data[pos];
  }

  __host__ __device__ const __forceinline__ DataT& operator()(SizeT pos) const
  {
      return const_cast<DataT&>(Data[pos]);
  }

  __host__ __device__ __forceinline__ DataT* operator+(const IndexT& idx)
  {
      return Data + Get_Pos(idx);
  }*/
};

template <typename _IndexT, typename _SizeT, typename _DataT>
class SparseArray1D_FLAT
    : public SparseArray1D_Base<_IndexT, _SizeT, _DataT>  //, FLAT>
{
 public:
  typedef _IndexT IndexT;
  typedef _SizeT SizeT;
  typedef _DataT DataT;

  __host__ __device__ __forceinline__ _SizeT Get_Pos(_IndexT idx) const {
#ifdef __CUDA_ARCH__
    // printf("(%4d, %4d) : accessing idx = %d, pos = %d\n",
    //    blockIdx.x, threadIdx.x, idx, idx);
#endif
    return idx;
  }

  cudaError_t Build_(unsigned int source, unsigned int processor,
                     unsigned int target) {
    cudaError_t retval = cudaSuccess;

    if (source != processor) {
      if (retval = this->temp_Data.Move(source, processor)) return retval;
    }
    if (retval = this->Data.Allocate(this->index_range, target | processor))
      return retval;
    if (processor == HOST) {
      memcpy(this->Data.GetPointer(HOST), this->temp_Data.GetPointer(HOST),
             sizeof(DataT) * this->index_range);
      if ((target & DEVICE) == DEVICE) {
        if (retval = this->Data.Move(HOST, DEVICE)) return retval;
      }
    } else if (processor == DEVICE) {
      MemsetCopyVectorKernel<<<256, 256>>>(this->Data.GetPointer(DEVICE),
                                           this->temp_Data.GetPointer(DEVICE),
                                           this->index_range);
      if ((target & HOST) == HOST) {
        if (retval = this->Data.Move(DEVICE, HOST)) return retval;
      }
    }

    if (retval = this->temp_Data.Release()) return retval;
    if ((target & processor) != processor) {
      if (retval = this->Data.Release(processor)) return retval;
    }
    this->num_occuplied_items = this->index_range;
    this->status = BUILT;
    return retval;
  }

  cudaError_t Build(std::string name, unsigned int source,
                    unsigned int processor, unsigned int target, DataT* data_in,
                    SizeT index_range = -1) {
    this->SetName(name);
    cudaError_t retval = cudaSuccess;
    if (index_range == -1)
      index_range = this->index_range;
    else
      this->index_range = index_range;

    if (retval = this->temp_Data.Release()) return retval;
    if (retval = this->temp_Data.SetPointer(data_in, index_range, source))
      return retval;
    return Build_(source, processor, target);
  }

  __host__ __device__ __forceinline__ DataT& operator[](IndexT idx) {
    if (this->status == BUILT) {
      SizeT pos = Get_Pos(idx);
      return this->Data[pos];
    } else  // if (status == INITED)
      return this->temp_Data[idx];
  }

  __host__ __device__ const __forceinline__ DataT& operator[](
      IndexT idx) const {
    if (this->status == BUILT) {
      SizeT pos = Get_Pos(idx);
      return const_cast<DataT&>(this->Data[pos]);
    } else  // if (status == INITED)
      return const_cast<DataT&>(this->temp_Data[idx]);
  }

  __host__ __device__ __forceinline__ DataT& operator()(SizeT pos) {
    return this->Data[pos];
  }

  __host__ __device__ const __forceinline__ DataT& operator()(SizeT pos) const {
    return const_cast<DataT&>(this->Data[pos]);
  }

  __host__ __device__ __forceinline__ DataT* operator+(const IndexT& idx) {
    return this->Data + Get_Pos(idx);
  }
};

template <typename IndexT, typename SizeT, typename DataT>
__global__ void Mark_Valid_Data_Kernel(DataT* d_data, SizeT* d_marker,
                                       SizeT index_range) {
  SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (i <= index_range) {
    if (isValid(d_data[i]))
      d_marker[i] = 1;
    else
      d_marker[i] = 0;
    i += STRIDE;
  }
}

template <typename IndexT, typename SizeT, typename DataT>
__global__ void Assign_Data_Index_Kernel(DataT* d_data_in, SizeT* d_marker,
                                         SizeT index_range, IndexT* d_index_out,
                                         DataT* d_data_out) {
  SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (i < index_range) {
    if (isValid(d_data_in[i])) {
      SizeT pos = d_marker[i];
      d_index_out[pos] = i;
      d_data_out[pos] = d_data_in[i];
    }
    i += STRIDE;
  }
}

template <typename _IndexT, typename _SizeT, typename _DataT>
class SparseArray1D_ORDERED
    : public SparseArray1D_Base<_IndexT, _SizeT, _DataT> {
 public:
  typedef _IndexT IndexT;
  typedef _SizeT SizeT;
  typedef _DataT DataT;

  /*template <
      typename T,
      typename SizeT>
  __host__ __device__ __forceinline__ SizeT BinarySearch(
      T item,
      T *data,
      SizeT data_length) const
  {
      SizeT lower_bound = 0, upper_bound = data_length -1;

      while (lower_bound < upper_bound)
      {
          SizeT mid_point = (lower_bound + upper_bound) >> 1;
          if (data[mid_point] < item)
              lower_bound = mid_point + 1;
          else upper_bound = mid_point;
      }

      if ((upper_bound == lower_bound) && data[upper_bound] == item)
          return upper_bound;
      else return InvalidValue<SizeT>();
  }*/

  __host__ __device__ __forceinline__ SizeT Get_Pos(IndexT idx) const {
    // return BinarySearch(idx, this -> Index.GetPointer(), this ->
    // Index.GetSize());
    SizeT lower_bound = 0, upper_bound = this->Index.GetSize() - 1;

    while (lower_bound < upper_bound) {
      SizeT mid_point = (lower_bound + upper_bound) >> 1;
      if (this->Index[mid_point] < idx)
        lower_bound = mid_point + 1;
      else
        upper_bound = mid_point;
    }

    if ((upper_bound == lower_bound) && this->Index[upper_bound] == idx)
      return upper_bound;
    else
      return InvalidValue<SizeT>();
  }

  cudaError_t Build_(unsigned int source, unsigned int processor,
                     unsigned int target) {
    cudaError_t retval = cudaSuccess;

    if (source != processor) {
      if (retval = this->temp_Data.Move(source, processor)) return retval;
    }
    if (processor == HOST) {
      this->num_occuplied_items = 0;
      for (SizeT i = 0; i < this->index_range; i++)
        if (isValid(this->temp_Data[i])) this->num_occuplied_items++;

      if (retval = this->Data.Allocate(this->num_occuplied_items,
                                       target | processor))
        return retval;
      if (retval = this->Index.Allocate(this->num_occuplied_items,
                                        target | processor))
        return retval;
      SizeT counter = 0;
      for (SizeT i = 0; i < this->index_range; i++) {
        if (!isValid(this->temp_Data[i])) continue;
        this->Data[counter] = this->temp_Data[i];
        this->Index[counter] = i;
        counter++;
      }

      if ((target & DEVICE) == DEVICE) {
        if (retval = this->Data.Move(HOST, DEVICE)) return retval;
        if (retval = this->Index.Move(HOST, DEVICE)) return retval;
      }
    } else if (processor == DEVICE) {
      Array1D<SizeT, SizeT> markers;
      Array1D<SizeT, unsigned char> cub_temp_storage;
      markers.SetName("markers");
      cub_temp_storage.SetName("cub_temp_storage");
      if (retval = markers.Allocate(this->index_range + 1, DEVICE))
        return retval;
      int num_blocks = this->index_range / 1024 + 1;
      if (num_blocks > 480) num_blocks = 480;

      Mark_Valid_Data_Kernel<IndexT, SizeT, DataT>
          <<<num_blocks, 1024>>>(this->temp_Data.GetPointer(DEVICE),
                                 markers.GetPointer(DEVICE), this->index_range);

      size_t request_bytes = 0;
      cub::DeviceScan::ExclusiveSum(cub_temp_storage.GetPointer(DEVICE),
                                    request_bytes, markers.GetPointer(DEVICE),
                                    markers.GetPointer(DEVICE),
                                    this->index_range + 1);

      if (retval = cub_temp_storage.Allocate(request_bytes, DEVICE))
        return retval;
      cub::DeviceScan::ExclusiveSum(cub_temp_storage.GetPointer(DEVICE),
                                    request_bytes, markers.GetPointer(DEVICE),
                                    markers.GetPointer(DEVICE),
                                    this->index_range + 1);

      cudaMemcpy(&(this->num_occuplied_items),
                 markers.GetPointer(DEVICE) + this->index_range, sizeof(SizeT),
                 cudaMemcpyDeviceToHost);
      if (retval = this->Data.Allocate(this->num_occuplied_items, DEVICE))
        return retval;
      if (retval = this->Index.Allocate(this->num_occuplied_items, DEVICE))
        return retval;
      Assign_Data_Index_Kernel<IndexT, SizeT, DataT><<<num_blocks, 1024>>>(
          this->temp_Data.GetPointer(DEVICE), markers.GetPointer(DEVICE),
          this->index_range, this->Index.GetPointer(DEVICE),
          this->Data.GetPointer(DEVICE));

      if (retval = markers.Release()) return retval;
      if ((target & HOST) == HOST) {
        if (retval = this->Data.Move(DEVICE, HOST)) return retval;
        if (retval = this->Index.Move(DEVICE, HOST)) return retval;
      }
    }

    if (retval = this->temp_Data.Release()) return retval;
    if ((target & processor) != processor) {
      if (retval = this->Data.Release(processor)) return retval;
      if (retval = this->Index.Release(processor)) return retval;
    }
    this->status = BUILT;
    return retval;
  }

  cudaError_t Build(std::string name, unsigned int source,
                    unsigned int processor, unsigned int target, DataT* data_in,
                    SizeT index_range = -1) {
    cudaError_t retval = cudaSuccess;
    this->SetName(name);
    if (index_range == -1)
      index_range = this->index_range;
    else
      this->index_range = index_range;

    if (retval = this->temp_Data.Release()) return retval;
    if (retval = this->temp_Data.SetPointer(data_in, index_range, source))
      return retval;
    return Build_(source, processor, target);
  }

  __host__ __device__ __forceinline__ DataT& operator[](IndexT idx) {
    if (this->status == BUILT) {
      SizeT pos = Get_Pos(idx);
      return this->Data[pos];
    } else  // if (status == INITED)
      return this->temp_Data[idx];
  }

  __host__ __device__ const __forceinline__ DataT& operator[](
      IndexT idx) const {
    if (this->status == BUILT) {
      SizeT pos = Get_Pos(idx);
      return const_cast<DataT&>(this->Data[pos]);
    } else  // if (status == INITED)
      return const_cast<DataT&>(this->temp_Data[idx]);
  }

  __host__ __device__ __forceinline__ DataT& operator()(SizeT pos) {
    return this->Data[pos];
  }

  __host__ __device__ const __forceinline__ DataT& operator()(SizeT pos) const {
    return const_cast<DataT&>(this->Data[pos]);
  }

  __host__ __device__ __forceinline__ DataT* operator+(const IndexT& idx) {
    return this->Data + Get_Pos(idx);
  }
};

enum {
  Step_Bits = 3,
  Index_Bits = 32 - Step_Bits,
  Max_Step = (1 << Step_Bits) - 1,
  Step_Mask = (1 << (32 - Step_Bits)) - 1,
};

template <typename IndexT, typename SizeT>
__host__ __device__ __forceinline__ SizeT Hash_Function(int fun_idx, IndexT idx,
                                                        SizeT table_size) {
  const SizeT Primes[5] = {7919, 7823, 7717, 7621, 7559};
  // const SizeT Multis[5] = {1, 2, 3, 5, 7};

  if (fun_idx >= Max_Step) return InvalidValue<SizeT>();
  SizeT pos;
  if ((fun_idx & 1) == 0) {
    pos = (idx + 1) * Primes[fun_idx >> 1] % table_size;
  } else
    pos = ((idx + 1) / (table_size >> (fun_idx >> 1))) % table_size;
  if (pos < 0) pos = 0 - pos;
  return pos;
}

template <typename IndexT, typename SizeT, typename DataT>
__global__ void Build_Cuckoo_Kernel(SizeT index_range, SizeT table_size,
                                    DataT* d_temp_data, IndexT* d_index,
                                    bool* d_has_conflict) {
  IndexT i = (IndexT)blockIdx.x * blockDim.x + threadIdx.x;
  const IndexT STRIDE = (IndexT)blockDim.x * gridDim.x;

  while (i < index_range && !d_has_conflict[0]) {
    if (!isValid(d_temp_data[i])) {
      i += STRIDE;
      continue;
    }

    SizeT pos = Hash_Function(0, i, table_size);
    IndexT new_index = i;
    IndexT old_index = atomicExch(d_index + pos, new_index);
    if (old_index == InvalidValue<IndexT>() || (old_index & Step_Mask) == i) {
      // printf("%d, 0, %d -> (%d)\n", i, new_index, pos);
      i += STRIDE;
      continue;
    }

    IndexT idx = old_index & Step_Mask;
    int step = (((unsigned int)old_index) >> Index_Bits) + 1;
    SizeT pos1 = Hash_Function(step, idx, table_size);
    while (pos1 != InvalidValue<SizeT>()) {
      new_index = idx | (step << Index_Bits);
      old_index = atomicExch(d_index + pos1, new_index);
      if (old_index == InvalidValue<IndexT>() ||
          (old_index & Step_Mask) == idx) {
        // printf("%d, %d, %d -> (%d)\n",
        //    idx, step, new_index, pos1);
        break;
      }
      // printf("%d, %d, %d -> (%d) %d, %d, %d\n",
      //    idx, step, new_index, pos1, old_index & Step_Mask, ((unsigned
      //    int)old_index >> Index_Bits), old_index);
      idx = old_index & Step_Mask;
      step = ((unsigned int)old_index >> Index_Bits) + 1;
      pos1 = Hash_Function(step, idx, table_size);
    }
    if (pos1 == InvalidValue<SizeT>()) {
      d_has_conflict[0] = true;
      break;
    }
    i += STRIDE;
  }
}

template <typename IndexT, typename SizeT, typename DataT>
__global__ void Assign_Cuckoo_Data_Kernel(SizeT table_size, DataT* d_temp_data,
                                          IndexT* d_index, DataT* d_data) {
  SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (i < table_size) {
    IndexT idx = d_index[i];
    if (idx != InvalidValue<IndexT>()) {
      IndexT index = idx & Step_Mask;
      d_data[i] = d_temp_data[index];
      d_index[i] = index;
      // printf("(%d, %d, %d) -> %d\n",
      //    index, d_temp_data[index], idx, i);
    }
    i += STRIDE;
  }
}

template <typename _IndexT, typename _SizeT, typename _DataT>
class SparseArray1D_CUCKOO
    : public SparseArray1D_Base<_IndexT, _SizeT, _DataT>  //, CUCKOO_HASH>
{
 public:
  typedef _IndexT IndexT;
  typedef _SizeT SizeT;
  typedef _DataT DataT;
  SizeT table_size;

  __host__ __device__ __forceinline__ SizeT Get_Pos(IndexT idx) const {
    int step = 0;
    SizeT pos = Hash_Function(step, idx, table_size);
    while (pos != InvalidValue<SizeT>()) {
      if (this->Index[pos] == idx) return pos;
      step++;
      pos = Hash_Function(step, idx, table_size);
    }
    return InvalidValue<SizeT>();
  }

  cudaError_t Build_(unsigned int source, unsigned int processor,
                     unsigned int target) {
    cudaError_t retval = cudaSuccess;

    if (source != processor) {
      if (retval = this->temp_Data.Move(source, processor)) return retval;
    }
    if (processor == HOST) {
      this->num_occuplied_items = 0;
      for (SizeT i = 0; i < this->index_range; i++)
        if (isValid(this->temp_Data[i])) this->num_occuplied_items++;

      bool has_conflict = true;
      float sizing_factor = 1.2;
      Array1D<SizeT, int> steps;
      while (has_conflict) {
        table_size = this->num_occuplied_items * sizing_factor;
        // printf("sizing_factor = %.1f, table_size = %d\n", sizing_factor,
        // table_size);
        has_conflict = false;
        if (retval = this->Index.Release()) return retval;
        if (retval = steps.Release()) return retval;
        if (retval = this->Index.Allocate(table_size, HOST)) return retval;
        if (retval = steps.Allocate(table_size, HOST)) return retval;
        for (SizeT i = 0; i < table_size; i++)
          this->Index[i] = InvalidValue<IndexT>();

        for (SizeT i = 0; i < this->index_range; i++) {
          if (!isValid(this->temp_Data[i])) continue;
          SizeT pos = Hash_Function(0, i, table_size);
          if (this->Index[pos] == InvalidValue<IndexT>() ||
              this->Index[pos] == i) {
            this->Index[pos] = i;
            steps[pos] = 0;
          } else {
            IndexT idx = this->Index[pos];
            int step = steps[pos] + 1;
            this->Index[pos] = i;
            steps[pos] = 0;
            SizeT pos1 = Hash_Function(step, idx, table_size);
            while (pos1 != InvalidValue<SizeT>()) {
              if (this->Index[pos1] == InvalidValue<IndexT>() ||
                  this->Index[pos1] == idx) {
                this->Index[pos1] = idx;
                steps[pos1] = step;
                break;
              }
              IndexT temp_idx = this->Index[pos1];
              int temp_step = steps[pos1];
              this->Index[pos1] = idx;
              steps[pos1] = step;
              idx = temp_idx;
              step = temp_step + 1;
              pos1 = Hash_Function(step, idx, table_size);
            }
            if (pos1 == InvalidValue<SizeT>()) {
              has_conflict = true;
              break;
            }
          }
        }
        if (has_conflict) sizing_factor += 0.1;
      }
      printf("sizing_factor = %.1f, table_size = %d\n", sizing_factor,
             table_size);
      if (retval = this->Data.Allocate(table_size, HOST)) return retval;
      for (SizeT i = 0; i < table_size; i++) {
        if (this->Index[i] == InvalidValue<IndexT>()) continue;
        this->Data[i] = this->temp_Data[this->Index[i]];
      }

      if ((target & DEVICE) == DEVICE) {
        if (retval = this->Data.Move(HOST, DEVICE)) return retval;
        if (retval = this->Index.Move(HOST, DEVICE)) return retval;
      }
    } else if (processor == DEVICE) {
      Array1D<SizeT, SizeT> markers;
      Array1D<SizeT, unsigned char> cub_temp_storage;
      Array1D<SizeT, SizeT> sum;
      markers.SetName("markers");
      cub_temp_storage.SetName("cub_temp_storage");
      if (retval = markers.Allocate(this->index_range + 1, DEVICE))
        return retval;
      if (retval = sum.Allocate(1, DEVICE | HOST)) return retval;
      int num_blocks = this->index_range / 1024 + 1;
      if (num_blocks > 480) num_blocks = 480;

      Mark_Valid_Data_Kernel<IndexT, SizeT, DataT>
          <<<num_blocks, 1024>>>(this->temp_Data.GetPointer(DEVICE),
                                 markers.GetPointer(DEVICE), this->index_range);

      size_t request_bytes = 0;
      cub::DeviceReduce::Sum(cub_temp_storage.GetPointer(DEVICE), request_bytes,
                             markers.GetPointer(DEVICE), sum.GetPointer(DEVICE),
                             this->index_range);

      if (retval = cub_temp_storage.Allocate(request_bytes, DEVICE))
        return retval;
      cub::DeviceReduce::Sum(cub_temp_storage.GetPointer(DEVICE), request_bytes,
                             markers.GetPointer(DEVICE), sum.GetPointer(DEVICE),
                             this->index_range);
      if (retval = sum.Move(DEVICE, HOST)) return retval;
      this->num_occuplied_items = sum[0];
      if (retval = sum.Release()) return retval;

      Array1D<SizeT, bool> has_conflict;
      if (retval = has_conflict.Allocate(1, DEVICE | HOST)) return retval;
      has_conflict[0] = true;
      float sizing_factor = 1.2;
      while (has_conflict[0]) {
        table_size = this->num_occuplied_items * sizing_factor;
        has_conflict[0] = false;
        if (retval = has_conflict.Move(HOST, DEVICE)) return retval;
        // printf("sizing_factor = %.1f, table_size = %d\n", sizing_factor,
        // table_size); if (retval = this -> Data .Allocate(table_size, DEVICE))
        // return retval;
        if (retval = this->Index.Release()) return retval;
        if (retval = this->Index.Allocate(table_size, DEVICE)) return retval;
        MemsetKernel<<<num_blocks, 1024>>>(this->Index.GetPointer(DEVICE),
                                           InvalidValue<IndexT>(), table_size);

        Build_Cuckoo_Kernel<<<num_blocks, 1024>>>(
            this->index_range, table_size, this->temp_Data.GetPointer(DEVICE),
            this->Index.GetPointer(DEVICE), has_conflict.GetPointer(DEVICE));

        if (retval = has_conflict.Move(DEVICE, HOST)) return retval;
        if (has_conflict[0]) sizing_factor += 0.1;
      }

      printf("sizing_factor = %.1f, table_size = %d\n", sizing_factor,
             table_size);
      if (retval = this->Data.Allocate(table_size, DEVICE)) return retval;
      Assign_Cuckoo_Data_Kernel<<<num_blocks, 1024>>>(
          table_size, this->temp_Data.GetPointer(DEVICE),
          this->Index.GetPointer(DEVICE), this->Data.GetPointer(DEVICE));

      if ((target & HOST) == HOST) {
        if (retval = this->Data.Move(DEVICE, HOST)) return retval;
        if (retval = this->Index.Move(DEVICE, HOST)) return retval;
      }
    }

    if (retval = this->temp_Data.Release()) return retval;
    if ((target & processor) != processor) {
      if (retval = this->Data.Release(processor)) return retval;
      if (retval = this->Index.Release(processor)) return retval;
    }
    this->status = BUILT;
    return retval;
  }

  cudaError_t Build(std::string name, unsigned int source,
                    unsigned int processor, unsigned int target, DataT* data_in,
                    SizeT index_range = -1) {
    this->SetName(name);
    cudaError_t retval = cudaSuccess;
    if (index_range == -1)
      index_range = this->index_range;
    else
      this->index_range = index_range;

    if (retval = this->temp_Data.Release()) return retval;
    if (retval = this->temp_Data.SetPointer(data_in, index_range, source))
      return retval;
    return Build_(source, processor, target);
  }

  __host__ __device__ __forceinline__ DataT& operator[](IndexT idx) {
    if (this->status == BUILT) {
      SizeT pos = Get_Pos(idx);
      return this->Data[pos];
    } else  // if (status == INITED)
      return this->temp_Data[idx];
  }

  __host__ __device__ const __forceinline__ DataT& operator[](
      IndexT idx) const {
    if (this->status == BUILT) {
      SizeT pos = Get_Pos(idx);
      return const_cast<DataT&>(this->Data[pos]);
    } else  // if (status == INITED)
      return const_cast<DataT&>(this->temp_Data[idx]);
  }

  __host__ __device__ __forceinline__ DataT& operator()(SizeT pos) {
    return this->Data[pos];
  }

  __host__ __device__ const __forceinline__ DataT& operator()(SizeT pos) const {
    return const_cast<DataT&>(this->Data[pos]);
  }

  __host__ __device__ __forceinline__ DataT* operator+(const IndexT& idx) {
    return this->Data + Get_Pos(idx);
  }
};

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
