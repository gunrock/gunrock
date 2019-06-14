// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * array_utils.cuh
 *
 * @brief array utilities for Array1D
 */

#pragma once

#include <string>
#include <cstring>
#include <fstream>
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/type_limits.cuh>
#include <gunrock/util/type_enum.cuh>
#include <gunrock/util/vector_utils.cuh>
//#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace util {

//#define ENABLE_ARRAY_DEBUG

/**
 * @brief location flags
 */
using Location = uint32_t;
// enum Location : unsigned int
enum : Location {
  LOCATION_NONE = 0x00,
  // LOCATION_BASE = 0x10,
  HOST = 0x01,
  CPU = 0x01,
  DEVICE = 0x02,
  GPU = 0x02,
  DISK = 0x04,
  LOCATION_ALL = 0x0F,
  LOCATION_DEFAULT = 0x10,
};

std::string Location_to_string(Location target) {
  std::string str = "";
  if ((target & HOST) == HOST)
    str = (str == "" ? "" : " ") + std::string("HOST");
  if ((target & DEVICE) == DEVICE)
    str = (str == "" ? "" : " ") + std::string("DEVICE");
  if ((target & DISK) == DISK)
    str = (str == "" ? "" : " ") + std::string("DISK");
  if (str == "") str = "NONE";
  return str;
}

/**
 * @brief flags for arrays
 */
using ArrayFlag = uint32_t;
// enum ArrayFlag : unsigned int
enum : ArrayFlag {
  ARRAY_NONE = 0x00,
  PINNED = 0x01,
  UNIFIED = 0x02,
  STREAM = 0x04,
  MAPPED = 0x04,
};

template <typename SizeT, typename ValueT>
__global__ void Memcpy_Kernel(ValueT *d_dest, ValueT *d_src, SizeT length) {
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    d_dest[i] = d_src[i];
    i += STRIDE;
  }
}

static const Location ARRAY_DEFAULT_TARGET = DEVICE;

// Dummy array with no storage
template <typename _SizeT, typename _ValueT, ArrayFlag FLAG = ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct NullArray {
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;

  void SetName(const char *const name) {}

  cudaError_t Allocate(SizeT size, Location target = ARRAY_DEFAULT_TARGET) {
    return cudaSuccess;
  }

  __host__ __device__ __forceinline__ Location GetAllocated() {
    return LOCATION_NONE;
  }

  cudaError_t Release(Location target = LOCATION_ALL) { return cudaSuccess; }

  cudaError_t EnsureSize(SizeT size, bool keep = false,
                         cudaStream_t stream = 0) {
    return cudaSuccess;
  }

  template <typename ArrayT>
  cudaError_t Set(ArrayT &array_in,
                  SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                  Location target = LOCATION_DEFAULT, cudaStream_t stream = 0) {
    return cudaSuccess;
  }

  cudaError_t Move(Location source, Location target,
                   SizeT size = util::PreDefinedValues<SizeT>::InvalidValue,
                   SizeT offset = 0, cudaStream_t stream = 0) {
    return cudaSuccess;
  }

  cudaError_t WriteBinary(std::string filename,
                          bool ignore_file_error = false) {
    return cudaSuccess;
  }

  cudaError_t ReadBinary(std::string filename, bool ignore_file_error = false) {
    return cudaSuccess;
  }

  template <typename ArrayT_in, typename ApplyLambda>
  cudaError_t ForEach(ArrayT_in &array_in, ApplyLambda apply,
                      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                      Location target = LOCATION_DEFAULT,
                      cudaStream_t stream = 0) {
    return cudaSuccess;
  }

  template <typename ApplyLambda>
  cudaError_t ForEach(ApplyLambda apply,
                      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                      Location target = LOCATION_DEFAULT,
                      cudaStream_t stream = 0) {
    return cudaSuccess;
  }

  __host__ __device__ __forceinline__ ValueT *GetPointer(
      Location target = ARRAY_DEFAULT_TARGET) {
    return (ValueT *)NULL;
  }

  __host__ __device__ __forceinline__ ValueT *operator+(const _SizeT &offset) {
    return (ValueT *)NULL;
  }

  __host__ __device__ __forceinline__ ValueT &operator[](std::size_t idx) {
    return *((ValueT *)this);
  }

  __host__ __device__ __forceinline__ ValueT &operator[](
      std::size_t idx) const {
    return *((ValueT *)this);
  }
};

template <typename _SizeT, typename _ValueT, ArrayFlag FLAG = ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct Array1D {
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  typedef Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag> Array1DT;

 private:
  char *name;
  // std::string  file_name;
  SizeT size;
  // unsigned int flag;
  // bool         use_cuda_alloc;
  Location setted, allocated;
  ValueT *h_pointer;
  ValueT *d_pointer;

 public:
  Array1D()
      : name(NULL),
        h_pointer(NULL),
        d_pointer(NULL),
        setted(LOCATION_NONE),
        allocated(LOCATION_NONE) {
    // name.reserve(40);
    // file_name.reserve(512);
    // name      = "";
    // file_name = "";
    // h_pointer = NULL;
    // d_pointer = NULL;
    // flag      = cudaHostAllocDefault;
    // setted    = LOCATION_NONE;
    // allocated = LOCATION_NONE;
    // use_cuda_alloc = false;
    this->name = (char *)malloc(sizeof(char) * 1);
    this->name[0] = '\0';
    Init(0, LOCATION_NONE);
  }  // Array1D()

  Array1D(const char *const name)
      :  // name     (std::string(name)),
        h_pointer(NULL),
        d_pointer(NULL),
        setted(LOCATION_NONE),
        allocated(LOCATION_NONE) {
    if (name != NULL) {
      this->name = (char *)malloc(sizeof(char) * (strlen(name) + 1));
      strcpy(this->name, name);
    }
    // this->name.reserve(40);
    // file_name.reserve(512);
    // this->name= std::string(name);
    // file_name = "";
    // h_pointer = NULL;
    // d_pointer = NULL;
    // setted    = LOCATION_NONE;
    // allocated = LOCATION_NONE;
    // flag      = cudaHostAllocDefault;
    // use_cuda_alloc = false;
    Init(0, LOCATION_NONE);
  }  // Array1D(const char* const)

  /*Array1D(SizeT size, std::string name = "", unsigned int target = HOST)
  {
      this->name= name;
      //file_name = "";
      h_pointer = NULL;
      d_pointer = NULL;
      setted    = NONE;
      allocated = NONE;
      Init(size,target,use_cuda_alloc,flag);
  } // Array1D(...)*/

  __host__ __device__ virtual ~Array1D() {
#ifdef __CUDA_ARCH__
#else
    // Release();
#endif
  }  // ~Array1D()

  template <typename ApplyLambda>
  cudaError_t ForAll(ApplyLambda apply,
                     SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                     Location target = LOCATION_DEFAULT,
                     cudaStream_t stream = 0,
                     int grid_size = PreDefinedValues<int>::InvalidValue,
                     int block_size = PreDefinedValues<int>::InvalidValue);

  template <typename ArrayT_in, typename ApplyLambda>
  cudaError_t ForAll(ArrayT_in &array_in, ApplyLambda apply,
                     SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                     Location target = LOCATION_DEFAULT,
                     cudaStream_t stream = 0);

  template <typename ArrayT_in1, typename ArrayT_in2, typename ApplyLambda>
  cudaError_t ForAll(ArrayT_in1 &array_in1, ArrayT_in2 &array_in2,
                     ApplyLambda apply,
                     SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                     Location target = LOCATION_DEFAULT,
                     cudaStream_t stream = 0);

  template <typename CondLambda, typename ApplyLambda>
  cudaError_t ForAllCond(CondLambda cond, ApplyLambda apply,
                         SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                         Location target = LOCATION_DEFAULT,
                         cudaStream_t stream = 0);

  template <typename ArrayT_in, typename CondLambda, typename ApplyLambda>
  cudaError_t ForAllCond(ArrayT_in &array_in, CondLambda cond,
                         ApplyLambda apply,
                         SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                         Location target = LOCATION_DEFAULT,
                         cudaStream_t stream = 0);

  template <typename ArrayT_in1, typename ArrayT_in2, typename CondLambda,
            typename ApplyLambda>
  cudaError_t ForAllCond(ArrayT_in1 &array_in1, ArrayT_in2 &array_in2,
                         CondLambda cond, ApplyLambda apply,
                         SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                         Location target = LOCATION_DEFAULT,
                         cudaStream_t stream = 0);

  template <typename ApplyLambda>
  cudaError_t ForEach(ApplyLambda apply,
                      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                      Location target = LOCATION_DEFAULT,
                      cudaStream_t stream = 0);

  template <typename ArrayT_in, typename ApplyLambda>
  cudaError_t ForEach(ArrayT_in &array_in, ApplyLambda apply,
                      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                      Location target = LOCATION_DEFAULT,
                      cudaStream_t stream = 0);

  template <typename ArrayT_in1, typename ArrayT_in2, typename ApplyLambda>
  cudaError_t ForEach(ArrayT_in1 &array_in1, ArrayT_in2 &array_in2,
                      ApplyLambda apply,
                      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                      Location target = LOCATION_DEFAULT,
                      cudaStream_t stream = 0);

  template <typename CondLambda, typename ApplyLambda>
  cudaError_t ForEachCond(CondLambda cond, ApplyLambda apply,
                          SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                          Location target = LOCATION_DEFAULT,
                          cudaStream_t stream = 0);

  template <typename ArrayT_in, typename CondLambda, typename ApplyLambda>
  cudaError_t ForEachCond(ArrayT_in &array_in, CondLambda cond,
                          ApplyLambda apply,
                          SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                          Location target = LOCATION_DEFAULT,
                          cudaStream_t stream = 0);

  template <typename ArrayT_in1, typename ArrayT_in2, typename CondLambda,
            typename ApplyLambda>
  cudaError_t ForEachCond(ArrayT_in1 &array_in1, ArrayT_in2 &array_in2,
                          CondLambda cond, ApplyLambda apply,
                          SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                          Location target = LOCATION_DEFAULT,
                          cudaStream_t stream = 0);

  cudaError_t Init(SizeT size, Location target = ARRAY_DEFAULT_TARGET)
  // bool use_cuda_alloc = false,
  // unsigned int flag   = cudaHostAllocDefault)
  {
    cudaError_t retval = cudaSuccess;

#ifdef ENABLE_ARRAY_DEBUG
    if (size != 0 || target != LOCATION_NONE)
      PrintMsg(std::string(name) + " Init size = " + std::to_string(size) +
               ", target = " + Location_to_string(target));
#endif
    if (retval = Release()) return retval;
    setted = LOCATION_NONE;
    allocated = LOCATION_NONE;
    this->size = size;
    // this->flag = flag;
    // this->use_cuda_alloc = use_cuda_alloc;

    if (size == 0) return retval;
    retval = Allocate(size, target);
    return retval;
  }  // Init(...)

  /*void SetFilename(std::string file_name)
  {
      this->file_name=file_name;
      setted = setted | DISK;
  }*/

  cudaError_t SetName(std::string name) {
    free(this->name);
    this->name = (char *)malloc(sizeof(char) * (name.length() + 1));
    strcpy(this->name, name.c_str());
    return cudaSuccess;
  }

  cudaError_t Allocate(SizeT size, Location target = ARRAY_DEFAULT_TARGET) {
    cudaError_t retval = cudaSuccess;

    /*if (((target & HOST) == HOST) && //((target & DEVICE) == DEVICE) &&
        (use_cuda_alloc ))
    {
        if (retval = Release(HOST  )) return retval;
        //if (retval = Release(DEVICE)) return retval;
        UnSetPointer(HOST);//UnSetPointer(DEVICE);
        if ((setted    & (~(target    | DISK)) == NONE) &&
            (allocated & (~(allocated | DISK)) == NONE))
            this -> size = size;

        if (retval = GRError(cudaHostAlloc((void **)&h_pointer,
                sizeof(ValueT) * size, flag),
                std::string(name) + "cudaHostAlloc failed", __FILE__, __LINE__))
            return retval;
        //if (retval = GRError(cudaHostGetDevicePointer((void **)&d_pointer,
        //        (void *)h_pointer,0),
        //        std::string(name) + "cudaHostGetDevicePointer failed",
__FILE__, __LINE__))
        //    return retval;
        allocated = allocated | HOST  ;
        //allocated = allocated | DEVICE;
#ifdef ENABLE_ARRAY_DEBUG
        PrintMsg(std::string(name) + "\t allocated on " +
Location_to_string(HOST) +
            ", size = " + std::to_string(size) +
            ", flag = " + std::to_string(flag));
#endif
    } else {*/
    if ((target & HOST) == HOST) {
      if (retval = Release(HOST)) return retval;
      UnSetPointer(HOST);
      if ((setted & (~(target | DISK)) == LOCATION_NONE) &&
          (allocated & (~(allocated | DISK)) == LOCATION_NONE))
        this->size = size;
      if (size != 0) {
        h_pointer = new ValueT[size];
        // if (retval = util::GRError(cudaMallocHost(
        //    (void**)&(h_pointer), sizeof(ValueT) * size),
        //        std::string(name) + " allocation on "
        //        + Location_to_string(HOST) + " failed",
        //        __FILE__, __LINE__))
        //    return retval;
        // for (SizeT i = 0; i < size; i++)
        //    h_pointer[i]();

        if (h_pointer == NULL)
          return GRError(cudaErrorUnknown,
                         std::string(name) + " allocation on " +
                             Location_to_string(HOST) + " failed",
                         __FILE__, __LINE__);

        if ((FLAG & PINNED) == PINNED) {
          if (retval = util::GRError(
                  cudaHostRegister((void *)h_pointer, sizeof(ValueT) * size,
                                   cudaHostRegisterFlag),
                  std::string(name) + " cudaHostRegister failed.", __FILE__,
                  __LINE__))
            return retval;
        }
      }
      allocated = allocated | HOST;
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(std::string(name) + "\t allocated on " +
               Location_to_string(HOST) + ", length =\t " +
               std::to_string(size) + ", size =\t " +
               std::to_string((long long)size * sizeof(ValueT)) +
               " bytes, pointer =\t " + to_string(h_pointer));
#endif
    }
    //}

    if ((target & DEVICE) == DEVICE) {
      if (retval = Release(DEVICE)) return retval;
      UnSetPointer(DEVICE);
      if ((setted & (~(target | DISK)) == LOCATION_NONE) &&
          (allocated & (~(allocated | DISK)) == LOCATION_NONE))
        this->size = size;

      /*#ifdef ENABLE_ARRAY_DEBUG
                  PrintMsg(std::string(name) + "\t allocating on " +
      Location_to_string(DEVICE) +
                      ", length =\t " + std::to_string(size) +
                      ", size =\t " + std::to_string((long long)size *
      sizeof(ValueT)) + " bytes, pointer =\t " + std::to_string(d_pointer));
      #endif*/
      if (size != 0) {
        retval = GRError(
            cudaMalloc((void **)&(d_pointer), sizeof(ValueT) * size),
            std::string(name) + " cudaMalloc failed", __FILE__, __LINE__);
        if (retval) return retval;
      }
      allocated = allocated | DEVICE;
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(std::string(name) + "\t allocated on " +
               Location_to_string(DEVICE) + ", length =\t " +
               std::to_string(size) + ", size =\t " +
               std::to_string((long long)size * sizeof(ValueT)) +
               " bytes, pointer =\t " + to_string(d_pointer));
#endif
    }
    //}
    this->size = size;
    return retval;
  }  // Allocate(...)

  cudaError_t Release(Location target = LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;

    /*if (((allocated & HOST) == HOST) && ((target & DEVICE) == HOST) &&
    {
        if (retval = GRError(cudaFreeHost(h_pointer),
            std::string(name) + " cudaFreeHost failed", __FILE__, __LINE__))
            return retval;
        h_pointer = NULL;
        //d_pointer = NULL;
        allocated = allocated & (~HOST);
        //allocated = allocated & (~DEVICE);
#ifdef ENABLE_ARRAY_DEBUG
        PrintMsg(std::string(name) + " released on " + Location_to_string(HOST |
DEVICE)); #endif } else {*/
    if (((target & HOST) == HOST) && ((allocated & HOST) == HOST)) {
      if ((FLAG & PINNED) == PINNED) {
        if (retval = GRError(cudaHostUnregister((void *)h_pointer),
                             std::string(name) + " cudaHostUnregister failed",
                             __FILE__, __LINE__))
          return retval;
      }
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(std::string(name) + "\t released on " +
               Location_to_string(HOST) + ", length =\t " +
               std::to_string(size) + ", pointer =\t " + to_string(h_pointer));
#endif
      delete[] h_pointer;
      h_pointer = NULL;
      // if (retval = GRError(cudaFreeHost(h_pointer),
      //    std::string(name) + " cudaFreeHot failed", __FILE__, __LINE__))
      //    return retval;
      // h_pointer = NULL;
      allocated = allocated & (~HOST);
    } else if ((target & HOST) == HOST && (setted & HOST) == HOST) {
      UnSetPointer(HOST);
    }
    //}

    if (((target & DEVICE) == DEVICE) && ((allocated & DEVICE) == DEVICE)) {
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(std::string(name) + "\t releasing on " +
               Location_to_string(DEVICE) + ", length =\t " +
               std::to_string(size) + ", pointer = " + to_string(d_pointer));
#endif
      retval =
          GRError(cudaFree((void *)d_pointer),
                  std::string(name) + " cudaFree failed", __FILE__, __LINE__);
      if (retval) return retval;
      d_pointer = NULL;
      allocated = allocated & (~DEVICE);
    } else if ((target & DEVICE) == DEVICE && (setted & DEVICE) == DEVICE) {
      UnSetPointer(DEVICE);
    }

    if (target == LOCATION_ALL) size = 0;
    return retval;
  }  // Release(...)

  __host__ __device__ __forceinline__ SizeT GetSize() const {
    return this->size;
  }

  __host__ __device__ __forceinline__ Location GetSetted() {
    return this->setted;
  }

  __host__ __device__ __forceinline__ Location GetAllocated() {
    return this->allocated;
  }

  cudaError_t EnsureSize_(SizeT size, Location target = ARRAY_DEFAULT_TARGET) {
    cudaError_t retval = cudaSuccess;

    if (GetSize() < size) {
      retval = Allocate(size, target | allocated);
      return retval;
    } else
      size = GetSize();

    if ((target & DEVICE) != 0 && (GetPointer(DEVICE) == NULL)) {
      retval = Allocate(size, DEVICE);
      if (retval) return retval;
    }

    if ((target & HOST) != 0 && (GetPointer(HOST) == NULL)) {
      retval = Allocate(size, HOST);
      if (retval) return retval;
    }
    return retval;
  }

  cudaError_t EnsureSize(SizeT size, bool keep = false,
                         cudaStream_t stream = 0) {
#ifdef ENABLE_ARRAY_DEBUG
    PrintMsg(std::string(name) + " EnsureSize : " + std::to_string(this->size) +
             " -> " + std::to_string(size));
#endif
    if (this->size >= size) return cudaSuccess;

    /*#ifdef ENABLE_ARRAY_DEBUG
            PrintMsg(nane + " Expanding : " + std::to_string(this -> size) +
                " -> " + std::to_string(size));
    #endif*/
    if (!keep) return Allocate(size, allocated);

    Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag> temp_array;
    cudaError_t retval = cudaSuccess;
    Location org_allocated = allocated;

    temp_array.SetName("t_array");
    if (retval = temp_array.Allocate(size, allocated)) return retval;
    if ((allocated & HOST) == HOST)
      memcpy(temp_array.GetPointer(HOST), h_pointer,
             sizeof(ValueT) * this->size);
    if ((allocated & DEVICE) == DEVICE)
      Memcpy_Kernel<<<256, 256, 0, stream>>>(temp_array.GetPointer(DEVICE),
                                             d_pointer, this->size);
    if (retval = Release(HOST)) return retval;
    if (retval = Release(DEVICE)) return retval;
    if ((org_allocated & HOST) == HOST) h_pointer = temp_array.GetPointer(HOST);
    if ((org_allocated & DEVICE) == DEVICE)
      d_pointer = temp_array.GetPointer(DEVICE);
    allocated = org_allocated;
    this->size = size;
    if ((allocated & DEVICE) == DEVICE) temp_array.ForceUnSetPointer(DEVICE);
    if ((allocated & HOST) == HOST) temp_array.ForceUnSetPointer(HOST);
    return retval;
  }  // EnsureSize(...)

  cudaError_t ShrinkSize(SizeT size, bool keep = false,
                         cudaStream_t stream = 0) {
#ifdef ENABLE_ARRAY_DEBUG
    PrintMsg(std::string(name) + " ShrinkSize : " + std::to_string(this->size) +
             " -> " + std::to_string(size));
#endif
    if (this->size <= size) return cudaSuccess;

    /*#ifdef ENABLE_ARRAY_DEBUG
            PrintMsg(std::string(name) + " Shrinking : "+ std::to_string(this ->
    size) + " -> " + std::to_string(size)); #endif*/
    if (!keep) return Allocate(size, allocated);

    Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag> temp_array;
    cudaError_t retval = cudaSuccess;
    Location org_allocated = allocated;

    temp_array.SetName("t_array");
    if (retval = temp_array.Allocate(size, allocated)) return retval;
    if ((allocated & HOST) == HOST)
      memcpy(temp_array.GetPointer(HOST), h_pointer,
             sizeof(ValueT) * this->size);
    if ((allocated & DEVICE) == DEVICE)
      Memcpy_Kernel<<<256, 256, 0, stream>>>(temp_array.GetPointer(DEVICE),
                                             d_pointer, this->size);
    if (retval = Release(HOST)) return retval;
    if (retval = Release(DEVICE)) return retval;
    if ((org_allocated & HOST) == HOST) h_pointer = temp_array.GetPointer(HOST);
    if ((org_allocated & DEVICE) == DEVICE)
      d_pointer = temp_array.GetPointer(DEVICE);
    allocated = org_allocated;
    this->size = size;
    if ((allocated & DEVICE) == DEVICE) temp_array.ForceUnSetPointer(DEVICE);
    if ((allocated & HOST) == HOST) temp_array.ForceUnSetPointer(HOST);
    return retval;
  }  // ShrinkSize(...)

  __host__ __device__ __forceinline__ ValueT *GetPointer(
      Location target = ARRAY_DEFAULT_TARGET) const {
    if (target == DEVICE) {
      /*#ifdef ENABLE_ARRAY_DEBUG
                  PrintMsg(std::string(name) + " \tpointer on " +
      Location_to_string(DEVICE) + "   get = " + std::to_string(d_pointer));
      #endif*/
      return d_pointer;
    }

    if (target == HOST) {
      /*#ifdef ENABLE_ARRAY_DEBUG
                  PrintMsg(std::string(name) + " \tpointer on " +
      Location_to_string(HOST) + "   get = " + std::to_string(h_pointer));
      #endif*/
      return h_pointer;
    }

    return NULL;
  }  // GetPointer(...)

  cudaError_t SetPointer(ValueT *pointer,
                         SizeT size = PreDefinedValues<SizeT>::InvalidValue,
                         Location target = ARRAY_DEFAULT_TARGET) {
    cudaError_t retval = cudaSuccess;
    if (size == PreDefinedValues<SizeT>::InvalidValue) size = this->size;
    if (size < this->size) {
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(
          std::string(name) +
          "\t setting pointer, size too small, size = " + std::to_string(size) +
          ", this->size = " + std::to_string(this->size));
#endif
      return GRError(std::string(name) + " SetPointer size is too small",
                     __FILE__, __LINE__);
    }

    if (target == HOST) {
      if (retval = Release(HOST)) return retval;
      if ((FLAG & PINNED) == PINNED) {
        retval = util::GRError(cudaHostRegister(pointer, sizeof(ValueT) * size,
                                                cudaHostRegisterFlag),
                               std::string(name) + " cudaHostRegister failed.",
                               __FILE__, __LINE__);
        if (retval) return retval;
      }
      h_pointer = pointer;
      if (setted == LOCATION_NONE && allocated == LOCATION_NONE)
        this->size = size;
      setted = setted | HOST;
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(std::string(name) + "\t setted on " + Location_to_string(HOST) +
               ", size =\t " + std::to_string(this->size) + ", pointer =\t " +
               to_string(h_pointer) + ", setted =\t " +
               Location_to_string(setted));
#endif
    }

    if (target == DEVICE) {
      if (retval = Release(DEVICE)) return retval;
      d_pointer = pointer;
      if (setted == LOCATION_NONE && allocated == LOCATION_NONE)
        this->size = size;
      setted = setted | DEVICE;
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(
          std::string(name) + "\t setted on " + Location_to_string(DEVICE) +
          ", size =\t " + std::to_string(this->size) + ", pointer =\t " +
          to_string(d_pointer) + ", setted =\t " + Location_to_string(setted));
#endif
    }
    return retval;
  }  // SetPointer(...)

  cudaError_t ForceSetPointer(ValueT *pointer,
                              Location target = ARRAY_DEFAULT_TARGET) {
    if (target == HOST) h_pointer = pointer;
    if (target == DEVICE) d_pointer = pointer;
    return cudaSuccess;
  }

  cudaError_t ForceUnSetPointer(Location target = ARRAY_DEFAULT_TARGET) {
    cudaError_t retval = cudaSuccess;
    if ((setted & target) == target) setted = setted & (~target);
    if ((allocated & target) == target) allocated = allocated & (~target);

    if ((target & HOST) == HOST && h_pointer != NULL) {
      if ((FLAG & PINNED) == PINNED)
        if (retval = GRError(cudaHostUnregister((void *)h_pointer),
                             std::string(name) + " cudaHostUnregister failed.",
                             __FILE__, __LINE__))
          return retval;
      h_pointer = NULL;
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(std::string(name) + "\t unsetted on " +
               Location_to_string(HOST));
#endif
    }

    if ((target & DEVICE) == DEVICE && d_pointer != NULL) {
      d_pointer = NULL;
#ifdef ENABLE_ARRAY_DEBUG
      PrintMsg(std::string(name) + "\t unsetted on " +
               Location_to_string(DEVICE));
#endif
    }
    // if (target == DISK  )
    //    file_name = "";
    return retval;
  }  // UnSetPointer(...)

  void UnSetPointer(Location target = ARRAY_DEFAULT_TARGET) {
    if ((setted & target) == target) {
      ForceUnSetPointer(target);
    }
  }  // UnSetPointer(...)

  void SetMarker(int t, Location target = ARRAY_DEFAULT_TARGET, bool s = true) {
    if (t == 0) {
      if ((setted & target) != target && s)
        setted = setted | target;
      else if ((setted & target) == target && (!s))
        setted = setted & (~target);
    } else if (t == 1) {
      if ((allocated & target) != target && s)
        allocated = allocated | target;
      else if ((setted & target) == target && (!s))
        allocated = allocated & (~target);
    }
  }

  cudaError_t Move(Location source, Location target,
                   SizeT size = util::PreDefinedValues<SizeT>::InvalidValue,
                   SizeT offset = 0, cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;
    if (source == target) return retval;
    if ((source == HOST || source == DEVICE) && ((source & setted) != source) &&
        ((source & allocated) != source))
      return GRError(cudaErrorUnknown,
                     std::string(name) + " movment source is not valid",
                     __FILE__, __LINE__);
    if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
        ((target & setted) != target) && ((target & allocated) != target))
      if (retval = Allocate(this->size, target)) return retval;
    if ((target == DISK || source == DISK) && ((setted & DISK) != DISK))
      return GRError(cudaErrorUnknown, std::string(name) + " filename not set",
                     __FILE__, __LINE__);
    if (!isValid(size)) size = this->size;
    if (size > this->size)
      return GRError(cudaErrorUnknown, std::string(name) + " size is invalid",
                     __FILE__, __LINE__);
    if (size + offset > this->size)
      return GRError(cudaErrorUnknown,
                     std::string(name) + " size+offset is invalid", __FILE__,
                     __LINE__);
    if (size == 0) return retval;
#ifdef ENABLE_ARRAY_DEBUG
    PrintMsg(std::string(name) + " Moving from " + Location_to_string(source) +
             " to " + Location_to_string(target) + ", size = " +
             std::to_string(size) + ", offset = " + std::to_string(offset) +
             ", stream = " + std::to_string((long long)stream) +
             ", d_pointer = " + to_string(d_pointer) +
             ", h_pointer = " + to_string(h_pointer));
#endif

    if (source == HOST && target == DEVICE) {
      if ((FLAG & PINNED) == PINNED && stream != 0) {
        retval = GRError(cudaMemcpyAsync(d_pointer + offset, h_pointer + offset,
                                         sizeof(ValueT) * size,
                                         cudaMemcpyHostToDevice, stream),
                         std::string(name) + " cudaMemcpyAsync H2D failed",
                         __FILE__, __LINE__);
        if (retval) return retval;
      } else {
        retval = GRError(
            cudaMemcpy(d_pointer + offset, h_pointer + offset,
                       sizeof(ValueT) * size, cudaMemcpyHostToDevice),
            std::string(name) + " cudaMemcpy H2D failed", __FILE__, __LINE__);
        if (retval) return retval;
      }
    }

    else if (source == DEVICE && target == HOST) {
      if ((FLAG & PINNED) == PINNED && stream != 0) {
        // printf("%s MemcpyAsync\n");
        retval = GRError(cudaMemcpyAsync(h_pointer + offset, d_pointer + offset,
                                         sizeof(ValueT) * size,
                                         cudaMemcpyDeviceToHost, stream),
                         std::string(name) + " cudaMemcpyAsync D2H failed",
                         __FILE__, __LINE__);
        if (retval) return retval;
      } else {
        retval = GRError(
            cudaMemcpy(h_pointer + offset, d_pointer + offset,
                       sizeof(ValueT) * size, cudaMemcpyDeviceToHost),
            std::string(name) + " cudaMemcpy D2H failed", __FILE__, __LINE__);
        if (retval) return retval;
      }
    }

    /*else if (source == HOST   && target == DISK  ) {
       std::ofstream fout;
       fout.open(file_name.c_str(), std::ios::binary);
       fout.write((const char*)(h_pointer+offset),sizeof(Value)*size);
       fout.close();
    }

    else if (source == DISK   && target == HOST  ) {
       std::ifstream fin;
       fin.open(file_name.c_str(), std::ios::binary);
       fin.read((char*)(h_pointer+offset),sizeof(Value)*size);
       fin.close();
    }

    else if (source == DEVICE && target == DISK  ) {
       bool t_allocated = false;
       if (((setted & HOST) != HOST) && ((allocated & HOST) !=HOST))
       {
           if (retval = Allocate(this- > size, HOST)) return retval;
           t_allocated = true;
       }
       if (retval = Move(DEVICE, HOST, size, offset, stream)) return retval;
       if (retval = Move(HOST, DISK, size, offset)) return retval;
       if (t_allocated)
       {
           if (retval = Release(HOST)) return retval;
       }
    }

    else if (source == DISK   && target == DEVICE) {
       bool t_allocated = false;
       if (((setted & HOST) != HOST) && ((allocated & HOST) != HOST))
       {
           if (retval = Allocate(this->size, HOST)) return retval;
           t_allocated = true;
       }
       if (retval = Move(DISK, HOST, size, offset)) return retval;
       if (retval = Move(HOST, DEVICE, size, offset, stream)) return retval;
       if (t_allocated)
       {
           if (retval = Release(HOST)) return retval;
       }
   }*/
    return retval;
  }  // Move(...)

  Array1D &operator=(const Array1D &other) {
#ifdef ENABLE_ARRAY_DEBUG
    PrintMsg(std::string(other.name) + " Assigment");
#endif
    // name      = other.name     ;
    // file_name = other.file_name;
    size = other.size;
    // flag      = other.flag     ;
    // use_cuda_alloc = other.use_cuda_alloc;
    setted = other.setted;
    allocated = other.allocated;
    h_pointer = other.h_pointer;
    d_pointer = other.d_pointer;
    return *this;
  }

  __host__ __device__ __forceinline__ ValueT &operator[](std::size_t idx) {
#ifdef __CUDA_ARCH__
    return d_pointer[idx];
#else
#ifdef ENABLE_ARRAY_DEBUG
    if (h_pointer == NULL)
      GRError(cudaErrorInvalidHostPointer,
              std::string(name) + " not defined on " + Location_to_string(HOST),
              __FILE__, __LINE__);
    if (idx >= size)
      GRError(cudaErrorInvalidHostPointer,
              std::string(name) + " access out of bound", __FILE__, __LINE__);
      // printf("%s @ %p [%ld]ed1\n", name.c_str(),
      // h_pointer,idx);fflush(stdout);
#endif
    return h_pointer[idx];
#endif
  }

  __host__ __device__ __forceinline__ ValueT &operator[](
      std::size_t idx) const {
#ifdef __CUDA_ARCH__
    return d_pointer[idx];
#else
#ifdef ENABLE_ARRAY_DEBUG
    if (h_pointer == NULL)
      GRError(cudaErrorInvalidHostPointer,
              std::string(name) + " not defined on " + Location_to_string(HOST),
              __FILE__, __LINE__);
    if (idx >= size)
      GRError(cudaErrorInvalidHostPointer,
              std::string(name) + " access out of bound", __FILE__, __LINE__);
      // PrintMsg(name + " [" + std::string(idx) + "]ed2");
#endif
    return h_pointer[idx];
#endif
  }

  __host__ __device__ __forceinline__ ValueT *operator->() const {
#ifdef __CUDA_ARCH__
    return d_pointer;
#else
#ifdef ENABLE_ARRAY_DEBUG
    if (h_pointer == NULL)
      GRError(cudaErrorInvalidHostPointer,
              std::string(name) + " not deined on " + Location_to_string(HOST),
              __FILE__, __LINE__);
      // PrintMsg(std::string(name) + " -> ed");
#endif
    return h_pointer;
#endif
  }

  template <typename T>
  __host__ __device__ __forceinline__ ValueT *operator+(const T &offset) const {
#ifdef __CUDA_ARCH__
    return d_pointer + offset;
#else
#ifdef ENABLE_ARRAY_DEBUG
    if (h_pointer == NULL)
      GRError(cudaErrorInvalidHostPointer,
              std::string(name) + " not deined on " + Location_to_string(HOST),
              __FILE__, __LINE__);
      // PrintMsg(name + " -> ed");
#endif
    return h_pointer + offset;
#endif
  }

  __host__ __device__ __forceinline__ bool isEmpty() const {
#ifdef __CUDA_ARCH__
    return (d_pointer == NULL) ? true : false;
#else
    return (h_pointer == NULL) ? true : false;
#endif
  }

  template <typename T>
  Array1DT &operator=(T val);
  template <typename T>
  Array1DT &operator+=(T val);
  template <typename T>
  Array1DT &operator-=(T val);
  template <typename T>
  Array1DT &operator*=(T val);
  template <typename T>
  Array1DT &operator/=(T val);

  // operation with scalar
  template <typename T>
  cudaError_t Set(T value, SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                  Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename T = ValueT>
  cudaError_t SetIdx(T scale = 1,
                     SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                     Location target = LOCATION_DEFAULT,
                     cudaStream_t stream = 0);

  template <typename T>
  cudaError_t Add(T value, SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                  Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename T>
  cudaError_t Minus(T value,
                    SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                    Location target = LOCATION_DEFAULT,
                    cudaStream_t stream = 0);

  template <typename T>
  cudaError_t Mul(T value, SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                  Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename T>
  cudaError_t Div(T value, SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                  Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename CompareT, typename AssignT>
  cudaError_t CAS(CompareT compare, AssignT val,
                  SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                  Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  // operations with other arrays
  template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
            unsigned int cudaHostRegisterFlag_in>
  cudaError_t Set(
      Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
      Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
            unsigned int cudaHostRegisterFlag_in>
  cudaError_t Add(
      Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
      Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
            unsigned int cudaHostRegisterFlag_in>
  cudaError_t Minus(
      Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
      Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
            unsigned int cudaHostRegisterFlag_in>
  cudaError_t Mul(
      Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
      Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
            unsigned int cudaHostRegisterFlag_in>
  cudaError_t Div(
      Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
      SizeT length = PreDefinedValues<SizeT>::InvalidValue,
      Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  template <typename SizeT_in1, typename ValueT_in1, ArrayFlag FLAG_in1,
            unsigned int cudaHostRegisterFlag_in1, typename SizeT_in2,
            typename ValueT_in2, ArrayFlag FLAG_in2,
            unsigned int cudaHostRegisterFlag_in2, typename T>
  cudaError_t Mad(Array1D<SizeT_in1, ValueT_in1, FLAG_in1,
                          cudaHostRegisterFlag_in1> &array_in1,
                  Array1D<SizeT_in2, ValueT_in2, FLAG_in2,
                          cudaHostRegisterFlag_in2> &array_in2,
                  T scale, SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                  Location target = LOCATION_DEFAULT, cudaStream_t stream = 0);

  // Sorting
  template <typename CompareLambda>
  cudaError_t Sort(CompareLambda compare =
                       [] __host__ __device__(const ValueT &a,
                                              const ValueT &b) {
                         return a < b;
                       },
                   SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                   SizeT offset = 0, Location target = LOCATION_DEFAULT,
                   cudaStream_t stream = 0);

  template <typename ArrayT, typename CompareLambda>
  cudaError_t Sort_by_Key(ArrayT &array_in,
                          CompareLambda compare =
                              [] __host__ __device__(const ValueT &a,
                                                     const ValueT &b) {
                                return a < b;
                              },
                          SizeT length = PreDefinedValues<SizeT>::InvalidValue,
                          SizeT offset = 0, Location target = LOCATION_DEFAULT,
                          cudaStream_t stream = 0);

  template <typename T>
  cudaError_t tRead(std::string filename) {
    Array1D<SizeT, T> tArray;
    cudaError_t retval = cudaSuccess;
    int64_t tLength, tType;
    std::ifstream fin;
    fin.open(filename.c_str());
    if (!fin.is_open())
      return GRError(cudaErrorInvalidValue, "Unable to read file " + filename,
                     __FILE__, __LINE__);

    fin >> tLength >> tType;
    if (retval = tArray.Allocate(tLength, HOST)) return retval;

    for (SizeT i = 0; i < tLength; i++) fin >> tArray[i];

    if (retval = EnsureSize(tLength)) return retval;
    if (retval = this->ForEach(
            tArray,
            [] __host__ __device__(ValueT & element, const T &tElement) {
              element = tElement;
            },
            tLength, HOST))
      return retval;
    if (retval = tArray.Release()) return retval;
    return retval;
  }

  cudaError_t Read(std::string filename) {
    cudaError_t retval = cudaSuccess;
    int64_t tLength, tType;
    std::ifstream fin;
    fin.open(filename.c_str());
    if (!fin.is_open())
      return GRError(cudaErrorInvalidValue, "Unable to read file " + filename,
                     __FILE__, __LINE__);

    fin >> tLength >> tType;
    tType = tType & 0xFF;
    if (tType == Type2Enum<ValueT>::Id) {
      if (retval = EnsureSize(tLength)) return retval;
      for (SizeT i = 0; i < size; i++) fin >> h_pointer[i];
      fin.close();
    } else {
      fin.close();
      switch (tType) {
        case Type2Enum<char>::Id:
          retval = tRead<char>(filename);
          break;
        case Type2Enum<unsigned char>::Id:
          retval = tRead<unsigned char>(filename);
          break;
        case Type2Enum<short>::Id:
          retval = tRead<short>(filename);
          break;
        case Type2Enum<unsigned short>::Id:
          retval = tRead<unsigned short>(filename);
          break;
        case Type2Enum<int>::Id:
          retval = tRead<int>(filename);
          break;
        case Type2Enum<unsigned int>::Id:
          retval = tRead<unsigned int>(filename);
          break;
        case Type2Enum<long>::Id:
          retval = tRead<long>(filename);
          break;
        case Type2Enum<unsigned long>::Id:
          retval = tRead<unsigned long>(filename);
          break;
        case Type2Enum<long long>::Id:
          retval = tRead<long long>(filename);
          break;
        case Type2Enum<unsigned long long>::Id:
          retval = tRead<unsigned long long>(filename);
          break;
        case Type2Enum<float>::Id:
          retval = tRead<float>(filename);
          break;
        case Type2Enum<double>::Id:
          retval = tRead<double>(filename);
          break;

        case Type2Enum<char2>::Id:
          retval = tRead<char2>(filename);
          break;
        case Type2Enum<uchar2>::Id:
          retval = tRead<uchar2>(filename);
          break;
        case Type2Enum<short2>::Id:
          retval = tRead<short2>(filename);
          break;
        case Type2Enum<ushort2>::Id:
          retval = tRead<ushort2>(filename);
          break;
        case Type2Enum<int2>::Id:
          retval = tRead<int2>(filename);
          break;
        case Type2Enum<uint2>::Id:
          retval = tRead<uint2>(filename);
          break;
        case Type2Enum<long2>::Id:
          retval = tRead<long2>(filename);
          break;
        case Type2Enum<ulong2>::Id:
          retval = tRead<ulong2>(filename);
          break;
        case Type2Enum<longlong2>::Id:
          retval = tRead<longlong2>(filename);
          break;
        case Type2Enum<ulonglong2>::Id:
          retval = tRead<ulonglong2>(filename);
          break;
        case Type2Enum<float2>::Id:
          retval = tRead<float2>(filename);
          break;
        case Type2Enum<double2>::Id:
          retval = tRead<double2>(filename);
          break;

        case Type2Enum<char3>::Id:
          retval = tRead<char3>(filename);
          break;
        case Type2Enum<uchar3>::Id:
          retval = tRead<uchar3>(filename);
          break;
        case Type2Enum<short3>::Id:
          retval = tRead<short3>(filename);
          break;
        case Type2Enum<ushort3>::Id:
          retval = tRead<ushort3>(filename);
          break;
        case Type2Enum<int3>::Id:
          retval = tRead<int3>(filename);
          break;
        case Type2Enum<uint3>::Id:
          retval = tRead<uint3>(filename);
          break;
        case Type2Enum<long3>::Id:
          retval = tRead<long3>(filename);
          break;
        case Type2Enum<ulong3>::Id:
          retval = tRead<ulong3>(filename);
          break;
        case Type2Enum<longlong3>::Id:
          retval = tRead<longlong3>(filename);
          break;
        case Type2Enum<ulonglong3>::Id:
          retval = tRead<ulonglong3>(filename);
          break;
        case Type2Enum<float3>::Id:
          retval = tRead<float3>(filename);
          break;
        case Type2Enum<double3>::Id:
          retval = tRead<double3>(filename);
          break;

        case Type2Enum<char4>::Id:
          retval = tRead<char4>(filename);
          break;
        case Type2Enum<uchar4>::Id:
          retval = tRead<uchar4>(filename);
          break;
        case Type2Enum<short4>::Id:
          retval = tRead<short4>(filename);
          break;
        case Type2Enum<ushort4>::Id:
          retval = tRead<ushort4>(filename);
          break;
        case Type2Enum<int4>::Id:
          retval = tRead<int4>(filename);
          break;
        case Type2Enum<uint4>::Id:
          retval = tRead<uint4>(filename);
          break;
        case Type2Enum<long4>::Id:
          retval = tRead<long4>(filename);
          break;
        case Type2Enum<ulong4>::Id:
          retval = tRead<ulong4>(filename);
          break;
        case Type2Enum<longlong4>::Id:
          retval = tRead<longlong4>(filename);
          break;
        case Type2Enum<ulonglong4>::Id:
          retval = tRead<ulonglong4>(filename);
          break;
        case Type2Enum<float4>::Id:
          retval = tRead<float4>(filename);
          break;
        case Type2Enum<double4>::Id:
          retval = tRead<double4>(filename);
          break;

        // case Type2Enum<std::string>::Id :
        //    retval = tRead<std::string>(filename); break;
        // case util::Type2Enum<char*>::Id :
        //    retval = tRead<char*>(filename); break;
        default:
          retval =
              GRError(cudaErrorInvalidValue,
                      "Unsupported type (Id = " + std::to_string(tType) + ")",
                      __FILE__, __LINE__);
      }
      // if (tType == util::Type2Enum<char>::Id)
      //    retval = tRead<char>(fin);
    }
    return retval;
  }

  cudaError_t Write(std::string filename) {
    cudaError_t retval = cudaSuccess;
    std::ofstream fout;
    fout.open(filename.c_str());
    if (!fout.is_open())
      return GRError(cudaErrorInvalidValue, "Unable to write file " + filename,
                     __FILE__, __LINE__);

    fout << size << " " << Type2Enum<ValueT>::Id << std::endl;
    for (SizeT i = 0; i < size; i++) fout << h_pointer[i] << std::endl;
    fout.close();
    return retval;
  }

  template <typename T>
  cudaError_t tReadBinary(std::string filename) {
    Array1D<SizeT, T> tArray;
    cudaError_t retval = cudaSuccess;
    int64_t tLength, tType;
    std::ifstream fin;
    fin.open(filename.c_str(), std::ios::in | std::ios::binary);
    if (!fin.is_open())
      return GRError(cudaErrorInvalidValue, "Unable to read file " + filename,
                     __FILE__, __LINE__);

    fin.read((char *)(&tLength), 8);
    fin.read((char *)(&tType), 8);
    if (retval = tArray.Allocate(tLength, HOST)) return retval;
    PrintMsg("Reading " + std::to_string(tLength) + " " + typeid(T).name() +
             " from " + filename);

    fin.read((char *)(tArray + 0), tLength * sizeof(T));
    fin.close();

    if (retval = EnsureSize_(tLength, util::HOST)) return retval;
    if (retval = ForEach(
            tArray,
            [] __host__ __device__(ValueT & element, const T &tElement) {
              CrossAssign(element, tElement);
            },
            tLength, HOST))
      return retval;
    if (retval = tArray.Release()) return retval;
    return retval;
  }

  // struct ReadStruct
  // {
  //     Array1DT &array;
  //     std::string filename;
  //
  //     ReadStruct(Array1DT &a, std::string f) :
  //         array(a),
  //         filename(f)
  //     {
  //
  //     }
  //
  //     template <typename T>
  //     cudaError_t operator()(T &t)
  //     {
  //         return array.tReadBinary<T>(filename);
  //     }
  // };

  cudaError_t ReadBinary(std::string filename, bool ignore_file_error = false) {
    cudaError_t retval = cudaSuccess;
    int64_t tLength, tType;
    std::ifstream fin;
    fin.open(filename.c_str(), std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
      if (ignore_file_error) {
        return cudaErrorInvalidValue;
      } else
        return GRError(cudaErrorInvalidValue, "Unable to read file " + filename,
                       __FILE__, __LINE__);
    }

    fin.read((char *)(&tLength), 8);
    fin.read((char *)(&tType), 8);
    tType = tType & 0xFFF;
    PrintMsg("Reading from " + filename +
             ", typeId = " + std::to_string(tType) +
             ", targetId = " + std::to_string(util::Type2Enum<ValueT>::Id) +
             ", length = " + std::to_string(tLength));
    if (tType == util::Type2Enum<ValueT>::Id) {
      if (retval = EnsureSize_(tLength, util::HOST)) return retval;
      fin.read((char *)h_pointer, sizeof(ValueT) * tLength);
      fin.close();
    } else {
      fin.close();
      switch (tType) {
        case Type2Enum<char>::Id:
          retval = tReadBinary<char>(filename);
          break;
        case Type2Enum<unsigned char>::Id:
          retval = tReadBinary<unsigned char>(filename);
          break;
        case Type2Enum<short>::Id:
          retval = tReadBinary<short>(filename);
          break;
        case Type2Enum<unsigned short>::Id:
          retval = tReadBinary<unsigned short>(filename);
          break;
        case Type2Enum<int>::Id:
          retval = tReadBinary<int>(filename);
          break;
        case Type2Enum<unsigned int>::Id:
          retval = tReadBinary<unsigned int>(filename);
          break;
        case Type2Enum<long>::Id:
          retval = tReadBinary<long>(filename);
          break;
        case Type2Enum<unsigned long>::Id:
          retval = tReadBinary<unsigned long>(filename);
          break;
        case Type2Enum<long long>::Id:
          retval = tReadBinary<long long>(filename);
          break;
        case Type2Enum<unsigned long long>::Id:
          retval = tReadBinary<unsigned long long>(filename);
          break;
        case Type2Enum<float>::Id:
          retval = tReadBinary<float>(filename);
          break;
        case Type2Enum<double>::Id:
          retval = tReadBinary<double>(filename);
          break;

        case Type2Enum<char2>::Id:
          retval = tReadBinary<char2>(filename);
          break;
        case Type2Enum<uchar2>::Id:
          retval = tReadBinary<uchar2>(filename);
          break;
        case Type2Enum<short2>::Id:
          retval = tReadBinary<short2>(filename);
          break;
        case Type2Enum<ushort2>::Id:
          retval = tReadBinary<ushort2>(filename);
          break;
        case Type2Enum<int2>::Id:
          retval = tReadBinary<int2>(filename);
          break;
        case Type2Enum<uint2>::Id:
          retval = tReadBinary<uint2>(filename);
          break;
        case Type2Enum<long2>::Id:
          retval = tReadBinary<long2>(filename);
          break;
        case Type2Enum<ulong2>::Id:
          retval = tReadBinary<ulong2>(filename);
          break;
        case Type2Enum<longlong2>::Id:
          retval = tReadBinary<longlong2>(filename);
          break;
        case Type2Enum<ulonglong2>::Id:
          retval = tReadBinary<ulonglong2>(filename);
          break;
        case Type2Enum<float2>::Id:
          retval = tReadBinary<float2>(filename);
          break;
        case Type2Enum<double2>::Id:
          retval = tReadBinary<double2>(filename);
          break;

        case Type2Enum<char3>::Id:
          retval = tReadBinary<char3>(filename);
          break;
        case Type2Enum<uchar3>::Id:
          retval = tReadBinary<uchar3>(filename);
          break;
        case Type2Enum<short3>::Id:
          retval = tReadBinary<short3>(filename);
          break;
        case Type2Enum<ushort3>::Id:
          retval = tReadBinary<ushort3>(filename);
          break;
        case Type2Enum<int3>::Id:
          retval = tReadBinary<int3>(filename);
          break;
        case Type2Enum<uint3>::Id:
          retval = tReadBinary<uint3>(filename);
          break;
        case Type2Enum<long3>::Id:
          retval = tReadBinary<long3>(filename);
          break;
        case Type2Enum<ulong3>::Id:
          retval = tReadBinary<ulong3>(filename);
          break;
        case Type2Enum<longlong3>::Id:
          retval = tReadBinary<longlong3>(filename);
          break;
        case Type2Enum<ulonglong3>::Id:
          retval = tReadBinary<ulonglong3>(filename);
          break;
        case Type2Enum<float3>::Id:
          retval = tReadBinary<float3>(filename);
          break;
        case Type2Enum<double3>::Id:
          retval = tReadBinary<double3>(filename);
          break;

        case Type2Enum<char4>::Id:
          retval = tReadBinary<char4>(filename);
          break;
        case Type2Enum<uchar4>::Id:
          retval = tReadBinary<uchar4>(filename);
          break;
        case Type2Enum<short4>::Id:
          retval = tReadBinary<short4>(filename);
          break;
        case Type2Enum<ushort4>::Id:
          retval = tReadBinary<ushort4>(filename);
          break;
        case Type2Enum<int4>::Id:
          retval = tReadBinary<int4>(filename);
          break;
        case Type2Enum<uint4>::Id:
          retval = tReadBinary<uint4>(filename);
          break;
        case Type2Enum<long4>::Id:
          retval = tReadBinary<long4>(filename);
          break;
        case Type2Enum<ulong4>::Id:
          retval = tReadBinary<ulong4>(filename);
          break;
        case Type2Enum<longlong4>::Id:
          retval = tReadBinary<longlong4>(filename);
          break;
        case Type2Enum<ulonglong4>::Id:
          retval = tReadBinary<ulonglong4>(filename);
          break;
        case Type2Enum<float4>::Id:
          retval = tReadBinary<float4>(filename);
          break;
        case Type2Enum<double4>::Id:
          retval = tReadBinary<double4>(filename);
          break;

        // case Type2Enum<std::string>::Id :
        //    retval = tReadBinary<std::string>(filename); break;
        // case util::Type2Enum<char*>::Id :
        //    retval = tReadBinary<char*>(filename); break;
        default:
          retval =
              GRError(cudaErrorInvalidValue,
                      "Unsupported type (Id = " + std::to_string(tType) + ")",
                      __FILE__, __LINE__);
      }

      // if (tType == util::Type2Enum<char>::Id)
      //    retval = tRead<char>(fin);
    }
    return retval;
  }

  cudaError_t WriteBinary(std::string filename,
                          bool ignore_file_error = false) {
    cudaError_t retval = cudaSuccess;
    std::ofstream fout;
    fout.open(filename.c_str(), std::ios::out | std::ios::binary);
    if (!fout.is_open()) {
      if (ignore_file_error) {
        return cudaErrorInvalidValue;
      } else
        return GRError(cudaErrorInvalidValue,
                       "Unable to write file " + filename, __FILE__, __LINE__);
    }

    int64_t tLength = size;
    int64_t tType = util::Type2Enum<ValueT>::Id;
    fout.write((char *)(&tLength), 8);
    fout.write((char *)(&tType), 8);
    fout.write((char *)h_pointer, sizeof(ValueT) * size);
    fout.close();
    return retval;
  }

  cudaError_t Print(std::string message = "",
                    SizeT limit = PreDefinedValues<SizeT>::InvalidValue,
                    Location target = LOCATION_DEFAULT,
                    cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;

    if (message == "") message = std::string(name) + " : ";
    if (!isValid(limit) || limit > size) limit = size;
    if (target == LOCATION_DEFAULT) target = setted | allocated;

    ValueT *h_array = GetPointer(HOST);
    bool temp_allocated = false;
    if (h_array == NULL && limit != 0) {
      temp_allocated = true;
      h_array = new ValueT[limit];
      if (h_array == NULL)
        return GRError(cudaErrorMemoryAllocation,
                       "temp arrary for printing " + std::string(name) +
                           " allocation failed.",
                       __FILE__, __LINE__);
    }

    if (limit != 0 && (target & DEVICE) != 0) {
      GUARD_CU2(cudaMemcpyAsync(h_array, d_pointer, sizeof(ValueT) * limit,
                                cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyDeviceToHost failed.");
      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    }

    std::string str = message;
    for (SizeT i = 0; i < limit; i++) {
      if ((i % 10) == 0) str = str + " |(" + std::to_string(i) + ")";
      str = str + " " + std::to_string(h_array[i]);
    }
    PrintMsg(str);

    if (temp_allocated) {
      delete[] h_array;
      h_array = NULL;
      temp_allocated = false;
    }
    return retval;
  }

};  // struct Array1D

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
