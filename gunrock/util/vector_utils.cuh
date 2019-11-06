// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * vector_utils.cuh
 *
 * @brief utility functions for vectors
 */

#pragma once

#include <gunrock/util/vector_types.cuh>
#include <gunrock/util/type_enum.cuh>

namespace gunrock {
namespace util {

template <typename BaseT, typename OpT>
cudaError_t VectorSwitch(int vec_length, OpT op) {
  cudaError_t retval = cudaSuccess;

  if (vec_length == 1) {
    BaseT t;
    return op(t);
  }

  if (vec_length == 2) {
    typename VectorType<BaseT, 2>::Type t;
    return op(t);
  }

  if (vec_length == 3) {
    typename VectorType<BaseT, 3>::Type t;
    return op(t);
  }

  if (vec_length == 4) {
    typename VectorType<BaseT, 4>::Type t;
    return op(t);
  }
  return retval;
}

template <typename OpT>
cudaError_t TypeSwitch(TypeId id, OpT op) {
  cudaError_t retval = cudaSuccess;

  int vec_length = (id >> 8) + 1;
  TypeId base_id = (id & 0xFF);

  switch (base_id) {
    case Type2Enum<char>::Id:
      retval = VectorSwitch<char, OpT>(vec_length, op);
      break;
    case Type2Enum<unsigned char>::Id:
      retval = VectorSwitch<unsigned char, OpT>(vec_length, op);
      break;
    case Type2Enum<short>::Id:
      retval = VectorSwitch<short, OpT>(vec_length, op);
      break;
    case Type2Enum<unsigned int>::Id:
      retval = VectorSwitch<unsigned int, OpT>(vec_length, op);
      break;
    case Type2Enum<long>::Id:
      retval = VectorSwitch<long, OpT>(vec_length, op);
      break;
    case Type2Enum<unsigned long>::Id:
      retval = VectorSwitch<unsigned long, OpT>(vec_length, op);
      break;
    case Type2Enum<long long>::Id:
      retval = VectorSwitch<long long, OpT>(vec_length, op);
      break;
    case Type2Enum<unsigned long long>::Id:
      retval = VectorSwitch<unsigned long long, OpT>(vec_length, op);
      break;
    case Type2Enum<float>::Id:
      retval = VectorSwitch<float, OpT>(vec_length, op);
      break;
    case Type2Enum<double>::Id:
      retval = VectorSwitch<double, OpT>(vec_length, op);
      break;
    case Type2Enum<std::string>::Id:
      retval = VectorSwitch<std::string, OpT>(vec_length, op);
      break;
    case Type2Enum<char *>::Id:
      retval = VectorSwitch<char *, OpT>(vec_length, op);
      break;
    default:
      retval = GRError("Unsupported type (Id = " + std::to_string(id) + ")",
                       __FILE__, __LINE__);
  }
  return retval;
}

template <int T1_SIZE, int T2_SIZE>
struct CAssign {};

template <int T1_SIZE>
struct CAssign<T1_SIZE, 1> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result.x = val;
    return result;
  }
};

template <int T1_SIZE>
struct CAssign<T1_SIZE, 2> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result.x = val.x;
    result.y = val.y;
    return result;
  }
};

template <int T1_SIZE>
struct CAssign<T1_SIZE, 3> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result.x = val.x;
    result.y = val.y;
    result.z = val.z;
    return result;
  }
};

template <int T1_SIZE>
struct CAssign<T1_SIZE, 4> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result.x = val.x;
    result.y = val.y;
    result.z = val.z;
    result.w = val.w;
    return result;
  }
};

template <>
struct CAssign<1, 1> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result = val;
    return result;
  }
};

template <>
struct CAssign<1, 2> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result = val.x;
    return result;
  }
};

template <>
struct CAssign<1, 3> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result = val.x;
    return result;
  }
};

template <>
struct CAssign<1, 4> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result = val.x;
    return result;
  }
};

template <>
struct CAssign<2, 3> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result.x = val.x;
    result.y = val.y;
    return result;
  }
};

template <>
struct CAssign<2, 4> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result.x = val.x;
    result.y = val.y;
    return result;
  }
};

template <>
struct CAssign<3, 4> {
  template <typename T1, typename T2>
  __host__ __device__ __forceinline__ static T1 &Assign(T1 &result,
                                                        const T2 &val) {
    result.x = val.x;
    result.y = val.y;
    result.z = val.z;
    return result;
  }
};

template <typename T1, typename T2>
__host__ __device__ __forceinline__ T1 &CrossAssign(T1 &result, const T2 &val) {
  return CAssign<(Type2Enum<T1>::Id >> 8) + 1,
                 (Type2Enum<T2>::Id >> 8) + 1>::Assign(result, val);
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
