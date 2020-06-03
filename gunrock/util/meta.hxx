#pragma once

/**
 * @see
 * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#compilation-phases
 * Guiding CUDA architecture's compilation paths.
 *
 */
#ifdef __CUDA_ARCH__ // Device-side compilation
#define GUNROCK_CUDA_ARCH __CUDA_ARCH__
#endif

#define GUNROCK_CUDACC __CUDACC__      // Use nvcc to compile
#define GUNROCK_DEBUG __CUDACC_DEBUG__ // Library-wide debug flag

#ifdef GUNROCK_CUDACC

#ifndef GUNROCK_HOST_DEVICE
#define GUNROCK_HOST_DEVICE __forceinline__ __device__ __host__
#endif

#ifndef GUNROCK_DEVICE
#define GUNROCK_DEVICE __device__
#endif

#ifndef GUNROCK_F_DEVICE
#define GUNROCK_F_DEVICE __forceinline__ __device__
#endif

// Requires --extended-lambda (-extended-lambda) flag.
// Allow __host__, __device__ annotations in lambda declarations.
#ifndef GUNROCK_LAMBDA
#define GUNROCK_LAMBDA __device__ __host__
#endif

#else  // #ifndef GUNROCK_CUDACC
#endif // #ifdef GUNROCK_CUDACC

namespace gunrock {
namespace util {

/**
 * @namespace meta
 * Metaprogramming utilities.
 */
namespace meta {

/**
 * @brief null type
 */
struct null_t
{
  template<typename T>
  GUNROCK_HOST_DEVICE null_t& operator=(const T&)
  {
    return *this;
  }
};

/**
 * @brief Int2Type
 */
template<int N>
struct int2_t
{
  enum
  {
    VALUE = N
  };
};

/**
 * @brief if/then/else
 */
template<bool IF, typename then_t, typename else_t>
struct _if
{
  // true
  typedef then_t type_t;
};

template<typename then_t, typename else_t>
struct _if<false, then_t, else_t>
{
  // false
  typedef else_t type_t;
};

template<bool IF, unsigned THEN_VALUE, unsigned ELSE_VALUE>
struct _if_value
{
  // true
  static const unsigned VALUE = THEN_VALUE;
};

template<unsigned THEN_VALUE, unsigned ELSE_VALUE>
struct _if_value<false, THEN_VALUE, ELSE_VALUE>
{
  // false
  static const unsigned VALUE = ELSE_VALUE;
};

template<bool IF>
struct _if_op
{
  template<typename op_t>
  static void Exec(op_t op)
  {}
};

template<>
struct _if_op<true>
{
  template<typename op_t>
  static void Exec(op_t op)
  {
    op();
  }
};

/**
 * @brief equals
 */
template<typename a_t, typename b_t>
struct equals
{
  enum
  {
    VALUE = 0,
    NEGATE = 1
  };
};

template<typename a_t>
struct equals<a_t, a_t>
{
  enum
  {
    VALUE = 1,
    NEGATE = 0
  };
};

/**
 * @brief Is volatile
 */
template<typename tp_t>
struct is_volatile
{
  enum
  {
    VALUE = 0
  };
};
template<typename tp_t>
struct is_volatile<tp_t volatile>
{
  enum
  {
    VALUE = 1
  };
};

/**
 * @brief removes pointers
 */
template<typename tp_t, typename up_t>
struct remove_pointers_helper
{
  typedef tp_t type_t;
};
template<typename tp_t, typename up_t>
struct remove_pointers_helper<tp_t, up_t*>
{
  typedef typename remove_pointers_helper<up_t, up_t>::type_t type_t;
};
template<typename Tp>
struct remove_pointers : remove_pointers_helper<tp_t, tp_t>
{};

template<typename type_t>
std::string
to_string(type_t* ptr)
{
  char temp_str[128];
  sprintf(temp_str, "%p", ptr);
  return std::string(temp_str);
}

} // namespace metaprogramming

} // namespace util
} // namespace gunrock