// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * basic_utils.h
 *
 * @brief Basic utility macros/routines
 */

#pragma once
#include <string>

namespace gunrock {
namespace util {

/*****************************************************************
 * Macro utilities
 *****************************************************************/

/**
 * Select maximum
 */
#define GR_MAX(a, b) ((a > b) ? a : b)

/**
 * Select maximum
 */
#define GR_MIN(a, b) ((a < b) ? a : b)

/**
 * Return the size in quad-words of a number of bytes
 */
#define GR_QUADS(bytes) (((bytes + sizeof(uint4) - 1) / sizeof(uint4)))

/*****************************************************************
 * Simple templated utilities
 *****************************************************************/

/**
 * Supress warnings for unused constants
 */
template <typename T>
__host__ __device__ __forceinline__ void SuppressUnusedConstantWarning(
    const T) {}

/**
 * Perform a swap
 */
template <typename T>
void __host__ __device__ __forceinline__ Swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

template <typename K, int magnitude, bool shift_left>
struct MagnitudeShiftOp;

/**
 * MagnitudeShift().  Allows you to shift left for positive magnitude values,
 * right for negative.
 *
 * N.B. This code is a little strange; we are using this meta-programming
 * pattern of partial template specialization for structures in order to
 * decide whether to shift left or right.  Normally we would just use a
 * conditional to decide if something was negative or not and then shift
 * accordingly, knowing that the compiler will elide the untaken branch,
 * i.e., the out-of-bounds shift during dead code elimination. However,
 * the pass for bounds-checking shifts seems to happen before the DCE
 * phase, which results in a an unsightly number of compiler warnings, so
 * we force the issue earlier using structural template specialization.
 */
template <typename K, int magnitude>
__device__ __forceinline__ K MagnitudeShift(K key)
{
    return MagnitudeShiftOp<K, (magnitude > 0) ? magnitude : magnitude * -1, (magnitude > 0)>::Shift(key);
}

template <typename K, int magnitude>
struct MagnitudeShiftOp<K, magnitude, true> {
  __device__ __forceinline__ static K Shift(K key) { return key << magnitude; }
};

template <typename K, int magnitude>
struct MagnitudeShiftOp<K, magnitude, false> {
  __device__ __forceinline__ static K Shift(K key) { return key >> magnitude; }
};

/*****************************************************************
 * Metaprogramming Utilities
 *****************************************************************/

/**
 * Null type
 */
struct NullType {

    template <typename T>
    __host__ __device__ __forceinline__
    NullType& operator =(const T&)
    {
        return *this;
    }
};

/**
 * Int2Type
 */
template <int N>
struct Int2Type {
  enum { VALUE = N };
};

/**
 * Statically determine log2(N), rounded up, e.g.,
 *      Log2<8>::VALUE == 3
 *      Log2<3>::VALUE == 2
 */
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
  // Inductive case
  static const int VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE;
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
  // Base case
  static const int VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1;
};

/**
 * If/Then/Else
 */
template <bool IF, typename ThenType, typename ElseType>
struct If {
  // true
  typedef ThenType Type;
};

template <typename ThenType, typename ElseType>
struct If<false, ThenType, ElseType> {
  // false
  typedef ElseType Type;
};

template <bool IF, unsigned int ThenVal, unsigned int ElseVal>
struct If_Val
{
    // true
    static const unsigned int Value = ThenVal;
};

template <unsigned int ThenVal, unsigned int ElseVal>
struct If_Val<false, ThenVal, ElseVal>
{
    // false
    static const unsigned int Value = ElseVal;
};

template <bool IF>
struct If_Op
{
    template <typename Op>
    static void Exec(Op op)
    {
    }
};

template <>
struct If_Op<true>
{
    template <typename Op>
    static void Exec(Op op)
    {
        op();
    }
};

/**
 * Equals
 */
template <typename A, typename B>
struct Equals {
  enum { VALUE = 0, NEGATE = 1 };
};

template <typename A>
struct Equals<A, A> {
  enum { VALUE = 1, NEGATE = 0 };
};

/**
 * Is volatile
 */
template <typename Tp>
struct IsVolatile {
  enum { VALUE = 0 };
};
template <typename Tp>
struct IsVolatile<Tp volatile> {
  enum { VALUE = 1 };
};

/**
 * Removes pointers
 */
template <typename Tp, typename Up>
struct RemovePointersHelper {
  typedef Tp Type;
};
template <typename Tp, typename Up>
struct RemovePointersHelper<Tp, Up *> {
  typedef typename RemovePointersHelper<Up, Up>::Type Type;
};
template <typename Tp>
struct RemovePointers : RemovePointersHelper<Tp, Tp> {};

template <typename T>
std::string to_string(T* ptr)
{
    char temp_str[128];
    sprintf(temp_str, "%p", ptr);
    return std::string(temp_str);
}

template <typename T>
void SeperateFileName(
    T _filename,
    std::string &dir,
    std::string &file,
    std::string &extension)
{
    std::string filename(_filename);

    auto dir_pos = std::string::npos;
    #ifdef _WIN32
        dir_pos = filename.find_last_of('\\');
    #else
        dir_pos = filename.find_last_of('/');
    #endif
    auto extension_pos = filename.find_last_of('.');

    if (dir_pos == std::string::npos)
    {
        dir_pos = 0;
        dir = "";
        file = filename.substr(0, extension_pos);
    } else {
        if (dir_pos == 0)
            dir = "/";
        else
            dir  = filename.substr(0, dir_pos);
        if (extension_pos == std::string::npos)
            file = filename.substr(dir_pos + 1);
        else
            file = filename.substr(dir_pos + 1, extension_pos - dir_pos - 1);
    }

    if (extension_pos != std::string::npos)
        extension = filename.substr(extension_pos + 1);
    else
        extension = "";
}

} // namespace util
} // namespace gunrock
