// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * multiple_buffering.cuh
 *
 * @brief Storage wrapper for multi-pass stream transformations that
 * require a secondary problem storage array to stream results back
 * and forth from.
 */

#pragma once

#include <gunrock/util/basic_utils.h>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace util {

/**
 * Storage wrapper for multi-pass stream transformations that require a
 * more than one problem storage array to stream results back and forth from.
 *
 * This wrapper provides maximum flexibility for re-using device allocations
 * for subsequent transformations.  As such, it is the caller's responsibility
 * to free any non-NULL storage arrays when no longer needed.
 *
 * Many multi-pass stream computations require at least two problem storage
 * arrays, e.g., one for reading in from, the other for writing out to.
 * (And their roles can be reversed for each subsequent pass.) This structure
 * tracks two sets of device vectors (a keys and a values sets), and a
 * "selector" member to index which vector in each set is "currently valid".
 * I.e., the valid data within "MultipleBuffer<2, int, int> b" is accessible by:
 *
 *      b.d_keys[b.selector];
 *
 */
template <int BUFFER_COUNT, typename _KeyType, typename _SizeType,
          typename _ValueType = util::NullType,
          unsigned int TARGET = util::DEVICE>
struct MultipleBuffer {
  typedef _SizeType SizeType;
  typedef _KeyType KeyType;
  typedef _ValueType ValueType;

  // Set of device vector pointers for keys
  Array1D<SizeType, KeyType> keys[BUFFER_COUNT];

  // Set of device vector pointers for values
  Array1D<SizeType, ValueType> values[BUFFER_COUNT];

  // Selector into the set of device vector pointers (i.e., where the results
  // are)
  int selector;

  // Constructor
  MultipleBuffer() {
    selector = 0;
    for (int i = 0; i < BUFFER_COUNT; i++) {
      keys[i].SetName("keys");
      values[i].SetName("values");
    }
  }
};

/**
 * Double buffer (a.k.a. page-flip, ping-pong, etc.) version of the
 * multi-buffer storage abstraction above.
 *
 * Many of the B40C primitives are templated upon the DoubleBuffer type: they
 * are compiled differently depending upon whether the declared type contains
 * keys-only versus key-value pairs (i.e., whether ValueType is util::NullType
 * or some real type).
 *
 * Declaring keys-only storage wrapper:
 *
 *      DoubleBuffer<KeyType> key_storage;
 *
 * Declaring key-value storage wrapper:
 *
 *      DoubleBuffer<KeyType, ValueType> key_value_storage;
 *
 */
template <typename KeyType, typename SizeType,
          typename ValueType = util::NullType,
          unsigned int TARGET = util::DEVICE>
struct DoubleBuffer : MultipleBuffer<2, KeyType, SizeType, ValueType, TARGET> {
  typedef MultipleBuffer<2, KeyType, SizeType, ValueType, TARGET> ParentType;

  // Constructor
  DoubleBuffer() : ParentType() {}

  // Constructor
  DoubleBuffer(SizeType size, KeyType* keys) : ParentType() {
    this->keys[0].SetPointer(keys, size, TARGET);
  }

  // Constructor
  DoubleBuffer(SizeType size, KeyType* keys, ValueType* values) : ParentType() {
    this->keys[0].SetPointer(keys, size, TARGET);
    this->values[0].SetPointer(values, size, TARGET);
  }

  // Constructor
  DoubleBuffer(SizeType size0, SizeType size1, KeyType* keys0, KeyType* keys1,
               ValueType* values0, ValueType* values1)
      : ParentType() {
    this->keys[0].SetPointer(keys0, size0, TARGET);
    this->keys[1].SetPointer(keys1, size1, TARGET);
    this->values[0].SetPointer(values0, size0, TARGET);
    this->values[1].SetPointer(values1, size1, TARGET);
  }
};

/**
 * Triple buffer version of the multi-buffer storage abstraction above.
 */
template <typename KeyType, typename SizeType,
          typename ValueType = util::NullType,
          unsigned int TARGET = util::DEVICE>
struct TripleBuffer : MultipleBuffer<3, KeyType, SizeType, ValueType, TARGET> {
  typedef MultipleBuffer<3, KeyType, SizeType, ValueType, TARGET> ParentType;

  // Constructor
  TripleBuffer() : ParentType() {}
};

}  // namespace util
}  // namespace gunrock
