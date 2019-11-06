// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * random_bits.h
 *
 * @brief A random bits generator
 */

#pragma once

#include <stdlib.h>

namespace gunrock {
namespace util {

/**
 * Generates random 32-bit keys.
 *
 * We always take the second-order byte from rand() because the higher-order
 * bits returned by rand() are commonly considered more uniformly distributed
 * than the lower-order bits.
 *
 * We can decrease the entropy level of keys by adopting the technique
 * of Thearling and Smith in which keys are computed from the bitwise AND of
 * multiple random samples:
 *
 * entropy_reduction    | Effectively-unique bits per key
 * -----------------------------------------------------
 * -1                   | 0
 * 0                    | 32
 * 1                    | 25.95
 * 2                    | 17.41
 * 3                    | 10.78
 * 4                    | 6.42
 * ...                  | ...
 *
 */
template <typename K>
void RandomBits(K &key, int entropy_reduction = 0,
                int lower_key_bits = sizeof(K) * 8) {
  const unsigned int NUM_UCHARS =
      (sizeof(K) + sizeof(unsigned char) - 1) / sizeof(unsigned char);
  unsigned char key_bits[NUM_UCHARS];

  do {
    for (int j = 0; j < NUM_UCHARS; j++) {
      unsigned char quarterword = 0xff;
      for (int i = 0; i <= entropy_reduction; i++) {
        quarterword &= (rand() >> 7);
      }
      key_bits[j] = quarterword;
    }

    if (lower_key_bits < sizeof(K) * 8) {
      unsigned long long base = 0;
      memcpy(&base, key_bits, sizeof(K));
      base &= (1 << lower_key_bits) - 1;
      memcpy(key_bits, &base, sizeof(K));
    }

    memcpy(&key, key_bits, sizeof(K));

  } while (key !=
           key);  // avoids NaNs when generating random floating point numbers
}

}  // namespace util
}  // namespace gunrock
