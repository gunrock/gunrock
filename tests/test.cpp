// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test.cpp
 *
 * @brief Main test driver for all googletests.
 * @source https://github.com/google/googletest/blob/master/googletest/docs/Primer.md
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

// Add google tests
#include "bfs/test_lib_bfs.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
