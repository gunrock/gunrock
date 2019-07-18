// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_unitests.cu
 *
 * @brief Main test driver for all googletests.
 * @source
 * https://github.com/google/googletest/blob/master/googletest/docs/Primer.md
 */

#include <stdio.h>
#include <gunrock/gunrock.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

/**
 * @brief: Gunrock: Google tests -- list of tests
 * found in this directory, testing core functionality
 * of gunrock: primitives, operators, device intrinsics,
 * etc.
 *
 */

// bug:: malloc_consolidate(): invalid chunk size
//#include "test_lib_pr.h"

// Tests the RepeatFor Operator
#include "test_repeatfor.h"

// Tests Segmented Reduction (device)
#include "test_segreduce.h"

// Tests Binary Search
#include "test_binarysearch.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
