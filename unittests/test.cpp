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
 * @source
 * https://github.com/google/googletest/blob/master/googletest/docs/Primer.md
 */

#include <stdio.h>
#include <gunrock/gunrock.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Add google tests
#include "bfs/test_lib_bfs.h"
#include "cc/test_lib_cc.h"
#include "bc/test_lib_bc.h"
#include "pr/test_lib_pr.h"
#include "sssp/test_lib_sssp.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
