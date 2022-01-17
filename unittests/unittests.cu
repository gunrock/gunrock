/**
 * @file test_unit.cpp
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Unit testing driver for Googletests.
 * @version 0.1
 * @date 2021-12-24
 *
 * @copyright Copyright (c) 2021
 *
 */

/// Add your test to the following header file:
#include "unittests.hxx"

/// Main Google test driver.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}