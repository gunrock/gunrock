/**
 * @file smtx.cuh
 * @author Jonathan Wapman (jdwapman@ucdavis.edu)
 * @brief Unit test for smtx loading.
 * @version 0.1
 * @date 2022-05-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <gunrock/error.hxx>    // error checking
#include <gunrock/io/smtx.hxx>  // smtx support

#include <gtest/gtest.h>

TEST(io, smtx) {
  using namespace gunrock;

  // Load the smtx matrix
  using row_t = int;
  using edge_t = int;
  using nonzero_t = float;
  using csr_t = format::csr_t<memory_space_t::device, row_t, edge_t, nonzero_t>;

  io::smtx_t<row_t, edge_t, nonzero_t> smtx;

  csr_t csr = smtx.load(
      "datasets/layers.0.blocks.0.attn.proj_swin_tiny_unstructured_50.smtx");

  EXPECT_EQ(csr.number_of_rows, 96);
  EXPECT_EQ(csr.number_of_columns, 96);
  EXPECT_EQ(csr.number_of_nonzeros, 4608);
}