#include <gunrock/io/smtx.hxx>

#include <gtest/gtest.h>

TEST(io, smtx_loads_compact_csr_fixture) {
  gunrock::io::smtx_t<int, int, float> loader;
  auto csr = loader.load(test_data_path("tiny.smtx"));

  EXPECT_EQ(csr.number_of_rows, 4);
  EXPECT_EQ(csr.number_of_columns, 4);
  EXPECT_EQ(csr.number_of_nonzeros, 5);
  expect_values(csr.row_offsets, {0, 2, 3, 5, 5});
  expect_values(csr.column_indices, {1, 2, 3, 0, 2});
  EXPECT_EQ(csr.nonzero_values.size(), 5);
}

TEST(io, smtx_rejects_inconsistent_row_count) {
  gunrock::io::smtx_t<int, int, float> loader;
  EXPECT_THROW(loader.load(test_data_path("malformed.smtx")),
               std::invalid_argument);
}