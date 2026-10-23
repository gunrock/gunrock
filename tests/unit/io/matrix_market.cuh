#include <gunrock/io/matrix_market.hxx>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace {

std::string test_data_path(char const* filename) {
  return (std::filesystem::path(GUNROCK_TEST_DATA_DIR) / filename).string();
}

}  // namespace

TEST(io, filepath_helpers_classify_supported_graph_files) {
  EXPECT_EQ(gunrock::util::extract_filename("/graphs/sample.mtx"),
            "sample.mtx");
  EXPECT_EQ(gunrock::util::extract_dataset("sample.mtx"), "sample");
  EXPECT_TRUE(gunrock::util::is_market("sample.mtx"));
  EXPECT_TRUE(gunrock::util::is_market("sample.mmio"));
  EXPECT_FALSE(gunrock::util::is_market("sample.csr"));
  EXPECT_TRUE(gunrock::util::is_binary_csr("sample.csr"));
  EXPECT_FALSE(gunrock::util::is_binary_csr("sample.mtx"));
}

TEST(io, matrix_market_loads_pattern_graph) {
  gunrock::io::matrix_market_t<int, int, float> loader;
  auto [properties, coo] = loader.load(test_data_path("tiny.mtx"));

  EXPECT_TRUE(properties.directed);
  EXPECT_FALSE(properties.symmetric);
  EXPECT_FALSE(properties.weighted);
  EXPECT_EQ(coo.number_of_rows, 5);
  EXPECT_EQ(coo.number_of_columns, 5);
  EXPECT_EQ(coo.number_of_nonzeros, 6);
  expect_values(coo.row_indices, {0, 1, 1, 2, 3, 4});
  expect_values(coo.column_indices, {1, 2, 3, 4, 4, 0});
  expect_values(coo.nonzero_values, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
}

TEST(io, matrix_market_expands_weighted_symmetric_graph) {
  gunrock::io::matrix_market_t<int, int, float> loader;
  auto [properties, coo] =
      loader.load(test_data_path("weighted-symmetric.mtx"));

  EXPECT_FALSE(properties.directed);
  EXPECT_TRUE(properties.symmetric);
  EXPECT_TRUE(properties.weighted);
  EXPECT_EQ(coo.number_of_nonzeros, 5);
  expect_values(coo.row_indices, {0, 0, 1, 1, 2});
  expect_values(coo.column_indices, {0, 1, 0, 2, 1});
  expect_values(coo.nonzero_values, {2.0f, 3.5f, 3.5f, -1.0f, -1.0f});
}

TEST(io, matrix_market_rejects_zero_indexed_input) {
  gunrock::io::matrix_market_t<int, int, float> loader;
  EXPECT_ANY_THROW(loader.load(test_data_path("zero-indexed.mtx")));
}
