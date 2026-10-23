#include <gunrock/io/matrix_market.hxx>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string test_data_path(char const* filename) {
  return (std::filesystem::path(GUNROCK_TEST_DATA_DIR) / filename).string();
}

std::string write_matrix_market_fixture(char const* filename,
                                        char const* contents) {
  auto path = std::filesystem::temp_directory_path() / filename;
  std::ofstream output(path);
  output << contents;
  output.close();
  return path.string();
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

TEST(io, matrix_market_loads_integer_weights) {
  auto path = write_matrix_market_fixture(
      "gunrock-integer.mtx",
      "%%MatrixMarket matrix coordinate integer general\n"
      "2 2 2\n"
      "1 2 -7\n"
      "2 1 11\n");
  gunrock::io::matrix_market_t<int, int, int> loader;
  auto [properties, coo] = loader.load(path);

  EXPECT_TRUE(properties.weighted);
  EXPECT_EQ(loader.data, gunrock::io::matrix_market_data_t::integer);
  expect_values(coo.row_indices, {0, 1});
  expect_values(coo.column_indices, {1, 0});
  expect_values(coo.nonzero_values, {-7, 11});
  std::filesystem::remove(path);
}

TEST(io, matrix_market_rejects_truncated_and_overflowing_inputs) {
  auto truncated = write_matrix_market_fixture(
      "gunrock-truncated.mtx",
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 1\n"
      "1 2\n");
  gunrock::io::matrix_market_t<int, int, float> loader;
  EXPECT_ANY_THROW(loader.load(truncated));
  std::filesystem::remove(truncated);

  auto vertex_overflow = write_matrix_market_fixture(
      "gunrock-vertex-overflow.mtx",
      "%%MatrixMarket matrix coordinate pattern general\n"
      "127 1 0\n");
  gunrock::io::matrix_market_t<signed char, int, float> narrow_vertex_loader;
  EXPECT_ANY_THROW(narrow_vertex_loader.load(vertex_overflow));
  std::filesystem::remove(vertex_overflow);

  auto edge_overflow = write_matrix_market_fixture(
      "gunrock-edge-overflow.mtx",
      "%%MatrixMarket matrix coordinate pattern general\n"
      "1 1 127\n");
  gunrock::io::matrix_market_t<int, signed char, float> narrow_edge_loader;
  EXPECT_ANY_THROW(narrow_edge_loader.load(edge_overflow));
  std::filesystem::remove(edge_overflow);
}

TEST(io, matrix_market_rejects_unsupported_encodings) {
  auto missing =
      (std::filesystem::temp_directory_path() / "gunrock-missing-market.mtx")
          .string();
  std::filesystem::remove(missing);
  EXPECT_EXIT((gunrock::io::matrix_market_t<int, int, float>{}.load(missing)),
              testing::ExitedWithCode(1), "File could not be opened");

  auto invalid_banner = write_matrix_market_fixture(
      "gunrock-invalid-banner.mtx", "not a matrix\n");
  EXPECT_EXIT(
      (gunrock::io::matrix_market_t<int, int, float>{}.load(invalid_banner)),
      testing::ExitedWithCode(1), "Could not process Matrix Market banner");
  std::filesystem::remove(invalid_banner);

  auto invalid_size = write_matrix_market_fixture(
      "gunrock-invalid-size.mtx",
      "%%MatrixMarket matrix coordinate real general\n"
      "not dimensions\n");
  EXPECT_EXIT(
      (gunrock::io::matrix_market_t<int, int, float>{}.load(invalid_size)),
      testing::ExitedWithCode(1), "Could not read file info");
  std::filesystem::remove(invalid_size);

  auto dense =
      write_matrix_market_fixture("gunrock-dense.mtx",
                                  "%%MatrixMarket matrix array real general\n"
                                  "2 2\n"
                                  "1\n2\n3\n4\n");
  EXPECT_EXIT((gunrock::io::matrix_market_t<int, int, float>{}.load(dense)),
              testing::ExitedWithCode(1), "File is not a sparse matrix");
  std::filesystem::remove(dense);

  auto complex = write_matrix_market_fixture(
      "gunrock-complex.mtx",
      "%%MatrixMarket matrix coordinate complex general\n"
      "2 2 1\n"
      "1 2 1.0 2.0\n");
  EXPECT_EXIT((gunrock::io::matrix_market_t<int, int, float>{}.load(complex)),
              testing::ExitedWithCode(1),
              "Unrecognized matrix market format type");
  std::filesystem::remove(complex);
}
