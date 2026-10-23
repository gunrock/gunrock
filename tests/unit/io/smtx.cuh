#include <gunrock/io/smtx.hxx>

#include <gtest/gtest.h>

namespace {

std::string write_smtx_fixture(char const* filename, char const* contents) {
  auto path = std::filesystem::temp_directory_path() / filename;
  std::ofstream output(path);
  output << contents;
  output.close();
  return path.string();
}

}  // namespace

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

TEST(io, smtx_loads_csv_header_and_trims_fields) {
  EXPECT_EQ(gunrock::io::leading_trim("   4, 4, 2"), "4, 4, 2");
  EXPECT_EQ(gunrock::io::leading_trim("   "), "");

  auto path = write_smtx_fixture("gunrock-csv.smtx",
                                 "% generated fixture\n"
                                 " 3, 3, 2\n"
                                 "0 1 1 2\n"
                                 "1 2\n");
  gunrock::io::smtx_t<int, int, float> loader;
  auto csr = loader.load(path, true);
  EXPECT_EQ(csr.number_of_rows, 3);
  EXPECT_EQ(csr.number_of_columns, 3);
  EXPECT_EQ(csr.number_of_nonzeros, 2);
  expect_values(csr.row_offsets, {0, 1, 1, 2});
  expect_values(csr.column_indices, {1, 2});
  std::filesystem::remove(path);
}

TEST(io, smtx_reports_missing_nonzeros_and_type_overflow) {
  gunrock::io::smtx_t<int, int, float> loader;
  auto missing =
      (std::filesystem::temp_directory_path() / "gunrock-missing.smtx")
          .string();
  std::filesystem::remove(missing);
  EXPECT_THROW(loader.load(missing), std::runtime_error);

  auto nonzeros = write_smtx_fixture("gunrock-nonzero-mismatch.smtx",
                                     "2 2 2\n"
                                     "0 1 1\n"
                                     "1\n");
  EXPECT_THROW(loader.load(nonzeros), std::invalid_argument);
  std::filesystem::remove(nonzeros);

  auto vertex_overflow = write_smtx_fixture("gunrock-smtx-vertex-overflow.smtx",
                                            "127 1 0\n"
                                            "0\n"
                                            "\n");
  gunrock::io::smtx_t<signed char, int, float> narrow_vertex_loader;
  EXPECT_ANY_THROW(narrow_vertex_loader.load(vertex_overflow));
  std::filesystem::remove(vertex_overflow);

  auto edge_overflow = write_smtx_fixture("gunrock-smtx-edge-overflow.smtx",
                                          "1 1 127\n"
                                          "0 0\n"
                                          "\n");
  gunrock::io::smtx_t<int, signed char, float> narrow_edge_loader;
  EXPECT_ANY_THROW(narrow_edge_loader.load(edge_overflow));
  std::filesystem::remove(edge_overflow);
}