#include <gunrock/formats/formats.hxx>

#include <gtest/gtest.h>

#include <initializer_list>

namespace {

template <typename range_t, typename value_t>
void expect_values(range_t const& actual,
                   std::initializer_list<value_t> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  std::size_t index = 0;
  for (auto value : expected) {
    EXPECT_EQ(actual[index], value) << "at index " << index;
    ++index;
  }
}

using host_coo_t =
    gunrock::format::coo_t<gunrock::memory_space_t::host, int, int, float>;
using host_csr_t =
    gunrock::format::csr_t<gunrock::memory_space_t::host, int, int, float>;
using host_csc_t =
    gunrock::format::csc_t<gunrock::memory_space_t::host, int, int, float>;

host_coo_t make_test_coo() {
  host_coo_t coo(4, 4, 5);
  coo.row_indices = {0, 0, 1, 2, 2};
  coo.column_indices = {1, 2, 2, 0, 3};
  coo.nonzero_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  return coo;
}

}  // namespace

TEST(formats, default_constructors_are_empty) {
  host_coo_t coo;
  host_csr_t csr;
  host_csc_t csc;

  EXPECT_EQ(coo.number_of_nonzeros, 0);
  EXPECT_EQ(csr.number_of_nonzeros, 0);
  EXPECT_EQ(csc.number_of_nonzeros, 0);
  EXPECT_TRUE(coo.row_indices.empty());
  EXPECT_TRUE(csr.row_offsets.empty());
  EXPECT_TRUE(csc.column_offsets.empty());
}

TEST(formats, host_conversions_preserve_topology_and_weights) {
  auto coo = make_test_coo();

  host_csr_t csr;
  csr.from_coo(coo);
  EXPECT_EQ(csr.number_of_rows, 4);
  EXPECT_EQ(csr.number_of_columns, 4);
  EXPECT_EQ(csr.number_of_nonzeros, 5);
  expect_values(csr.row_offsets, {0, 2, 3, 5, 5});
  expect_values(csr.column_indices, {1, 2, 2, 0, 3});
  expect_values(csr.nonzero_values, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  host_coo_t round_trip;
  round_trip.from_csr(csr);
  expect_values(round_trip.row_indices, {0, 0, 1, 2, 2});
  expect_values(round_trip.column_indices, {1, 2, 2, 0, 3});
  expect_values(round_trip.nonzero_values, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  host_csc_t csc;
  csc.from_csr(csr);
  expect_values(csc.column_offsets, {0, 1, 2, 4, 5});
  expect_values(csc.row_indices, {2, 0, 0, 1, 2});
  expect_values(csc.nonzero_values, {4.0f, 1.0f, 2.0f, 3.0f, 5.0f});
}
