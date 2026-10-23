#include <gunrock/formats/formats.hxx>  // csr support
#include <gunrock/memory.hxx>

#include <gtest/gtest.h>

TEST(formats, format) {
  using namespace gunrock;
  using namespace gunrock::format;
  using namespace memory;

  using offset_t = int;
  using index_t = int;
  using value_t = float;

  // CSR, CSC, COO classes with default constructors
  csr_t<memory_space_t::host, index_t, offset_t, value_t> csr;
  csc_t<memory_space_t::host, index_t, offset_t, value_t> csc;
  coo_t<memory_space_t::host, index_t, index_t, value_t> coo;
}