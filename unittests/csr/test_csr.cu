// XXX: dummy template for unit testing

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <gunrock/formats/csr.hxx>

void test_csr()
{
  using namespace gunrock;
  using namespace gunrock::format;

  using offset_t = int;
  using index_t = int;
  using value_t = float;

  error::error_t status = cudaSuccess;

  // CSR array with default constructor
  csr_t<offset_t, index_t, value_t> csr;

  // CSR array with space allocated (4x4x4)
  std::size_t r, c, nnz = 4;
  memory::memory_space_t location = memory::memory_space_t::host;
  csr_t<offset_t, index_t, value_t> _csr(r, c, nnz, location);

  // CSR array with pre-populated pointers (4x4x4)
  // V         = [ 5 8 3 6 ]
  // COL_INDEX = [ 0 1 2 1 ]
  // ROW_INDEX = [ 0 0 2 3 4 ]
  offset_t Ap[] = {0, 0, 2, 3, 4};
  index_t Aj[] = {0, 1, 2, 1};
  value_t Ax[] = {5, 8, 3, 6};
  csr_t<offset_t, index_t, value_t> __csr(r, c, nnz,
    Ap, Aj, Ax, location);
}

int
main(int argc, char** argv)
{
  test_csr();
  return;
}