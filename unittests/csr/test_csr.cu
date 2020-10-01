#include <gunrock/error.hxx>
#include <gunrock/memory.hxx>
#include <gunrock/formats/csr.hxx>

void test_csr()
{
  using namespace gunrock;
  using namespace gunrock::format;
  using namespace gunrock::memory;

  using offset_t = int;
  using index_t = int;
  using value_t = float;

  error::error_t status = cudaSuccess;

  constexpr memory_space_t space = memory_space_t::device;    // memory_space_t = device;
  using csr_type = csr_t<index_t, offset_t, value_t, space>;  // csr_t typename

  index_t r = 4, c = 4, nnz = 4;

  // let's use device_vector<type_t> for initial arrays
  thrust::device_vector<offset_t> row_offsets(r+1);
  thrust::device_vector<index_t> column_indices(nnz);
  thrust::device_vector<value_t> nonzero_values(nnz);

  // wrap it with shared_ptr<csr_type>
  std::shared_ptr<csr_type> csr_ptr(
    new csr_type{ r, c, nnz, row_offsets, column_indices, nonzero_values });
}

int
main(int argc, char** argv)
{
  test_csr();
  return;
}