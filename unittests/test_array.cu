#include <gunrock/data_structs/array.cuh>
typedef gunrock::error_t error_t;

error_t
test_array()
{
  using namespace gunrock;

  error_t retval = cudaSuccess;
  size_t N = 128;
  datastruct::dense::array<int, N> a;

  return retval;
}

int
main(int argc, char** argv)
{
  return test_array();
}