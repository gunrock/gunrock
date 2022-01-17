#include <gunrock/error.hxx>  // error checking
#include <gunrock/memory.hxx>
#include <gunrock/formats/csr.hxx>

void test_csr() {
  using namespace gunrock;
  using namespace gunrock::format;
  using namespace gunrock::memory;

  using offset_t = int;
  using index_t = int;
  using value_t = float;

  constexpr memory_space_t space =
      memory_space_t::device;  // memory_space_t = device;
  using csr_type = csr_t<space, index_t, offset_t, value_t>;  // csr_t typename

  index_t r = 4, c = 4, nnz = 4;

  // wrap it with shared_ptr<csr_type>
  std::shared_ptr<csr_type> csr_ptr(new csr_type(r, c, nnz));
}

int main(int argc, char** argv) {
  test_csr();
}