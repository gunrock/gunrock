#include <gunrock/cuda/launch_box.hxx>

using namespace gunrock::cuda;

int main(void) {
  typedef launch_box_t<
    ampere<16, 64, 0>,
    turing<16, 32, 0>,
    kepler<32, 64, 0>
  > launch_t;
}
