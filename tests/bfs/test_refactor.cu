#include <gunrock/util/array_utils.cuh>

using namespace gunrock;
using namespace gunrock::util;

int main(int argc, char* argv[])
{
    Array1D<int, int, PINNED> test_array;
    test_array.SetName("test_array");
    test_array.Allocate(1024, HOST | DEVICE);
    test_array.EnsureSize(2048);
    test_array.Move(HOST, DEVICE);
    test_array.Release();
    return 0;
}
