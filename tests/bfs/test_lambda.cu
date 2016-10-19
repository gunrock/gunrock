// this code requires nvcc 8.0 with --expt-extended-lambda to compile
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <gunrock/util/array_utils.cuh>

using namespace std;
using namespace gunrock;

template <typename CondFunction, typename ApplyFunction>
__global__ void lambda_kernel(
    int num_elements,
    int *in__a,
    int *out_a,
    CondFunction cond,
    ApplyFunction apply)
{
    const int STRIDE = blockDim.x * gridDim.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    while ( i < num_elements)
    {
        if (cond(in__a, out_a, i))
            apply(in__a, out_a, i);
        i += STRIDE;
    }
}

int main()
{
    cudaError_t retval = cudaSuccess;
    util::Array1D<int, int> in__array;
    util::Array1D<int, int> out_array;
    util::Array1D<int, int> ref_array;
    int num_elements = 10000;

    in__array.SetName("in__array");
    out_array.SetName("out_array");
    ref_array.SetName("ref_array");
    if (retval = in__array.Allocate(num_elements, util::HOST | util::DEVICE))
        return 1;
    if (retval = out_array.Allocate(num_elements, util::HOST | util::DEVICE))
        return 2;
    if (retval = ref_array.Allocate(num_elements, util::HOST))
        return 3;

    srand(time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        in__array[i] = rand();
        out_array[i] = 0;
    }
    if (retval = in__array.Move(util::HOST, util::DEVICE))
        return 4;
    if (retval = out_array.Move(util::HOST, util::DEVICE))
        return 5;

    auto cond1 = [] __host__ __device__ (int *in__a, int *out_a, int pos)
    {
        if ((in__a[pos] & 0x1) == 0) return true;
        out_a[pos] = 1;
        return false;
    };

    auto apply1 = [] __host__ __device__ (int *in__a, int *out_a, int pos)
    {
        out_a[pos] = in__a[pos] + 10;
    };

    for (int i = 0; i < num_elements; i++)
    {
        if (cond1(in__array + 0, ref_array + 0, i))
            apply1(in__array + 0, ref_array + 0, i);
    }

    int block_size = 512;
    int grid_size = 64;
    lambda_kernel<<<grid_size, block_size>>>(
        num_elements,
        in__array.GetPointer(util::DEVICE),
        out_array.GetPointer(util::DEVICE),
        cond1, apply1);

    if (retval = out_array.Move(util::DEVICE, util::HOST))
        return 6;

    int num_errors = 0;
    for (int i = 0; i < num_elements; i++)
    {
        if (ref_array[i] != out_array[i]) num_errors ++;
    }
    cout << "num_errors = " << num_errors << endl;
    cout << "first 40 inputs : ";
    for (int i=0; i < 40; i++) cout << ((i==0)?"":", ") << in__array[i];
    cout << endl << "first 40 CPU results : ";
    for (int i=0; i < 40; i++) cout << ((i==0)?"":", ") << ref_array[i];
    cout << endl << "first 40 GPU results : ";
    for (int i=0; i < 40; i++) cout << ((i==0)?"":", ") << out_array[i];
    cout << endl;

    if (retval = in__array.Release())
        return 11;
    if (retval = out_array.Release())
        return 12;
    if (retval = ref_array.Release())
        return 13;
    return 0;
}

