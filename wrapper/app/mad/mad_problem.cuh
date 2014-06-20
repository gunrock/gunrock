#pragma once
namespace wrapper {
namespace app {
namespace mad {

template<typename    _Value>
struct MADProblem
{
    typedef _Value      Value;
    struct DataSlice
    {
        Value    *d_results;
    };

    int         num_elements;

    DataSlice   *data_slice;
    DataSlice   *d_data_slice;

    MADProblem(): num_elements(0)
    {}

    ~MADProblem()
    {
        if (data_slice->d_results)  cudaFree(data_slice->d_results);
        if (d_data_slice) cudaFree(d_data_slice);
        if (data_slice) delete data_slice;
    }

    cudaError_t Extract(Value *h_results)
    {
        cudaError_t retval = cudaSuccess;
        do {
            if (retval = cudaMemcpy(h_results,
                                    data_slice->d_results,
                                    sizeof(Value) * num_elements,
                                    cudaMemcpyDeviceToHost)) break;
        } while(0);

        return retval;
    }

    cudaError_t Init(int num_elem, Value *origin_elements)
    {
        num_elements = num_elem;
        cudaError_t retval = cudaSuccess;
        data_slice = new DataSlice;
        if (retval = cudaMalloc(
                                    (void**)&d_data_slice,
                                    sizeof(DataSlice))) return retval;

        do {
                Value   *d_results;
                if (retval = cudaMalloc(
                        (void**)&d_results,
                        num_elements * sizeof(Value))) return retval;

                if (retval = cudaMemcpy(d_results,
                                    origin_elements,
                                    sizeof(Value) * num_elem,
                                    cudaMemcpyHostToDevice)) break;
                data_slice->d_results = d_results;

                if (retval = cudaMemcpy(
                            d_data_slice,
                            data_slice,
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice)) break;
            } while (0);

            return retval;
    }
};


} //mad
} //app
} //wrapper
