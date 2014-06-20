#pragma once

#include <wrapper/cuda/kernel_policy.cuh>
#include <wrapper/cuda/kernel.cuh>

#include <wrapper/app/mad/mad_problem.cuh>
#include <wrapper/app/mad/mad_functor.cuh>

namespace wrapper {
namespace app {
namespace mad {

class MADEnactor
{
    public:

    MADEnactor() {}

    virtual ~MADEnactor() {}

    template<
    typename KernelPolicy,
    typename MADProblem>
    cudaError_t EnactMAD(
    MADProblem      *problem)
    {
        typedef MADFunctor<
            typename MADProblem::Value,
            MADProblem> MadFunctor;

        cudaError_t retval = cudaSuccess;

        int *d_in_queue;
        int queue_length = problem->num_elements;
        int *in_queue = new int[queue_length];
        if (retval = cudaMalloc(
            (void**)&d_in_queue,
            sizeof(int) * queue_length)) return retval;

        for (int i = 0; i < queue_length; ++i)
        {
            in_queue[i] = i;
        }
        if (retval = cudaMemcpy(
            d_in_queue,
            in_queue,
            sizeof(int)*queue_length,
            cudaMemcpyHostToDevice)) return retval;

        int num_block = (problem->num_elements + KernelPolicy::THREADS - 1) / KernelPolicy::THREADS;
        wrapper::cuda::Compute<KernelPolicy, MADProblem, MadFunctor>
        <<< num_block, KernelPolicy::THREADS>>>(
            d_in_queue,
            queue_length,
            problem->d_data_slice);

        return retval;
    }

    template<typename MADProblem>
    cudaError_t Enact(
        MADProblem  *problem)
        {
            typedef wrapper::cuda::KernelPolicy<
                MADProblem,
                300,
                8,
                10> MADKernelPolicy;
            return EnactMAD<MADKernelPolicy, MADProblem>(problem);
        }
};

} //mad
} //app
} //wrapper
