#pragma once
namespace wrapper {
namespace cuda {

template <
    typename _ProblemData,
    int      _CUDA_ARCH,
    int      _MIN_CTA_OCCUPANCY,
    int      _LOG_THREADS>

struct KernelPolicy
{
    typedef _ProblemData    ProblemData;
    typedef typename ProblemData::Value Value;

    enum {
        CUDA_ARCH           = _CUDA_ARCH,
        MIN_CTA_OCCUPANCY   = _MIN_CTA_OCCUPANCY,
        LOG_THREADS         = _LOG_THREADS,
        THREADS             = 1 << LOG_THREADS,
        };

    struct SmemStorage
    {
        struct {
            Value   s_values[THREADS];
            };
    };

};

} //cuda
} //wrapper
