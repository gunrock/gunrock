#ifndef CUDA_COMMON
#define CUDA_COMMON
#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x>>5))
#define LANE_ ((threadIdx.x+blockIdx.x*blockDim.x)&31)
#define MAX_U32 (~(uint32_t)0)
#define MAX_SZ (~(size_t)0)

__device__ unsigned int lanemask_lt() {
       int lane = threadIdx.x & 31;
          return (1<<lane) - 1;;
}

#if defined(_WIN64) || defined(__LP64__)
# define PTR_CONSTRAINT "l"
#else
# define PTR_CONSTRAINT "r"
#endif
__device__ int isShared(void *ptr)
{
    int res;
    asm("{"
        ".reg .pred p;\n\t"
        "isspacep.shared p, %1;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}" :
        "=r"(res): PTR_CONSTRAINT(ptr));
        return res;
}
 
#endif

#ifndef ROUND_UTIL
#define ROUND_UTIL
template<class Size>
__device__ __host__ Size roundup_power2(Size num)
{
    if(num && !(num&(num-1)))
        return num;
    num--;
    for(int i=1; i<=sizeof(Size)*4; i=i*2)
        num |= num>>i;
    return ++num;
}

template<class Size>
__device__ __host__ Size rounddown_power2(Size num)
{
    if(num && !(num&(num-1)))
        return num;
    num--;
    for(int i=1; i<=sizeof(Size)*4; i=i*2)
        num |= num>>i;
    return (++num)>>1;
}
#endif

#ifndef align_up_yx
#define align_up_yx(num, align) \
        (((num) + ((align) - 1)) & ~((align) - 1))

//template<typename T, typename Y>
//__device__ __host__ T Align_up(T num, Y align)
//{
//    return (num/align+(num%align !=0))*align;
//}
#endif

#ifndef align_down
#define align_down(num, align) \
        ((num) & ~((align) - 1))
#endif



