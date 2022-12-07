#pragma once

// Includes CUDA
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <utility>

namespace cg = cooperative_groups;

inline void for_each_argument_address(void**) {}

template <typename arg_t, typename... args_t>
inline void for_each_argument_address(void** collected_addresses,
                                      arg_t&& arg,
                                      args_t&&... args) {
  collected_addresses[0] = const_cast<void*>(static_cast<const void*>(&arg));
  for_each_argument_address(collected_addresses + 1,
                            ::std::forward<args_t>(args)...);
}

/**
 * @brief Launch a given kernel using cudaLaunchCooperativeKernel API for
 * Cooperative Groups (CG). This is a C++ wrapper that makes the C-based API of
 * CG more accessible. See the example below for use:
 *
 * @note For an example use see the commented code below.
 * @note GodBolt.org link: https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXAMx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIAruiakl9ZATwDKjdAGFUtEywYhJAVlKOAMngMmABy7gBGmMQgAJykAA6oCoR2DC5uHl6%2Bicm2AoHBYSyR0XFWmDapQgRMxATp7p4%2BlpjWeQzVtQQFoRFRsZY1dQ2ZzQpD3UG9xf0xAJSWqCbEyOwcAPTrANQApABMkvHETMAsTFsCqzsaAILXN5tbAJIMyG5YCltOcgAid7f7SRBN5mTC7SROUzmAD6xBMgjwbAAdAgdpJsPdAcD3mC0ZDUKh4lEmLYAG6YaHAYhLeIKFFojEAgEHbGg8FOEy2WiEACeDMxt2YbAU8SYqy2yGA4J%2BEoJROOZIpVJpCjRACEBTcgtzgltSag8OgtvxiNDMGKENDasB3IwCFb0OhiJgFAoIPrDQAqT1zXYAdg1fp%2BmoImBY8QMofZBB5RKFYOt0IIpC2MbjrEwSKzW2tCiT/Nu2qmeoNRpNZotVuINrYggdTpdbo96G9stobUw6HrztdLtI9y2g6Hw5Ho7H48HiYI%2BwAbLOc9X%2B7cJyvV%2BPc/m9nOt1mkQvgApfTsAwO2x2u0xHT2lKrvGqNDtvDK0TK0AxxtDROM8c3PQyIOMJJ4Mgn5MN%2BEJvuMJZev%2Bs7WnMczqqe5bmsglrWradaXg2roQGg7YVKGF5Xo2Lq7HsapbFwS43GudFDiAIDjOgjEmgA7rU6B4hu07ohAuZzLuiGSBqAJBpq6zegOnpbAAAuExB4JgVBbP4TDwmhOZbMAeDkgwWwANZRMEtBbCYyRGBKZhMGpGkIC4hLEoqADSxmtFsNzKE8xokNJXxyk5ulggA4tSJi0lsEBOMFglbAAKggeCfElWlOPsGoUVsbHHPE8qpggJJbGcRmfAQCBgk4AC04RgZ2HleRcVB%2BdFRUkAmyCrK6eDhPQe5CJgYJlWCmCqKwEZgpEtCoGxPnEGZSggNJfmyQwqBRgAYiQOb6SNY30PNYJKIN5WyiwtZEbKWBbJN01In5y0EKKxwsMaGlJqmsZgqgKlGcQJkPU9rD7nmBAfUSnzfflCbVphBCfO6tR4JewGIcuMmyYDL3jM6QNQuc2Pmiwy2Y8ar2vO0Wzxushp2ngNBRMTtRAz1qDIAZ0L4LWyQCJ8LNs1snOMNz%2BkQJzkio7R6Mk1Shoc4iQupJ82mKUagvvqkkVixLWxS0zWMFc6XZsCwJA8tC4Q8qGnxiFNogXegPJCsBWwKAbdXG6bWxBNdlsuozz3Azm7ZQ/usOlagIdUBpFNNWj6z3KG4aRriEJpowGZkyByZg%2BnyLZjxBY3M2WwGHZn4BQqQV4dZ1Q4ywm4zi7BB1zR9ErpBoNR68DfGq3bfjmL122%2BzavCwoff96Og8y12o%2BKxPk/DsxjHJAAXhSoOu7UnbQh7xBmxbVsL4vk7ViDs6zruwNHiey6yu%2BoaqEc99QepRCUwI0Lr9S0IMO40KY0%2BC%2BU8S88Dr2%2BrufiZ8kxHkkMGOBWwNDgnWlRLYIAXZgMwBArMUCDwwKQnfX8odzoAObreNUq0GBfyiKgX%2B/9AGPngaJWis0KxoSrDWO03ZGy4NhqQ4g480FMQICxEA7FOLcWgbxbAuDDxCQISw08eNbKvHshXIC5JXJ/VaHiLuWdYJ32HLOKgKYZ5yy5vPIerMR7y3VjzY%2BxCuHxDISmLehtd5hlNubX2giCasGEsw48wYmT/FCdCSkU0aq0HCdBVW7gWBm1%2BiZCAQQCAyVUCmVJMkeSZMEFsEIN9mHDlSV7aU%2BUcboCeOgVQSJVDkUonzAyPxES1J1lYtmVSamqAUcUqgKT2T5O1mOPkd48CMLKd00Z4yZJ4HqV7BRQTNRYleDiaMCA4TjHWFgUkwEKTkhsCQek6J7iPGWSCK6%2Bw9il1UUiUwqI9h7E1CUs4QQUl5OtMgFMaFagv1Bp6a0pJHwaifBLY8RShyPCEKgNgWwjioB6mGVMEdzJghNCHJJrQ7qGN%2BSNZ%2BJSQhlK4HiPEewHwiRAeU8yBBGLbN2dCfZRBiB4lSQyLYqgIAhBTFwAJFKyobOpSAWlqx6WEV8hCFl6Itg8g5TykJw5IWYAIJVcKWw1TD3Io3UKhoBa2MVlilhg5HhVPNO2HJUqlhZUMKDLVqtbHCzKe/P%2BMw5qQyEAAWUiltPYnp3ULApY8QwRoCrkihi9MQHEeS8zBD2GmeSIDymSOMO0lUMW0EEhSwejTmksDdFwPYAAOWVBqdUsEkMrQ02a3QQHhMkYAwQjSpN9KsOgHKtjbCzYiOkqgEILOxY8JwOMozfD%2BE3Ous10VuVoPq4ceNa6E3en4om5LsWzuboTAd5pQwQFnIuot/rtgqM0kNQyk75pBClGgRylcQ3XLQotbFt7LSXvlBozAAE12sBTI8PGo7CYUuHOgeJiTJ1fu2KmsAHBPh6IpvGf9Q4O05tA%2B0gyJaFYCF2Ni4cM9K1IZnqhux%2Bk4ODg0Ehtx7tPH7y9vpQ%2BftMMQu2Bhc64cJ3aNMtBjWbyLjECwMQI8dHBw1PMDUCAglgCKpEymHkSIhNMBE0iMTBAJODIWUGDgCxaCcG8LwTwHAtCkFQJwYdtE0oUXSi7JYKxcQHB4KQalum1MLAMiAbwewkR7D9FwbwAAWDQXmZwxBiH6SQ%2Bb82%2BA0xwLzvAWAgC8zEJE%2Ba83eG8JIPYMQuAaC4B5v0pAdN6YMxwXgCgQAkbs1oBYcBYAwEQCgaF8Q6BRHIJQNA4Z6vRAYKSDqXBMt8DoKGARlBwiaF4OEIItQeScBs8186AB5BgtBxv2dIFgM4RhxCLfwM6So5IiuLZGhUTkaw9OpNaEN/Q3Vjj7xcFgCbvBm6IhuwsKgBgDwADUlJsWm3GG7MhBAiDEOwLgXmfvyCUGoU7uhqIGCMCgayZ3whFdgPGGH5hkCqAyeSaIzd4QGQQvp5xitOCVTE68DA0M0Ivma/EA70I/QkbQFgF8Ch66062JVSquKlXOgMKoTslUO64rmpVZiFP0rpUB4V1ohFUgOAYM4VwjQ9ABCmEUEoegcgpAECMTw1F1ftB6Cr/o1FyiVAEJ0YY8vMhG8lybjoEx9d9GiEbiYWu9CATqPb51EgFgKAs6sPQa61g8HU5p7Tp38shBe04JwVEuBIj9EiJBUVfjGdFxRFMuBCBbUBNRfyLX6BzWz3MW7Q3cflUvP0CTTnvAkfC5F0g0WZxeYTx50LM4C1%2Bi83sbwnmcth84IV4rtmS%2BkAq9VpYj1OSNYgJ2TPKs9D8F%2B6IcQgPgeKBUOoRbEPSDZSYPEB7%2BgQ%2B98W/l6bnIqeg0hhHqPMe48J8ii4PPURyKSG5cX%2BzpfzQ8coMHiLUWshIhiALUB0kA8y7w73zS8yB1y14HywHxKxL0cxi3jz9ALS828BiGSw0A0C3C6zC04EkFD2P37yH3fx/z2D/3Sw0ATywJoNoJoOomgP02INKwc1IAx2FhiyAA%3D
 * 
 * @tparam func_t type of kernel
 * @tparam args_t types of the arguments (variadic)
 * @param stream cuda stream
 * @param f function name/identifier
 * @param block_dimensions block dimension (dim3)
 * @param grid_dimensions  grid dimension (dim3)
 * @param shared_memory_bytes allocated dynamic shared memory in bytes
 * @param args all the arguments to the function f
 */
template <typename func_t, typename... args_t>
void launch_cooperative(cudaStream_t& stream,
                        const func_t& f,
                        dim3 block_dimensions,
                        dim3 grid_dimensions,
                        std::size_t shared_memory_bytes,
                        args_t&&... args) {
  constexpr const auto non_zero_num_params =
      sizeof...(args_t) == 0 ? 1 : sizeof...(args_t);
  void* argument_ptrs[non_zero_num_params];
  for_each_argument_address(argument_ptrs, ::std::forward<args_t>(args)...);
  
  cudaLaunchCooperativeKernel<func_t>(
      &f, grid_dimensions, block_dimensions,
      argument_ptrs, shared_memory_bytes, stream);
}



// __global__ void dummy_kernel(int* x, int* y, int N) {
//     int i = threadIdx.x + blockDim.x * blockIdx.x;
//     if(i < N)
//         y[i] = x[i] * i + i;
// }

// #include <thrust/device_vector.h>
// #include "launch.cuh" // include this file.

// int main(int argc, char const *argv[])
// {
//     // Some problem to use for the kernel.
//     constexpr int N = 1<<20;
//     thrust::device_vector<int> x(N, 1);
//     thrust::device_vector<int> y(N);

//     // Set-up Block & Grid dimenions.
//     // Ideally, you want Grid dimension = to number of SM (or 2*SM),
//     // and have them always be resident (persistent-kernel).
//     dim3 blockDims(128);
//     dim3 gridDims((unsigned int) ceil(N / blockDims.x));

//     // Create CUDA stream for the kernel.
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     // Launch the kernel using cooperative launch:
//     launch_cooperative(stream, // cuda stream
//         dummy_kernel, // kernel's function name
//         blockDims, // block dimension 
//         gridDims, // grid dimension 
//         0, // shared memory in bytes
//         // arguments to the kernel function (in order) 
//         x.data().get(), y.data().get(), N);
// }