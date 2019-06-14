#include <iostream>
#include <gunrock/util/test_utils.h>
#include <gunrock/util/io/cub_io.cuh>
#include <gunrock/oprtr/1D_oprtr/for.cuh>

cudaError_t RepeatForTest() {
  cudaError_t retval = cudaSuccess;

  typedef uint32_t T;
  typedef uint32_t SizeT;

  SizeT loop_size = 1024000;
  int num_repeats = 1000;
  cudaStream_t stream = 0;
  int num_launchs = 20;

  // gunrock::util::Array1D<SizeT, T> host_sfor_results;
  // gunrock::util::Array1D<SizeT, T> host_rfor_results;
  // gunrock::util::Array1D<SizeT, T> device_sfor_results;
  // gunrock::util::Array1D<SizeT, T> device_rfor_results;
  gunrock::util::Array1D<SizeT, T> results;
  gunrock::util::Array1D<SizeT, int> counter;

  // host_sfor_results  .Allocate(loop_size, gunrock::util::HOST);
  // host_rfor_results  .Allocate(loop_size, gunrock::util::HOST);
  // device_sfor_results.Allocate(loop_size, gunrock::util::HOST |
  // gunrock::util::DEVICE); device_rfor_results.Allocate(loop_size,
  // gunrock::util::HOST | gunrock::util::DEVICE);
  results.Allocate(loop_size, gunrock::util::HOST | gunrock::util::DEVICE);
  counter.Allocate(2, gunrock::util::HOST | gunrock::util::DEVICE);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  gunrock::util::CpuTimer cpu_timer;

  T extra = (T)num_repeats * (num_repeats - 1) / 2;  // * num_launchs;
  auto op = [counter, results] __host__ __device__(const int &r,
                                                   const SizeT &i) {
    if (i == 0) {
      // counter[(r + 1) % 2] = r + 1;
      gunrock::Store<cub::STORE_WB>(counter + ((r + 1) % 2), r + 1);
    }

    if (r == 0)
      results[i] = i;
    else
      results[i] += gunrock::Load<cub::LOAD_CA>(counter + (r % 2));
  };

  std::string test_names[] = {"CPU OpenMP", "GPU Cooperactive Group",
                              "GPU CUDA Graph", "GPU stacked kernels"};
  for (int t = 0; t < 4; t++) {
    auto target = (t < 1) ? gunrock::util::HOST : gunrock::util::DEVICE;
    counter[0] = 0;
    counter[1] = 1;

    counter.Move(gunrock::util::HOST, target, 2, 0, stream);
    cpu_timer.Start();

    for (int i = 0; i < num_launchs; i++) {
      gunrock::oprtr::RepeatFor(
          op, num_repeats, loop_size, target, stream,
          gunrock::util::PreDefinedValues<int>::InvalidValue,
          gunrock::util::PreDefinedValues<int>::InvalidValue, t - 1);
    }

    cudaStreamSynchronize(stream);
    cpu_timer.Stop();
    results.Move(target, gunrock::util::HOST, loop_size, 0, stream);

    std::cout << test_names[t] << " : ";
    SizeT num_errors = 0;
    for (SizeT i = 0; i < loop_size; i++) {
      if (results[i] != extra + i) {
        if (num_errors == 0)
          std::cout << " FAIL, results[" << i << "] = " << results[i]
                    << ", should be " << i + extra << std::endl;
        num_errors++;
      }
    }
    if (num_errors == 0) {
      std::cout << "PASS, time = " << cpu_timer.ElapsedMillis() << std::endl;
    } else {
      std::cout << "num_errors = " << num_errors << std::endl;
      retval = cudaErrorUnknown;
      break;
    }
  }

  GUARD_CU(counter.Release());
  GUARD_CU(results.Release());

  return retval;
}
