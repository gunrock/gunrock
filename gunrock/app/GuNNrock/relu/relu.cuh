//
// Created by husong on 5/13/20.
//
#ifndef RELU_H
#define RELU_H

#include <gunrock/app/gcn/module.h>
#include <gunrock/util/array_utils.cuh>

using namespace gunrock;

template <typename SizeT, typename ValueT>
struct relu : module {
  util::Array1D<SizeT, ValueT> a, a_grad;
  util::Array1D<SizeT, bool> keep;
  int len;
  float *fw_time, *bw_time;

  relu(util::Array1D<SizeT, ValueT> &_a, util::Array1D<SizeT, ValueT> &_a_grad, int _len, float *fw, float *bw) :
      a(_a), a_grad(_a_grad), len(_len), fw_time(fw), bw_time(bw) {
    keep.Allocate (a.GetSize (), util::DEVICE);
  };

  virtual void forward(bool train) override {
    timer.Start ();
    dofw(train);
    timer.Stop ();
    *fw_time += timer.ElapsedMillis ();
  }

  virtual void backward() override {
    timer.Start ();
    dobw();
    timer.Stop ();
    *bw_time += timer.ElapsedMillis ();
  }

  cudaError_t dofw(bool train) {
    cudaError_t retval = cudaSuccess;

    GUARD_CU(a.ForEach(keep,
        [train]__host__ __device__(ValueT &x, bool &k) {
        if (train) k = x > 0;
        if (!k) x = 0;
      }, a.GetSize (), util::DEVICE))

    return retval;
  }

  cudaError_t dobw() {
    cudaError_t retval = cudaSuccess;

    GUARD_CU(a_grad.ForEach(keep,
        []__host__ __device__(ValueT &grad, bool &k) {
      if (!k) grad = 0;
    }, a_grad.GetSize (), util::DEVICE))

    return retval;
  }
};

#endif