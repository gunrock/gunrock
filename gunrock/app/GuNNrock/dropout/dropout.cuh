//
// Created by husong on 5/13/20.
//
#ifndef DROPOUT_H
#define DROPOUT_H

#include <gunrock/app/gcn/module.h>
#include <gunrock/util/array_utils.cuh>

template <typename SizeT, typename ValueT>
struct dropout : module {
  typedef gunrock::util::Array1D<SizeT, ValueT> Array;
  Array mask, data, *grad;
  ValueT p;
  curandGenerator_t *gen;
  float *fw_time, *bw_time;
  dropout(Array _data, Array *_grad, ValueT _p, curandGenerator_t *_gen, float *fw, float *bw) :
      p(_p), gen(_gen), fw_time(fw), bw_time(bw) {
    data = _data;
    grad = _grad;
    mask.Allocate(data.GetSize (), util::DEVICE);
  };
  virtual void forward(bool train) override {
    timer.Start ();
    if (train) {
      dofw();
    }
    timer.Stop ();
    *fw_time += timer.ElapsedMillis ();
  }
  virtual void backward() override {
    timer.Start ();
    dobw();
    timer.Stop ();
    *bw_time += timer.ElapsedMillis ();
  }
  cudaError_t dofw() {
    auto retval = cudaSuccess;
    curandGenerateUniformDouble(*gen, mask.GetPointer(util::DEVICE), mask.GetSize ());
//      mask.Print ("dropout_mask: ", 10, util::DEVICE);
//      SizeT len = mask.GetSize ();
//      GUARD_CU (mask.ForAll([len]__host__ __device__(ValueT *x, SizeT &i) {
//        if (i & 1) x[i] = 0.0;
//        else x[i] = 1.0;
//      }))
//      ValueT tmp[data.GetSize ()];
//      std::mt19937 rng(std::time (nullptr));
//      for (int i = 0; i < data.GetSize (); i++) tmp[i] = rng() * 1.0 / rng.max ();
//      mask.SetPointer (tmp, mask.GetSize (), util::HOST);
//      mask.Move(util::HOST, util::DEVICE);
//      mask.UnSetPointer (util::HOST);

    ValueT scale = 1 / (1 - p);
    auto &p = this->p;
    GUARD_CU (mask.ForEach (data,
                            [p, scale]__host__ __device__(ValueT &x, ValueT &d) {
      d *= x >= p ? scale : 0;
    }, mask.GetSize (), util::DEVICE))
//      data.Print ("dropout_data: ", 10, util::DEVICE);
    return retval;
  }

  cudaError_t dobw() {
    auto retval = cudaSuccess;
    if (!grad) return retval;
    ValueT scale = 1 / (1 - p);
    auto &p = this->p;
    GUARD_CU (mask.ForEach (*grad,
                            [p, scale]__host__ __device__(ValueT &x, ValueT &g) {
      g *= x >= p ? scale : 0;
    }))
    return retval;
  }
};

#endif