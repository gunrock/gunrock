#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

#include <gunrock/app/GuNNrock/module.h>
#include <gunrock/util/array_utils.cuh>

using namespace gunrock;

template <typename SizeT, typename ValueT>
struct mat_mul : module {
//  typedef app::graphsum::Problem<GraphT> ProblemT;
//  typedef app::graphsum::Enactor<ProblemT> EnactorT;
  util::Array1D<SizeT, ValueT> b, c, b_grad, c_grad, a, a_grad;
//  ProblemT *problem;
//  EnactorT *enactor;
  int m, n, p;
  float *fw_time, *bw_time;

  mat_mul(util::Array1D<SizeT, ValueT> &_a, util::Array1D<SizeT, ValueT> &_a_grad,
          util::Array1D<SizeT, ValueT> &_b, util::Array1D<SizeT, ValueT> &_b_grad,
          util::Array1D<SizeT, ValueT> &_c, util::Array1D<SizeT, ValueT> &_c_grad, int _m, int _n, int _p,
          float *fw, float *bw) : fw_time(fw), bw_time(bw),
      a(_a), b(_b), c(_c), a_grad(_a_grad), b_grad(_b_grad), c_grad(_c_grad), m(_m), n(_n), p(_p) {}

  virtual void forward(bool train) override {
    timer.Start ();
    dofw();
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
    cudaError_t retval = cudaSuccess;
    auto &b = this->b, &c = this->c;
    auto &p = this->p, &n = this->n;

    GUARD_CU(
        c.ForEach([]__host__ __device__(ValueT &x) {
      x = 0;
    })
    )

    // Calculating matrix multiplication
    // which matrices are multiplied?
    // TODO gunrock optimised sparse matrix multiplication to replace a manual method
    GUARD_CU(a.ForAll(
        [b, c, p, n]__host__ __device__(ValueT *a_, const SizeT pos) {
          // a x b = c

          // n is hidden dimension
          // m is number of nodes
                   // p is out_dim
      int i = pos / n, j = pos % n;
      // i is the row and j is the col
      // 
      for (int k = 0; k < p; k++) {
        // the position that is being populated
        // is the outputs (i, k)
        // i being the row --> node
        // k being the feature --> output_dimension

        // for all k,
        // the value is added is:
        // the entry from a's (i, j)
        // i being the row --> node
        // j being the col --> input_dimension
        // multiplied by b's (j,k)
        // j being the input dimension
        // and k being the output feature
        atomicAdd(c + i * p + k, a_[pos] * b[j * p + k]);
        //            printf("i: %d\n", i * p + k);
      }
    }, m * n, util::DEVICE));

    return retval;
  }

  cudaError_t dobw() {
    cudaError_t retval = cudaSuccess;

    GUARD_CU(
        a_grad.ForEach (
            []__host__ __device__(ValueT &x) { x = 0; }
    )
    )

    GUARD_CU(
        b_grad.ForEach (
            []__host__ __device__(ValueT &x) { x = 0; }
    )
    )

    auto &b_grad = this->b_grad, &c_grad = this->c_grad, &b = this->b, &a = this->a;
    auto &p = this->p, &n = this->n;
    // Calculating matrix multiplication
    GUARD_CU(a_grad.ForAll(
        [b_grad, c_grad, p, n, a, b]__host__ __device__(ValueT *a_, const SizeT pos) {
      int i = pos / n, j = pos % n;
      ValueT tmp = 0;
      for (int k = 0; k < p; k++) {
        tmp += c_grad[i * p + k] * b[j * p + k];
        atomicAdd(b_grad + j * p + k, a[pos] * c_grad[i * p + k]);
//          printf("c[%d], b[%d], a[%d]\n", i * p + k, j * p + k, pos);
      }
      a_[i * n + j] = tmp;
    }, m * n, util::DEVICE));
//      b_grad.Print();

    return retval;
  }
};

#endif