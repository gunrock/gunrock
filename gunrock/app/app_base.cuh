// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sssp.cu
 *
 * @brief Simple test driver program for single source shortest path.
 */

#pragma once

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#ifdef BOOST_FOUND
#include <gunrock/util/info.cuh>
#else
#include <gunrock/util/info_rapidjson.cuh>
#endif

namespace gunrock {
namespace app {

cudaError_t UseParameters_app(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(util::UseParameters_info(parameters));

  GUARD_CU(parameters.Use<int>(
      "num-runs",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1, "Number of runs to perform the test, per parameter-set", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<double>(
      "preprocess-time",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::INTERNAL_PARAMETER,
      0.0, "Preprocessing time", __FILE__, __LINE__));

  return retval;
}

template <typename _VertexT = uint32_t, typename _SizeT = _VertexT,
          typename _ValueT = _VertexT,
          graph::GraphFlag _FLAG = graph::GRAPH_NONE,
          unsigned int _cudaHostRegisterFlag = cudaHostRegisterDefault>
struct TestGraph
    : public graph::Csr<_VertexT, _SizeT, _ValueT,
                        _FLAG &(~(graph::TypeMask - graph::HAS_CSR)),
                        _cudaHostRegisterFlag, (_FLAG & graph::HAS_CSR) != 0>,
      public graph::Coo<_VertexT, _SizeT, _ValueT,
                        _FLAG &(~(graph::TypeMask - graph::HAS_COO)),
                        _cudaHostRegisterFlag, (_FLAG & graph::HAS_COO) != 0>,
      public graph::Csc<_VertexT, _SizeT, _ValueT,
                        _FLAG &(~(graph::TypeMask - graph::HAS_CSC)),
                        _cudaHostRegisterFlag, (_FLAG & graph::HAS_CSC) != 0>,
      public graph::Gp<_VertexT, _SizeT, _ValueT, _FLAG,
                       _cudaHostRegisterFlag> {
  typedef _VertexT VertexT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  static const graph::GraphFlag FLAG = _FLAG;
  static const unsigned int cudaHostRegisterFlag = _cudaHostRegisterFlag;
  // typedef Csr<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CsrT;
  // typedef Coo<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CooT;
  // typedef Csc<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CscT;
  typedef graph::Csr<_VertexT, _SizeT, _ValueT,
                     _FLAG &(~(graph::TypeMask - graph::HAS_CSR)),
                     _cudaHostRegisterFlag, (_FLAG & graph::HAS_CSR) != 0>
      CsrT;
  typedef graph::Csc<_VertexT, _SizeT, _ValueT,
                     _FLAG &(~(graph::TypeMask - graph::HAS_CSC)),
                     _cudaHostRegisterFlag, (_FLAG & graph::HAS_CSC) != 0>
      CscT;
  typedef graph::Coo<_VertexT, _SizeT, _ValueT,
                     _FLAG &(~(graph::TypeMask - graph::HAS_COO)),
                     _cudaHostRegisterFlag, (_FLAG & graph::HAS_COO) != 0>
      CooT;
  typedef graph::Gp<_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag>
      GpT;

  SizeT nodes, edges;

  template <typename CooT_in>
  cudaError_t FromCoo(CooT_in &coo,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false,
                      bool self_coo = false) {
    cudaError_t retval = cudaSuccess;
    nodes = coo.CooT_in::CooT::nodes;
    edges = coo.CooT_in::CooT::edges;
    GUARD_CU(this->CsrT::FromCoo(coo, target, stream, quiet));
    GUARD_CU(this->CscT::FromCoo(coo, target, stream, quiet));
    if (!self_coo) GUARD_CU(this->CooT::FromCoo(coo, target, stream, quiet));
    return retval;
  }

  template <typename CsrT_in>
  cudaError_t FromCsr(CsrT_in &csr,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false,
                      bool self_csr = false) {
    typedef typename CsrT_in::CsrT CsrT_;

    cudaError_t retval = cudaSuccess;
    nodes = csr.CsrT_::nodes;
    edges = csr.CsrT_::edges;
    GUARD_CU(this->CooT::FromCsr(csr, target, stream, quiet));
    GUARD_CU(this->CscT::FromCsr(csr, target, stream, quiet));
    if (!self_csr) GUARD_CU(this->CsrT::FromCsr(csr, target, stream, quiet));
    return retval;
  }

  template <typename CscT_in>
  cudaError_t FromCsc(CscT_in &csc,
                      util::Location target = util::LOCATION_DEFAULT,
                      cudaStream_t stream = 0, bool quiet = false,
                      bool self_csc = false) {
    cudaError_t retval = cudaSuccess;
    nodes = csc.CscT::nodes;
    edges = csc.CscT::edges;
    GUARD_CU(this->CooT::FromCsc(csc, target, stream, quiet));
    GUARD_CU(this->CsrT::FromCsc(csc, target, stream, quiet));
    if (!self_csc) GUARD_CU(this->CscT::FromCsc(csc, target, stream, quiet));
    return retval;
  }

  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;

    // util::PrintMsg("GraphT::Realeasing on " +
    //    util::Location_to_string(target));
    GUARD_CU(this->CooT::Release(target));
    GUARD_CU(this->CsrT::Release(target));
    GUARD_CU(this->CscT::Release(target));
    GUARD_CU(this->GpT::Release(target));
    return retval;
  }

  CsrT &csr() { return (static_cast<CsrT *>(this))[0]; }

  CscT &csc() { return (static_cast<CscT *>(this))[0]; }

  CooT &coo() { return (static_cast<CooT *>(this))[0]; }

  double GetStddevDegree() {
    double retval = 0;
    if (FLAG & graph::HAS_CSR)
      retval = graph::GetStddevDegree(this->csr());
    else if (FLAG & graph::HAS_CSC)
      retval = graph::GetStddevDegree(this->csc());
    else if (FLAG & graph::HAS_COO)
      retval = graph::GetStddevDegree(this->coo());
    return retval;
  }

  double GetAverageDegree() {
    double retval = 0;
    if (FLAG & graph::HAS_CSR)
      retval = graph::GetAverageDegree(this->csr());
    else if (FLAG & graph::HAS_CSC)
      retval = graph::GetAverageDegree(this->csc());
    else if (FLAG & graph::HAS_COO)
      retval = graph::GetAverageDegree(this->coo());
    return retval;
  }

  __host__ __device__ SizeT GetNeighborListLength(const VertexT &v) const {
    SizeT retval = 0;
    if (FLAG & graph::HAS_CSR)
      retval = CsrT::GetNeighborListLength(v);
    else if (FLAG & graph::HAS_CSC)
      retval = CscT::GetNeighborListLength(v);
    else if (FLAG & graph::HAS_COO)
      retval = CooT::GetNeighborListLength(v);
    return retval;
  }

  cudaError_t Move(util::Location source, util::Location target,
                   cudaStream_t stream = 0) {
    cudaError_t retval = cudaSuccess;

    if (FLAG & graph::HAS_CSR) GUARD_CU(CsrT::Move(source, target, stream));
    if (FLAG & graph::HAS_CSC) GUARD_CU(CscT::Move(source, target, stream));
    if (FLAG & graph::HAS_COO) GUARD_CU(CooT::Move(source, target, stream));

    return retval;
  }

  cudaError_t Display(std::string graph_prefix = "", SizeT nodes_to_show = 40,
                      bool with_edge_values = true) {
    cudaError_t retval = cudaSuccess;

    if (FLAG & graph::HAS_CSR)
      GUARD_CU(CsrT::Display(graph_prefix, nodes_to_show, with_edge_values));
    if (FLAG & graph::HAS_CSC)
      GUARD_CU(CscT::Display(graph_prefix, nodes_to_show, with_edge_values));
    if (FLAG & graph::HAS_COO)
      GUARD_CU(CooT::Display(graph_prefix, nodes_to_show, with_edge_values));

    return retval;
  }

  template <typename ArrayT>
  cudaError_t GetHistogram(ArrayT &histogram) {
    cudaError_t retval = cudaSuccess;

    if (FLAG & graph::HAS_CSR) {
      GUARD_CU(GetHistogram(csr(), histogram));
    } else if (FLAG & graph::HAS_CSC) {
      GUARD_CU(GetHistogram(csc(), histogram));
    } else if (FLAG & graph::HAS_COO) {
      GUARD_CU(GetHistogram(coo(), histogram));
    }

    return retval;
  }
};

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
