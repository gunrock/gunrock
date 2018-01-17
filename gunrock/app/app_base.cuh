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
    #include <gunrock/util/info_noboost.cuh>
#endif

namespace gunrock {
namespace app {

cudaError_t UseParameters_app(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(util::UseParameters_info(parameters));

    GUARD_CU(parameters.Use<int>(
        "num-runs",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        1,
        "Number of runs to perform the test, per parameter-set",
        __FILE__, __LINE__));
    return retval;
}

template <
    typename _VertexT = uint32_t,
    typename _SizeT   = _VertexT,
    typename _ValueT  = _VertexT,
    graph::GraphFlag _FLAG   = graph::GRAPH_NONE,
    unsigned int _cudaHostRegisterFlag = cudaHostRegisterDefault>
struct TestGraph :
    public graph::Csr<_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag,
        (_FLAG & graph::HAS_CSR) != 0>,
    public graph::Coo<_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag,
        (_FLAG & graph::HAS_COO) != 0>,
    public graph::Csc<_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag,
        (_FLAG & graph::HAS_CSC) != 0>,
    public graph::Gp <_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag>
{
    typedef _VertexT VertexT;
    typedef _SizeT   SizeT;
    typedef _ValueT  ValueT;
    static const graph::GraphFlag FLAG = _FLAG;
    static const unsigned int cudaHostRegisterFlag = _cudaHostRegisterFlag;
    //typedef Csr<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CsrT;
    //typedef Coo<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CooT;
    //typedef Csc<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CscT;
    typedef graph::Csr<_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag,
        (_FLAG & graph::HAS_CSR) != 0> CsrT;
    typedef graph::Csc<_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag,
        (_FLAG & graph::HAS_CSC) != 0> CscT;
    typedef graph::Coo<_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag,
        (_FLAG & graph::HAS_COO) != 0> CooT;
    typedef graph::Gp <_VertexT, _SizeT, _ValueT, _FLAG, _cudaHostRegisterFlag>
        GpT;

    SizeT nodes, edges;

    template <typename CooT_in>
    cudaError_t FromCoo(CooT_in &coo, bool self_coo = false)
    {
        cudaError_t retval = cudaSuccess;
        nodes = coo.CooT_in::CooT::nodes;
        edges = coo.CooT_in::CooT::edges;
        GUARD_CU(this -> CsrT::FromCoo(coo));
        GUARD_CU(this -> CscT::FromCoo(coo));
        if (!self_coo)
            GUARD_CU(this -> CooT::FromCoo(coo));
        return retval;
    }

    template <typename CsrT_in>
    cudaError_t FromCsr(CsrT_in &csr, bool self_csr = false)
    {
        cudaError_t retval = cudaSuccess;
        nodes = csr.CsrT::nodes;
        edges = csr.CsrT::edges;
        GUARD_CU(this -> CooT::FromCsr(csr));
        GUARD_CU(this -> CscT::FromCsr(csr));
        if (!self_csr)
            GUARD_CU(this -> CsrT::FromCsr(csr));
        return retval;
    }

    template <typename CscT_in>
    cudaError_t FromCsc(CscT_in &csc, bool self_csc = false)
    {
        cudaError_t retval = cudaSuccess;
        nodes = csc.CscT::nodes;
        edges = csc.CscT::edges;
        GUARD_CU(this -> CooT::FromCsc(csc));
        GUARD_CU(this -> CsrT::FromCsc(csc));
        if (!self_csc)
            GUARD_CU(this -> CscT::FromCsr(csc));
        return retval;
    }

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;

        //util::PrintMsg("GraphT::Realeasing on " +
        //    util::Location_to_string(target));
        GUARD_CU(this -> CooT::Release(target));
        GUARD_CU(this -> CsrT::Release(target));
        GUARD_CU(this -> CscT::Release(target));
        GUARD_CU(this ->  GpT::Release(target));
        return retval;
    }

    CsrT &csr()
    {
        return (static_cast<CsrT*>(this))[0];
    }

    CscT &csc()
    {
        return (static_cast<CscT*>(this))[0];
    }

    CooT &coo()
    {
        return (static_cast<CooT*>(this))[0];
    }

    double GetStddevDegree()
    {
        double retval = 0;
        if (FLAG & graph::HAS_CSR)
            retval = graph::GetStddevDegree(this -> csr());
        else if (FLAG & graph::HAS_CSC)
            retval = graph::GetStddevDegree(this -> csc());
        else if (FLAG & graph::HAS_COO)
            retval = graph::GetStddevDegree(this -> coo());
        return retval;
    }

    double GetAverageDegree()
    {
        double retval = 0;
        if (FLAG & graph::HAS_CSR)
            retval = graph::GetAverageDegree(this -> csr());
        else if (FLAG & graph::HAS_CSC)
            retval = graph::GetAverageDegree(this -> csc());
        else if (FLAG & graph::HAS_COO)
            retval = graph::GetAverageDegree(this -> coo());
        return retval;
    }

    SizeT GetNeighborListLength(const VertexT &v)
    {
        SizeT retval = 0;
        if (FLAG & graph::HAS_CSR)
            retval = CsrT::GetNeighborListLength(v);
        else if (FLAG & graph::HAS_CSC)
            retval = CscT::GetNeighborListLength(v);
        else if (FLAG & graph::HAS_COO)
            retval = CooT::GetNeighborListLength(v);
        return retval;
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
