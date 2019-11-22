// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_knn.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

// KNN includes
#include <gunrock/app/knn/knn_app.cu>
#include <gunrock/app/knn/knn_helpers.cuh>

// App and test base includes
#include <gunrock/app/test_base.cuh>

//#define KNN_DEBUG 1
#ifdef KNN_DEBUG
    #define debug(a...) fprintf(stderr, a)
#else
    #define debug(a...)
#endif

using namespace gunrock;

namespace APP_NAMESPACE = app::knn;

/**
 * @brief Compute euclidean distance
 * @param dim Number of dimensions (2D, 3D ... ND)
 * @param points Points array to get the x, y, z...
 * @param p1 and p2 points to be compared
 * info \return distance value
 */
 template<typename SizeT, typename ValueT>
 __device__ __host__
 ValueT distances(const SizeT dim, ValueT* points, SizeT p1, SizeT p2) {
     // Get dimensional of labels
     ValueT result = (ValueT) 0;
     // p1 = (x_1, x_2, ..., x_dim)
     // p2 = (y_1, y_2, ..., y_dim)
     for (int i=0; i<dim; ++i){
         //(x_i - y_i)^2
         ValueT diff = points[p1 * dim + i] - points[p2 * dim + i];
         result += diff*diff;
     }
     return result;
 }

/******************************************************************************
 * Main
 ******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct {
  /**
   * @brief the actual main function, after type switching
   * @tparam VertexT    Type of vertex identifier
   * @tparam SizeT      Type of graph size, i.e. type of edge identifier
   * @tparam ValueT     Type of edge values
   * @param  parameters Command line parameters
   * @param  v,s,val    Place holders for type deduction
   * \return cudaError_t error message(s), if any
   */
  template <typename VertexT,  // Use int as the vertex identifier
            typename SizeT,    // Use int as the graph size type
            typename ValueT>   // Use int as the value type
  cudaError_t
  operator()(util::Parameters& parameters, VertexT v, SizeT s, ValueT val) {
    // CLI parameters
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    // Get n dimension tuplets
    std::string labels_file = parameters.Get<std::string>("labels-file");
    util::PrintMsg("Points File Input: " + labels_file, !quiet);

    typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR>
        GraphT;
    // Creating empty graph
    GraphT graph;

    cudaError_t retval = cudaSuccess;

    // Initialization of the points array
    util::Array1D<SizeT, ValueT> points;
    //Initialization is moved to gunrock::graphio::labels::Read ... ReadLabelsStream
    //GUARD_CU(points.Allocate(n*dim, util::HOST));
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    // graphio::labels is setting "n" and "dim"
    retval = gunrock::graphio::labels::Read(parameters, points);
    if (retval){
        util::PrintMsg("Reading error\n");
        return retval;
    }
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // Get number of points
    SizeT n = parameters.Get<SizeT>("n");
   
    // Get dimensional of space
    SizeT dim = parameters.Get<SizeT>("dim");

    // Get number of nearest neighbors, default k = 10
    SizeT k = parameters.Get<SizeT>("k");

    if (k >= n){
        util::PrintMsg("k has to be at most n-1", !quiet);
        return retval;
    }
 
#ifdef KNN_DEBUG
    // Debug of points:
    debug("debug points\n");
    for (int i=0; i<n; ++i){
        debug("for point %d: ", i);
        for (int j=0; j<dim; ++j){
            debug("%.lf ", points[i*dim + j]);
        }
        debug("\n");
    }
#endif

    // Reference result on CPU
    SizeT* ref_knns = NULL;
    SizeT* h_knns = (SizeT*)malloc(sizeof(SizeT) * n * k);

    if (!quick) {
      // Init datastructures for reference result on GPU
      ref_knns = (SizeT*)malloc(sizeof(SizeT) * n * k);

      // If not in `quick` mode, compute CPU reference implementation
      util::PrintMsg("__________________________", !quiet);
      util::PrintMsg("______ CPU Reference _____", !quiet);

      float elapsed = app::knn::CPU_Reference<VertexT, SizeT, ValueT>(
              points, n, dim, k, ref_knns, quiet);

      util::PrintMsg("--------------------------\n Elapsed: " + 
              std::to_string(elapsed), !quiet);
      util::PrintMsg("__________________________", !quiet);
      parameters.Set("cpu-elapsed", elapsed);
    }

    std::vector<std::string> switches{"advance-mode"};

    GUARD_CU((app::Switch_Parameters(parameters, graph, switches,
        [n, dim, k, h_knns, points, ref_knns]
        (util::Parameters& parameters, GraphT& graph) {
            return app::knn::RunTests(parameters, points, graph, n, dim, k, 
                    h_knns, ref_knns, util::DEVICE);
        })));

    if (!quick) {
      delete[] ref_knns;
    }

    return retval;
  }
};

int main(int argc, char** argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test knn");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::knn::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B | app::SIZET_U64B |
                           app::VALUET_F32B | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
