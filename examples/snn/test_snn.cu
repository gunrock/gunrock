// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_snn.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <gunrock/app/knn/knn_app.cu>
#include <gunrock/app/knn/knn_problem.cuh>
#include <gunrock/app/knn/knn_enactor.cuh>
#include <gunrock/app/knn/knn_test.cuh>
#include <gunrock/app/snn/snn_app.cu>
#include <gunrock/graphio/labels.cuh>
#include <gunrock/app/test_base.cuh>
#include <gunrock/app/problem_base.cuh>

// JSON includes
#include <gunrock/util/info_rapidjson.cuh>

//#define SNN_DEBUG
#ifdef SNN_DEBUG
    #define debug(a...) fprintf(stderr, a)
#else
    #define debug(a...)
#endif

using namespace gunrock;

namespace APP_NAMESPACE = app::snn;

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
  cudaError_t operator()(util::Parameters& parameters, VertexT v, SizeT s,
                         ValueT val) {
    // CLI parameters
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    // Get n dimension tuplets
    std::string labels_file = parameters.Get<std::string>("labels-file");
    util::PrintMsg("Points File Input: " + labels_file, !quiet);
    std::ifstream lfile(labels_file.c_str());
    if (labels_file == "" || !lfile.is_open()){
        util::PrintMsg("file cannot be open\n", !quiet);
        return (cudaError_t)1; 
    }

    cudaError_t retval = cudaSuccess;
    typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR>
        GraphT;
    GraphT graph;

    auto target = util::DEVICE;

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
    SizeT num_points = parameters.Get<SizeT>("n");
    // Get dimensional of space
    SizeT dim = parameters.Get<SizeT>("dim");
    // Get number of nearest neighbors, default k = 10
    SizeT k = parameters.Get<int>("k");
    if (k >= num_points)
      return util::GRError("K must be < N", __FILE__, __LINE__);

    // Get number of neighbors two close points should share
    SizeT eps = parameters.Get<SizeT>("eps");
    // Get the min density
    SizeT min_pts = parameters.Get<SizeT>("min-pts");
    if (min_pts > k)
      return util::GRError("Min-Pts must be < K", __FILE__, __LINE__);

#ifdef SNN_DEBUG
    // Debug of points:
    debug("debug points\n");
    for (SizeT i=0; i<num_points; ++i){
        debug("for point %d: ", i);
        for (SizeT j=0; j<dim; ++j){
            if (typeid(ValueT) == typeid(double))
                debug("%lf ", points[i*dim + j]);
            else 
                debug("%d ", points[i*dim + j]);
        }
        debug("\n");
    }
#endif

    util::PrintMsg("num_points = " + std::to_string(num_points) +
            ", k = " + std::to_string(k) +
            ", eps = " + std::to_string(eps) +
            ", min-pts = " + std::to_string(min_pts), !quiet);

    typedef app::knn::Problem<GraphT> ProblemKNN;
    typedef app::knn::Enactor<ProblemKNN> EnactorKNN;
    ProblemKNN knn_problem(parameters);
    EnactorKNN knn_enactor;
    GUARD_CU(knn_problem.Init(graph, util::DEVICE));
    GUARD_CU(knn_enactor.Init(knn_problem, util::DEVICE));
    GUARD_CU(knn_problem.Reset(points.GetPointer(util::HOST), target));
    GUARD_CU(knn_enactor.Reset(num_points, k, target));
   
    // Computing k Nearest Neighbors
    cpu_timer.Start();
    GUARD_CU(knn_enactor.Enact());
    cpu_timer.Stop();

    util::PrintMsg("KNN Elapsed: " 
              + std::to_string(cpu_timer.ElapsedMillis()), !quiet);
    // util::PrintMsg("__________________________", !quiet);
    // parameters.Set("knn-elapsed", cpu_timer.ElapsedMillis());

    // Extract kNN
    SizeT* h_knns = (SizeT*) malloc(sizeof(SizeT)*num_points*k);
    GUARD_CU(knn_problem.Extract(h_knns));
    
#ifdef SNN_DEBUG
    for (SizeT x = 0; x < num_points; ++x){
        debug("knn[%d]: ", x);
        for (int i = 0; i < k; ++i){
            if (typeid(ValueT) == typeid(double))
                debug("%lf ", h_knns[x * k + i]);
            else
                debug("%d ", h_knns[x * k + i]);
        }
        debug("\n");
    }
#endif

    // Reference result on CPU
    SizeT* ref_cluster = NULL;
    SizeT* ref_core_point_counter = NULL;
    SizeT* ref_cluster_counter = NULL;

    // Result on GPU
    SizeT* h_cluster = (SizeT*)malloc(sizeof(SizeT) * num_points);
    SizeT* h_core_point_counter = (SizeT*)malloc(sizeof(SizeT));
    SizeT* h_cluster_counter = (SizeT*)malloc(sizeof(SizeT));

    if (!quick) {
      // Init datastructures for reference result on GPU
      ref_cluster = (SizeT*)malloc(sizeof(SizeT) * num_points);
      for (auto i = 0; i < num_points; ++i) ref_cluster[i] = i;
      ref_core_point_counter = (SizeT*)malloc(sizeof(SizeT));
      ref_cluster_counter = (SizeT*)malloc(sizeof(SizeT));

      // If not in `quick` mode, compute CPU reference implementation
      util::PrintMsg("__________________________", !quiet);
      util::PrintMsg("______ CPU Reference _____", !quiet);

      float elapsed = app::snn::CPU_Reference(graph.csr(), num_points, k, 
              eps, min_pts, h_knns, ref_cluster, ref_core_point_counter,
              ref_cluster_counter, !quiet);

      util::PrintMsg("--------------------------\n Elapsed: " 
              + std::to_string(elapsed), !quiet);
      util::PrintMsg("__________________________", !quiet);
      parameters.Set("cpu-elapsed", elapsed);
    }

    std::vector<std::string> switches{"advance-mode"};

    GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
        [num_points, k, eps, min_pts, h_knns, h_cluster, h_core_point_counter,
         h_cluster_counter, ref_core_point_counter, ref_cluster_counter,
         ref_cluster]
         (util::Parameters& parameters, GraphT& graph) {
          return app::snn::RunTests(parameters, graph, num_points, k, eps, 
                  min_pts, 
                  h_knns, h_cluster, ref_cluster, h_core_point_counter,
                  ref_core_point_counter, h_cluster_counter,
                  ref_cluster_counter, util::DEVICE);
        }));

    if (!quick) {
      delete[] h_knns;
      delete[] ref_cluster;
      delete[] ref_core_point_counter;
      delete[] ref_cluster_counter;
    }

    delete[] h_cluster;
    return retval;
  }
};

int main(int argc, char** argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test knn");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::snn::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B | app::SIZET_U64B |
                           app::VALUET_S64B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
