// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sample_test.cu
 *
 * @brief Simple test driver program
 */

#include <stdio.h>
#include <string>
#include <iostream>

// utilities for correctness checking
#include <gunrock/util/test_utils.cuh>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// primitive-specific headers include
#include <gunrock/app/template/sample_enactor.cuh>
#include <gunrock/app/template/sample_problem.cuh>
#include <gunrock/app/template/sample_functor.cuh>

// gunrock abstraction graph operators
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sample;

// ----------------------------------------------------------------------------
// Housekeeping Routines
// ----------------------------------------------------------------------------
void Usage() {
    printf(
        " sample_test <graph type> <file name> [--undirected] [--quick]\n"
        "   [--device=<device_index>]\n"
        " [--quiet] [--json] [--jsonfile=<name>] [--jsondir=<dir>]"
        " Graph types and arguments:\n"
        "   market <file>\n"
        "     Reads a Matrix-Market coordinate-formatted graph,\n"
        "     edges from STDIN (or from the optionally-specified file)\n"
        "   --device=<device_index> Set GPU device to run. [Default: 0]\n"
        "   --undirected            Convert the graph to undirected\n"
        "   --quick                 Skip the CPU validation [Default: false]\n"
        "   --v                     Print verbose per iteration debug info\n");
        " --quiet                  No output (unless --json is specified).\n"
        " --json                   Output JSON-format statistics to stdout.\n"
        " --jsonfile=<name>        Output JSON-format statistics to file <name>\n"
        " --jsondir=<dir>          Output JSON-format statistics to <dir>/name,\n"
}

/**
 * @brief Displays primitive results.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph Reference to the CSR graph.
 */
template<typename VertexId, typename SizeT, typename Value>
void DisplaySolution(const Csr<VertexId, Value, SizeT> &graph) {
    printf("==> display solution: (currently missing)\n");
    // TODO(developer): code to print out results
}

// ----------------------------------------------------------------------------
// Testing Routines
// ----------------------------------------------------------------------------

/**
 * @brief A simple CPU-based reference implementation.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph Reference to the CSR graph we process on.
 */
template<typename VertexId, typename SizeT, typename Value>
void SimpleReference(const Csr<VertexId, Value, SizeT> &graph) {
    // initialization

    // perform calculation

    CpuTimer cpu_timer;
    cpu_timer.Start();

    // TODO(developer): CPU validation code here

    cpu_timer.Stop();

    float cpu_elapsed = cpu_timer.ElapsedMillis();
    printf("CPU reference finished in %lf ms.\n\n", cpu_elapsed);
}

/**
 * @brief Sample test entry
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 *
 * @param[in] parameter Test parameter settings.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool DEBUG,
    bool SIZE_CHECK >
void RunTest(Info<VertexId, Value, SizeT> *info) {
    typedef SampleProblem < VertexId, SizeT, Value,
        true,   // MARK_PREDECESSORS
        false,  // ENABLE_IDEMPOTENCE
        false > Problem;

    Csr<VertexId, Value, SizeT>* csr = Info->csr_ptr;

    ContextPtr* context             = (ContextPtr*)info -> context;
    std::string partition_method    = info->info["partition_method"].get_str();
    int         max_grid_size       = info->info["max_grid_size"].get_int();
    int         num_gpus            = info->info["num_gpus"].get_int();
    int         iterations          = 1; //force to 1 for now.
    bool        quick_mode          = info->info["quick_mode"].get_bool();
    bool        quiet_mode          = info->info["quiet_mode"].get_bool();
    bool        stream_from_host    = info->info["stream_from_host"].get_bool();
    double      max_queue_sizing    = info->info["max_queue_sizing"].get_real();

    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();


    // allocate host-side array (for both reference and GPU-computed results)
    VertexId *r_labels = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);
    VertexId *h_labels = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);

    SampleEnactor <
        Problem,
        false,  // INSTRUMENT
        false,  // DEBUG
        true >  // SIZE_CHECK
        enactor(gpu_idx);  // allocate primitive enactor map

    Problem *problem = new Problem;  // allocate primitive problem on GPU
    util::GRError(
        problem->Init(stream_from_host, *csr, num_gpus),
        "Problem Initialization Failed", __FILE__, __LINE__);

    //
    // perform calculation
    //

    GpuTimer gpu_timer;

    float elapsed = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        util::GRError(
            problem->Reset(enactor.GetFrontierType(),
                           max_queue_sizing),
            "Problem Data Reset Failed", __FILE__, __LINE__);
        gpu_timer.Start();
        util::GRError(
            enactor.template Enact<Problem>(*context, problem, max_grid_size),
            "Problem Enact Failed", __FILE__, __LINE__);
        gpu_timer.Stop();
        elapsed += gpu_timer.ElapsedMillis();
    }

    elapsed /= iterations;

    // extract results
    util::GRError(
        problem->Extract(h_labels),
        "Problem Data Extraction Failed", __FILE__, __LINE__);

    // compute reference CPU validation solution
    if (!quick_mode) {
        if (!quiet_mode) printf("==> computing reference value ... (currently missing)\n");
        SimpleReference<VertexId, SizeT, Value>(graph);
        if (!quiet_mode) printf("==> validation: (currently missing)\n");
    }

    if (!quiet_mode) DisplaySolution<VertexId, SizeT, Value>(graph);  // display solution

    


    // clean up
    if (stats)    { delete stats;   }
    if (problem)  { delete problem; }
    if (r_labels) { free(r_labels); }
    if (h_labels) { free(h_labels); }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam DEBUG
 *
 * @param[in] parameter Pointer to test parameter settings
 */
template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          DEBUG >
void RunTests_size_check(Test_Parameter *parameter) {
    if (parameter->size_check)
        RunTest <VertexId, Value, SizeT, DEBUG,  true>(parameter);
    else
        RunTest <VertexId, Value, SizeT, DEBUG, false>(parameter);
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] parameter Pointer to test parameter settings
 */
template <
    typename    VertexId,
    typename    Value,
    typename    SizeT >
void RunTests_debug(Test_Parameter *parameter) {
    if (parameter->debug)
        RunTests_size_check <VertexId, Value, SizeT,  true>(parameter);
    else
        RunTests_size_check <VertexId, Value, SizeT, false>(parameter);
}

/**
 * @brief Sample test entry
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph    Pointer to the CSR graph we process on.
 * @param[in] args     Reference to the command line arguments.
 * @param[in] num_gpus Number of GPUs.
 * @param[in] context  CudaContext pointer for moderngpu APIs.
 * @param[in] gpu_idx  GPU inddex to run algorithm.
 * @param[in] streams  CUDA streams.
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void RunTest(
    Csr<VertexId, Value, SizeT>* graph,
    CommandLineArgs&             args,
    int                          num_gpus,
    ContextPtr*                  context,
    int*                         gpu_idx,
    cudaStream_t*                streams = NULL) {
    Test_Parameter *parameter = new Test_Parameter;

    parameter -> Init(args);
    parameter -> graph    = graph;
    parameter -> num_gpus = num_gpus;
    parameter -> context  = context;
    parameter -> gpu_idx  = gpu_idx;
    parameter -> streams  = streams;

    RunTests_debug<VertexId, Value, SizeT>(parameter);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    CommandLineArgs args(argc, argv);

    if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
        Usage();
        return 1;
    }

    int device = 0;
    args.GetCmdLineArgument("device", device);
    ContextPtr context = mgpu::CreateCudaDevice(device);

    // parse graph-construction parameters
    bool g_undirected = args.CheckCmdLineFlag("undirected");

    std::string graph_type = argv[1];
    int flags = args.ParsedArgc();
    int graph_args = argc - flags - 1;
    if (graph_args < 1) {
        Usage();
        return 1;
    }

    typedef int VertexId;  // Use as the vertex identifier
    typedef int SizeT;     // Use as the graph size type
    typedef int Value;     // Use as the value type

    if (graph_type == "market") {
        // matrix-market coordinate-formatted graph
        Csr<VertexId, Value, SizeT> csr(false);
        char *name = (graph_args == 2) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<false>(
                    name, csr, g_undirected, false) != 0) {
            return 1;
        }

        csr.DisplayGraph();    // display graph adjacent list
        csr.PrintHistogram();  // display graph histogram
        RunTest<VertexId, Value, SizeT>(&csr, args, 1, &context, &device);

    } else {
        fprintf(stderr, "Unspecified graph type\n");
        return 1;
    }
    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
