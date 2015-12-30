// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_base.cuh
 *
 * @brief Base Graph Problem Enactor
 */

#pragma once
#include <time.h>

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/problem_base.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

/*
 * @brief Accumulate number function.
 *
 * @tparam SizeT1
 * @tparam SizeT2
 *
 * @param[in] num
 * @param[in] sum
 */
template <typename SizeT1, typename SizeT2>
__global__ void Accumulate_Num (
    SizeT1 *num,
    SizeT2 *sum)
{
    sum[0]+=num[0];
}

/**
 * @brief Structure for auxiliary variables used in enactor.
 */
struct EnactorStats
{
    long long                        iteration           ;
    unsigned long long               total_lifetimes     ;
    unsigned long long               total_runtimes      ;
    util::Array1D<int, long long>    edges_queued        ;
    util::Array1D<int, long long>    nodes_queued        ;
    unsigned int                     advance_grid_size   ;
    unsigned int                     filter_grid_size    ;
    util::KernelRuntimeStatsLifetime advance_kernel_stats;
    util::KernelRuntimeStatsLifetime filter_kernel_stats ;
    util::Array1D<int, unsigned int> node_locks          ;
    util::Array1D<int, unsigned int> node_locks_out      ;
    cudaError_t                      retval              ;
    clock_t                          start_time          ;

    /*
     * @brief Default EnactorStats constructor
     */
    EnactorStats()
    {
        iteration       = 0;
        total_lifetimes = 0;
        total_runtimes  = 0;
        retval          = cudaSuccess;
        node_locks    .SetName("node_locks"    );
        node_locks_out.SetName("node_locks_out");
        edges_queued  .SetName("edges_queued");
        nodes_queued  .SetName("nodes_queued");
    }

    /*
     * @brief Accumulate edge function.
     *
     * @tparam SizeT2
     *
     * @param[in] d_queue Pointer to the queue
     * @param[in] stream CUDA stream
     */
    template <typename SizeT2>
    void AccumulateEdges(SizeT2 *d_queued, cudaStream_t stream)
    {
        Accumulate_Num<<<1,1,0,stream>>> (
            d_queued, edges_queued.GetPointer(util::DEVICE));
    }

    /*
     * @brief Accumulate node function.
     *
     * @tparam SizeT2
     *
     * @param[in] d_queue Pointer to the queue
     * @param[in] stream CUDA stream
     */
    template <typename SizeT2>
    void AccumulateNodes(SizeT2 *d_queued, cudaStream_t stream)
    {
        Accumulate_Num<<<1,1,0,stream>>> (
            d_queued, nodes_queued.GetPointer(util::DEVICE));
    }

};

/**
 * @brief Info data structure contains test parameter and running statistics.
 * All test parameters and running statistics stored in json_spirit::mObject.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 */
// TODO: more robust empty info["value"] check.
template<typename VertexId, typename Value, typename SizeT>
struct Info
{
private:
    int            num_iters;  // Number of times invoke primitive test
    int            max_iters;  // Maximum number of super-steps allowed
    int            grid_size;  // Maximum grid size (0: up to the enactor)
    int            traversal;  // Load-balanced or Dynamic cooperative
    int             num_gpus;  // Number of GPUs used
    double          q_sizing;  // Maximum size scaling factor for work queues
    double         q_sizing1;  // Value of max_queue_sizing1
    double          i_sizing;  // Maximum size scaling factor for communication
    long long         source;  // Source vertex ID to start
    std::string ref_filename;  // CPU reference input file name
    std::string    file_stem;  // Market filename path stem
    std::string       ofname;  // Used for jsonfile command
    std::string          dir;  // Used for jsondir command
    std::string   par_method;  // Partition method
    double        par_factor;  // Partition factor
    int             par_seed;  // Partition seed
    int         delta_factor;  // Used in delta-stepping SSSP
    double             delta;  // Used in PageRank
    double             error;  // Used in PageRank
    double             alpha;  // Used in direction optimal BFS
    double              beta;  // Used in direction optimal BFS
    int            top_nodes;  // Used in Top-K

public:
    json_spirit::mObject info;  // test parameters and running statistics
    Csr<VertexId, Value, SizeT> *csr_ptr;  // pointer to CSR input graph
    Csr<VertexId, Value, SizeT> *csc_ptr;  // pointer to CSC input graph

    // TODO: following two already moved into Enactor in branch mgpu-cq
    void         *context;  // pointer to context array used by MordernGPU
    cudaStream_t *streams;  // pointer to array of GPU streams

    /**
     * @brief Info default constructor
     */
    Info()
    {
        // assign default values
        info["algorithm"]          = "";     // algorithm/primitive name
        info["average_duty"]       = 0.0f;   // average runtime duty
        info["command_line"]       = "";     // entire command line
        info["debug_mode"]         = false;  // verbose flag print debug info
        info["dataset"]            = "";     // dataset name used in test
        info["edges_visited"]      = 0;      // number of edges touched
        info["elapsed"]            = 0.0f;   // elapsed device running time
        info["preprocess_time"]    = 0.0f;   // elapsed preprocessing time
        info["postprocess_time"]   = 0.0f;   // postprocessing time
        info["total_time"]         = 0.0f;   // total run time of the program
        info["load_time"]          = 0.0f;   // data loading time
        info["write_time"]         = 0.0f;   // output writing time
        info["output_filename"]    = "";     // output filename
        info["engine"]             = "";     // engine name - Gunrock
        info["edge_value"]         = false;  // default don't load weights
        info["git_commit_sha1"]    = "";     // git commit sha1
        info["graph_type"]         = "";     // input graph type
        info["gunrock_version"]    = "";     // gunrock version number
        info["idempotent"]         = false;  // enable idempotent (BFS)
        info["instrument"]         = false;  // enable instrumentation
        info["num_iteration"]      = 1;      // number of runs
        info["json"]               = false;  // --json flag
        info["jsonfile"]           = "";     // --jsonfile
        info["jsondir"]            = "";     // --jsondir
        info["mark_predecessors"]  = false;  // mark predecessors (BFS, SSSP)
        info["max_grid_size"]      = 0;      // maximum grid size
        info["max_iteration"]      = 50;     // default maximum iteration
        info["max_in_sizing"]      = 1.0f;   // maximum in queue sizing factor
        info["max_queue_sizing"]   = 1.0f;   // maximum queue sizing factor
        info["max_queue_sizing1"]  = -1.0f;   // maximum queue sizing factor
        info["m_teps"]             = 0.0f;   // traversed edges per second
        info["num_gpus"]           = 1;      // number of GPU(s) used
        info["nodes_visited"]      = 0;      // number of nodes visited
        info["partition_method"]   = "random";  // default partition method
        info["partition_factor"]   = -1;     // partition factor
        info["partition_seed"]     = -1;     // partition seed
        info["quiet_mode"]         = false;  // don't print anything
        info["quick_mode"]         = false;  // skip CPU validation
        info["edges_redundance"]   = 0.0f;   // redundant edge work (BFS)
        info["nodes_redundance"]   = 0.0f;   // redundant node work
        info["ref_filename"]       = "";     // reference file input
        info["search_depth"]       = 0;      // search depth (iterations)
        info["size_check"]         = true;   // enable or disable size check
        info["source_type"]        = "";     // source type
        info["source_vertex"]      = 0;      // source (BFS, SSSP)
        info["stream_from_host"]   = false;  // stream from host to device
        info["traversal_mode"]     = -1;     // advance mode
        info["edges_queued"]       = 0;      // number of edges in queue
        info["nodes_queued"]       = 0;      // number of nodes in queue
        info["undirected"]         = true;   // default use undirected input
        info["delta_factor"]       = 16;     // default delta-factor for SSSP
        info["delta"]              = 0.85f;  // default delta for PageRank
        info["error"]              = 0.01f;  // default error for PageRank
        info["alpha"]              = 6.0f;   // default alpha for DOBFS
        info["beta"]               = 6.0f;   // default beta for DOBFS
        info["top_nodes"]          = 0;      // default number of nodes for top-k primitive
        info["normalized"]         = false;  // default normalized for PageRank
        // info["gpuinfo"]
        // info["device_list"]
        // info["sysinfo"]
        // info["time"]
        // info["userinfo"]
    }  // end Info()

    /**
     * @brief Initialization process for Info.
     *
     * @param[in] algorithm_name Algorithm name.
     * @param[in] args Command line arguments.
     */
    void InitBase(std::string algorithm_name, util::CommandLineArgs &args)
    {
        // put basic information into info
        info["engine"] = "Gunrock";
        info["command_line"] = json_spirit::mValue(args.GetEntireCommandLine());
        util::Sysinfo sysinfo;  // get machine / OS / user / time info
        info["sysinfo"] = sysinfo.getSysinfo();
        util::Gpuinfo gpuinfo;
        info["gpuinfo"] = gpuinfo.getGpuinfo();
        util::Userinfo userinfo;
        info["userinfo"] = userinfo.getUserinfo();
        time_t now = time(NULL); info["time"] = ctime(&now);
        info["gunrock_version"] = XSTR(GUNROCKVERSION);
        info["git_commit_sha1"] = g_GIT_SHA1;
        info["graph_type"] = args.GetCmdLineArgvGraphType();

        // get configuration parameters from command line arguments
        info["algorithm"]  =  algorithm_name;  // set algorithm name
        info["instrument"] =  args.CheckCmdLineFlag("instrumented");
        info["size_check"] = !args.CheckCmdLineFlag("disable-size-check");
        info["debug_mode"] =  args.CheckCmdLineFlag("v");
        info["quick_mode"] =  args.CheckCmdLineFlag("quick");
        info["quiet_mode"] =  args.CheckCmdLineFlag("quiet");
        info["idempotent"] =  args.CheckCmdLineFlag("idempotence");       // BFS
        info["mark_predecessors"] =  args.CheckCmdLineFlag("mark-pred");  // BFS
        info["normalized"] =  args.CheckCmdLineFlag("normalized"); // PR

        info["json"] = args.CheckCmdLineFlag("json");
        if (args.CheckCmdLineFlag("jsonfile"))
        {
            args.GetCmdLineArgument("jsonfile", ofname);
            info["jsonfile"] = ofname;
        }
        if (args.CheckCmdLineFlag("jsondir"))
        {
            args.GetCmdLineArgument("jsondir", dir);
            info["jsondir"] = dir;
        }

        // determine which source to start search
        if (args.CheckCmdLineFlag("src"))
        {
            std::string source_type;
            args.GetCmdLineArgument("src", source_type);
            if (source_type.empty())
            {
                source = 0;
                info["source_type"] = "default";
            }
            else if (source_type.compare("randomize") == 0)
            {
                source = graphio::RandomNode(csr_ptr->nodes);
                if (!args.CheckCmdLineFlag("quiet"))
                {
                    printf("Using random source vertex: %lld\n", source);
                }
                info["source_type"] = "random";
            }
            else if (source_type.compare("largestdegree") == 0)
            {
                int maximum_degree;
                source = csr_ptr->GetNodeWithHighestDegree(maximum_degree);
                if (!args.CheckCmdLineFlag("quiet"))
                {
                    printf("Using highest degree (%d), vertex: %lld\n",
                           maximum_degree, source);
                }
                info["source_type"] = "largest-degree";
            }
            else
            {
                args.GetCmdLineArgument("src", source);
                info["source_type"] = "user-defined";
            }
            info["source_vertex"] = (int64_t)source;
            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("Source vertex: %lld\n", source);
            }
        }
        if (args.CheckCmdLineFlag("grid-size"))
        {
            args.GetCmdLineArgument("grid-size", grid_size);
            info["max_grid_size"] = grid_size;
        }
        if (args.CheckCmdLineFlag("iteration-num"))
        {
            args.GetCmdLineArgument("iteration-num", num_iters);
            info["num_iteration"] = num_iters;
        }
        if (args.CheckCmdLineFlag("max-iter"))
        {
            args.GetCmdLineArgument("max-iter", max_iters);
            info["max_iteration"] = max_iters;
        }
        if (args.CheckCmdLineFlag("queue-sizing"))
        {
            args.GetCmdLineArgument("queue-sizing", q_sizing);
            info["max_queue_sizing"] = q_sizing;
        }
        if (args.CheckCmdLineFlag("queue-sizing1"))
        {
            args.GetCmdLineArgument("queue-sizing1", q_sizing1);
            info["max_queue_sizing1"] = q_sizing1;
        }
        if (args.CheckCmdLineFlag("in-sizing"))
        {
            args.GetCmdLineArgument("in-sizing", i_sizing);
            info["max_in_sizing"] = i_sizing;
        }
        if (args.CheckCmdLineFlag("partition-method"))
        {
            args.GetCmdLineArgument("partition-method", par_method);
            info["partition_method"] = par_method;
        }
        if (args.CheckCmdLineFlag("partition-factor"))
        {
            args.GetCmdLineArgument("partition-factor", par_factor);
            info["partition_factor"] = par_factor;
        }
        if (args.CheckCmdLineFlag("partition-seed"))
        {
            args.GetCmdLineArgument("partition-seed", par_seed);
            info["partition_seed"] = par_seed;
        }
        if (args.CheckCmdLineFlag("traversal-mode"))
        {
            args.GetCmdLineArgument("traversal-mode", traversal);
            info["traversal_mode"] = traversal;
        }
        if (traversal == -1)
        {
            traversal = csr_ptr->GetAverageDegree() > 5 ? 0 : 1;
            info["traversal_mode"] = traversal;
        }
        if (args.CheckCmdLineFlag("ref_filename"))
        {
            args.GetCmdLineArgument("ref_filename", ref_filename);
            info["ref_filename"] = ref_filename;
        }
        if (args.CheckCmdLineFlag("delta_factor"))  // SSSP
        {
            args.GetCmdLineArgument("delta_factor", delta_factor);
            info["delta_factor"] = delta_factor;
        }
        if (args.CheckCmdLineFlag("delta"))
        {
            args.GetCmdLineArgument("delta", delta);
            info["delta"] = delta;
        }
        if (args.CheckCmdLineFlag("error"))
        {
            args.GetCmdLineArgument("error", error);
            info["error"] = error;
        }
        if (args.CheckCmdLineFlag("alpha"))
        {
            args.GetCmdLineArgument("alpha", alpha);
            info["alpha"] = alpha;
        }
        if (args.CheckCmdLineFlag("beta"))
        {
            args.GetCmdLineArgument("beta", beta);
            info["beta"] = beta;
        }
        if (args.CheckCmdLineFlag("top_nodes"))
        {
            args.GetCmdLineArgument("top_nodes", top_nodes);
            info["top_nodes"] = top_nodes;
        }
        if (args.CheckCmdLineFlag("output_filename"))
        {
            std::string output_filename = "";
            args.GetCmdLineArgument("output_filename", output_filename);
            info["output_filename"] = output_filename;
        }
        
        // parse device count and device list
        info["device_list"] = GetDeviceList(args);

        ///////////////////////////////////////////////////////////////////////
        // initialize CUDA streams and context for MordernGPU API.
        // TODO: streams and context initialization can be removed after merge
        // with `mgpu-cq` branch. YC already moved them into Enactor code.
        std::vector<int> temp_devices;
        if (args.CheckCmdLineFlag("device"))  // parse device list
        {
            args.GetCmdLineArguments<int>("device", temp_devices);
            num_gpus = temp_devices.size();
        }
        else  // use single device with index 0
        {
            num_gpus = 1;
            temp_devices.push_back(0);
        }

        cudaStream_t*     streams_ = new cudaStream_t[num_gpus * num_gpus * 2];
        mgpu::ContextPtr* context_ = new mgpu::ContextPtr[num_gpus * num_gpus];

        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            util::SetDevice(temp_devices[gpu]);
            for (int i = 0; i < num_gpus * 2; i++)
            {
                int _i = gpu * num_gpus * 2 + i;
                util::GRError(cudaStreamCreate(&streams_[_i]),
                              "cudaStreamCreate failed.", __FILE__, __LINE__);
                if (i < num_gpus)
                {
                    context_[gpu * num_gpus + i] =
                        mgpu::CreateCudaDeviceAttachStream(
                            temp_devices[gpu], streams_[_i]);
                }
            }
        }

        context = (mgpu::ContextPtr*)context_;
        streams = (cudaStream_t*)streams_;
        ///////////////////////////////////////////////////////////////////////
    }

    /**
     * @brief Initialization process for Info.
     *
     * @param[in] algorithm_name Algorithm name.
     * @param[in] args Command line arguments.
     * @param[in] csr_ref Reference to the CSR structure.
     */
    void Init(
        std::string algorithm_name,
        util::CommandLineArgs &args,
        Csr<VertexId, Value, SizeT> &csr_ref)
    {
        // load or generate input graph
        if (info["edge_value"].get_bool())
        {
            LoadGraph<true, false>(args, csr_ref);  // load graph with weighs
        }
        else
        {
            LoadGraph<false, false>(args, csr_ref);  // load without weights
        }
        csr_ptr = &csr_ref;  // set graph pointer
        InitBase(algorithm_name, args);
    }

    /**
     * @brief Initialization process for Info.
     *
     * @param[in] algorithm_name Algorithm name.
     * @param[in] args Command line arguments.
     * @param[in] csr_ref Reference to the CSR structure.
     * @param[in] csc_ref Reference to the CSC structure.
     */
    void Init(
        std::string algorithm_name,
        util::CommandLineArgs &args,
        Csr<VertexId, Value, SizeT> &csr_ref,
        Csr<VertexId, Value, SizeT> &csc_ref)
    {
         // load or generate input graph
        if (info["edge_value"].get_bool())
        {
            if (info["undirected"].get_bool())
            {
                LoadGraph<true, false>(args, csr_ref);  // with weigh values
                LoadGraph<true, false>(args, csc_ref);  // same as CSR
            }
            else
            {
                LoadGraph<true, false>(args, csr_ref);  // load CSR input
                LoadGraph<true,  true>(args, csc_ref);  // load CSC input
            }
        }
        else  // does not need weight values
        {
            if (info["undirected"].get_bool())
            {
                LoadGraph<false, false>(args, csr_ref);  // without weights
                LoadGraph<false, false>(args, csc_ref);  // without weights
            }
            else
            {
                LoadGraph<false, false>(args, csr_ref);  // without weights
                LoadGraph<false,  true>(args, csc_ref);  // without weights
            }
        }
        csr_ptr = &csr_ref;  // set CSR pointer
        csc_ptr = &csc_ref;  // set CSC pointer
        InitBase(algorithm_name, args);
    }

    /**
     * @brief Display JSON mObject info. Should be called after ComputeStats.
     */
    void CollectInfo()
    {
        // output JSON if user specified
        if (info["json"].get_bool())
        {
            PrintJson();
        }
        if (ofname != "")
        {
            JsonFile();
        }
        if (dir != "")
        {
            JsonDir();
        }
    }

    /**
     * @brief Utility function to parse device list.
     *
     * @param[in] args Command line arguments.
     *
     * \return json_spirit::mArray object contain devices used.
     */
    json_spirit::mArray GetDeviceList(util::CommandLineArgs &args)
    {
        json_spirit::mArray device_list;      // return mArray
        std::vector<int> devices;             // temp storage
        if (args.CheckCmdLineFlag("device"))  // parse command
        {
            args.GetCmdLineArguments<int>("device", devices);
            num_gpus = devices.size();
            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("Using %d GPU(s): [", num_gpus);
                for (int i = 0; i < num_gpus; ++i)
                {
                    printf(" %d", devices[i]);
                }
                printf(" ].\n");
            }
            info["num_gpus"] = num_gpus;  // update number of devices
            for (int i = 0; i < num_gpus; i++)
            {
                device_list.push_back(devices[i]);
            }
        }
        else  // use single device with index 0
        {
            num_gpus = 1;
            device_list.push_back(0);
            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("Using 1 GPU: [ 0 ].\n");
            }
        }
        return device_list;
    }

    /**
     * @brief Writes the JSON structure to STDOUT (command line --json).
     */
    void PrintJson()
    {
        json_spirit::write_stream(
            json_spirit::mValue(info), std::cout,
            json_spirit::pretty_print);
        printf("\n");
    }

    /*
     * @brief Writes the JSON structure to filename (command line --jsonfile).
     */
    void JsonFile()
    {
        std::ofstream of(ofname.data());
        json_spirit::write_stream(
            json_spirit::mValue(info), of,
            json_spirit::pretty_print);
    }

    /**
     * @brief Writes the JSON structure to an automatically-uniquely-named
     * file in the dir directory (command line --jsondir).
     */
    void JsonDir()
    {
        std::string filename =
            dir + "/" + info["algorithm"].get_str() + "_" +
            ((file_stem != "") ? (file_stem + "_") : "") +
            info["time"].get_str() + ".json";
        // now filter out bad chars (the list in bad_chars)
        char bad_chars[] = ":\n";
        for (unsigned int i = 0; i < strlen(bad_chars); ++i)
        {
            filename.erase(
                std::remove(filename.begin(), filename.end(), bad_chars[i]),
                filename.end());
        }
        std::ofstream of(filename.data());
        json_spirit::write_stream(
            json_spirit::mValue(info), of,
            json_spirit::pretty_print);
    }

    /**
     * @brief Utility function to load input graph.
     *
     * @tparam EDGE_VALUE
     * @tparam INVERSE_GRAPH
     *
     * @param[in] args Command line arguments.
     * @param[in] csr_ref Reference to the CSR graph.
     *
     * \return int whether successfully loaded the graph (0 success, 1 error).
     */
    template<bool EDGE_VALUE, bool INVERSE_GRAPH>
    int LoadGraph(
        util::CommandLineArgs &args,
        Csr<VertexId, Value, SizeT> &csr_ref)
    {
        std::string graph_type = args.GetCmdLineArgvGraphType();
        if (graph_type == "market")  // Matrix-market graph
        {
            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("Loading Matrix-market coordinate-formatted graph ...\n");
            }
            char *market_filename = args.GetCmdLineArgvDataset();
            if (market_filename == NULL)
            {
                printf("YZH Log.");
                fprintf(stderr, "Input graph does not exist.\n");
                return 1;
            }
            boost::filesystem::path market_filename_path(market_filename);
            file_stem = market_filename_path.stem().string();
            info["dataset"] = file_stem;
            if (graphio::BuildMarketGraph<EDGE_VALUE>(
                        market_filename,
                        csr_ref,
                        info["undirected"].get_bool(),
                        INVERSE_GRAPH,
                        args.CheckCmdLineFlag("quiet")) != 0)
            {
                return 1;
            }
        }
        else if (graph_type == "rmat")  // R-MAT graph
        {
            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("Generating R-MAT graph ...\n");
            }
            // parse R-MAT parameters
            SizeT rmat_nodes = 1 << 10;
            SizeT rmat_edges = 1 << 10;
            SizeT rmat_scale = 10;
            SizeT rmat_edgefactor = 48;
            double rmat_a = 0.57;
            double rmat_b = 0.19;
            double rmat_c = 0.19;
            double rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
            int rmat_seed = -1;

            args.GetCmdLineArgument("rmat_scale", rmat_scale);
            rmat_nodes = 1 << rmat_scale;
            args.GetCmdLineArgument("rmat_nodes", rmat_nodes);
            args.GetCmdLineArgument("rmat_edgefactor", rmat_edgefactor);
            rmat_edges = rmat_nodes * rmat_edgefactor;
            args.GetCmdLineArgument("rmat_edges", rmat_edges);
            args.GetCmdLineArgument("rmat_a", rmat_a);
            args.GetCmdLineArgument("rmat_b", rmat_b);
            args.GetCmdLineArgument("rmat_c", rmat_c);
            rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
            args.GetCmdLineArgument("rmat_d", rmat_d);
            args.GetCmdLineArgument("rmat_seed", rmat_seed);

            // put everything into mObject info
            info["rmat_a"] = rmat_a;
            info["rmat_b"] = rmat_b;
            info["rmat_c"] = rmat_c;
            info["rmat_d"] = rmat_d;
            info["rmat_seed"] = rmat_seed;
            info["rmat_scale"] = rmat_scale;
            info["rmat_nodes"] = rmat_nodes;
            info["rmat_edges"] = rmat_edges;
            info["rmat_edgefactor"] = rmat_edgefactor;

            util::CpuTimer cpu_timer;
            cpu_timer.Start();

            // generate R-MAT graph
            if (graphio::BuildRmatGraph<EDGE_VALUE>(
                        rmat_nodes,
                        rmat_edges,
                        csr_ref,
                        info["undirected"].get_bool(),
                        rmat_a,
                        rmat_b,
                        rmat_c,
                        rmat_d,
                        1,
                        1,
                        rmat_seed,
                        args.CheckCmdLineFlag("quiet")) != 0)
            {
                return 1;
            }

            cpu_timer.Stop();
            float elapsed = cpu_timer.ElapsedMillis();

            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("R-MAT graph generated in %.3f ms, "
                       "a = %.3f, b = %.3f, c = %.3f, d = %.3f\n",
                       elapsed, rmat_a, rmat_b, rmat_c, rmat_d);
            }
        }
        else if (graph_type == "rgg")
        {
            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("Generating RGG (Random Geometry Graph) ...\n");
            }

            SizeT rgg_nodes = 1 << 10;
            SizeT rgg_scale = 10;
            double rgg_thfactor  = 0.55;
            double rgg_threshold =
                rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
            double rgg_vmultipiler = 1;
            int rgg_seed = -1;

            args.GetCmdLineArgument("rgg_scale", rgg_scale);
            rgg_nodes = 1 << rgg_scale;
            args.GetCmdLineArgument("rgg_nodes", rgg_nodes);
            args.GetCmdLineArgument("rgg_thfactor", rgg_thfactor);
            rgg_threshold = rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
            args.GetCmdLineArgument("rgg_threshold", rgg_threshold);
            args.GetCmdLineArgument("rgg_vmultipiler", rgg_vmultipiler);
            args.GetCmdLineArgument("rgg_seed", rgg_seed);

            // put everything into mObject info
            info["rgg_seed"]        = rgg_seed;
            info["rgg_scale"]       = rgg_scale;
            info["rgg_nodes"]       = rgg_nodes;
            info["rgg_thfactor"]    = rgg_thfactor;
            info["rgg_threshold"]   = rgg_threshold;
            info["rgg_vmultipiler"] = rgg_vmultipiler;

            util::CpuTimer cpu_timer;
            cpu_timer.Start();

            // generate random geometry graph
            if (graphio::BuildRggGraph<EDGE_VALUE>(
                        rgg_nodes,
                        csr_ref,
                        rgg_threshold,
                        info["undirected"].get_bool(),
                        rgg_vmultipiler,
                        1,
                        rgg_seed,
                        args.CheckCmdLineFlag("quiet")) != 0)
            {
                return 1;
            }

            cpu_timer.Stop();
            float elapsed = cpu_timer.ElapsedMillis();
            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("RGG generated in %.3f ms, "
                       "threshold = %.3lf, vmultipiler = %.3lf\n",
                       elapsed, rgg_threshold, rgg_vmultipiler);
            }
        }
        else
        {
            fprintf(stderr, "Unspecified graph type.\n");
            return 1;
        }

        if (!args.CheckCmdLineFlag("quiet"))
        {
            csr_ref.GetAverageDegree();
            csr_ref.PrintHistogram();
            if (info["algorithm"].get_str().compare("SSSP") == 0)
            {
                csr_ref.GetAverageEdgeValue();
                int max_degree;
                csr_ref.GetNodeWithHighestDegree(max_degree);
                printf("Maximum degree: %d\n", max_degree);
            }
        }
        return 0;
    }

    /**
     * @brief Compute statistics common to all primitives.
     *
     * @param[in] enactor_stats
     * @param[in] elapsed
     * @param[in] labels
     * @param[in] get_traversal_stats
     */
    void ComputeCommonStats(
        EnactorStats *enactor_stats,
        float elapsed,
        const VertexId *labels = NULL,
        bool get_traversal_stats = false)
    {
        double total_lifetimes = 0;
        double total_runtimes = 0;

        // traversal stats
        int64_t edges_queued = 0;
        int64_t nodes_queued = 0;
        int64_t search_depth = 0;
        int64_t nodes_visited = 0;
        int64_t edges_visited = 0;
        float   m_teps = 0.0f;
        double  edges_redundance = 0.0f;
        double  nodes_redundance = 0.0f;

        json_spirit::mArray device_list = info["device_list"].get_array();

        for (int gpu = 0; gpu < num_gpus; ++gpu)
        {
            int my_gpu_idx = device_list[gpu].get_int();
            if (num_gpus != 1)
            {
                if (util::SetDevice(my_gpu_idx)) return;
            }
            cudaThreadSynchronize();

            for (int peer = 0; peer < num_gpus; ++peer)
            {
                EnactorStats *estats = enactor_stats + gpu * num_gpus + peer;
                if (get_traversal_stats)
                {
                    edges_queued += estats->edges_queued[0];
                    estats->edges_queued.Move(util::DEVICE, util::HOST);
                    edges_queued += estats->edges_queued[0];

                    nodes_queued += estats->nodes_queued[0];
                    estats->nodes_queued.Move(util::DEVICE, util::HOST);
                    nodes_queued += estats->nodes_queued[0];

                    if (estats->iteration > search_depth)
                    {
                        search_depth = estats->iteration;
                    }
                }
                total_lifetimes += estats->total_lifetimes;
                total_runtimes  += estats->total_runtimes;
            }
        }

        double avg_duty = (total_lifetimes > 0) ?
            double(total_runtimes) / total_lifetimes * 100.0 : 0.0f;

        info["elapsed"] = elapsed;
        info["average_duty"] = avg_duty;
        info["search_depth"] = search_depth;

        if (get_traversal_stats)
        {
            info["edges_queued"] = edges_queued;
            info["nodes_queued"] = nodes_queued;
        }

        // TODO: compute traversal stats
        if (get_traversal_stats)
        {
            if (labels != NULL)
            for (VertexId i = 0; i < csr_ptr->nodes; ++i)
            {
                if (labels[i] < util::MaxValue<VertexId>() && labels[i] != -1)
                {
                    ++nodes_visited;
                    edges_visited +=
                        csr_ptr->row_offsets[i + 1] - csr_ptr->row_offsets[i];
                }
            }
            if (info["algorithm"].get_str().compare("BC") == 0)
            {
                // for betweenness should count the backward phase too.
                edges_visited = 2 * edges_queued;
            } else if (info["algorithm"].get_str().compare("PageRank") == 0)
            {
                edges_visited = csr_ptr -> edges;
            }

            if (nodes_queued > nodes_visited)
            {  // measure duplicate nodes put through queue
                nodes_redundance =
                    ((double)nodes_queued - nodes_visited) / nodes_visited;
            }

            if (edges_queued > edges_visited)
            {
                // measure duplicate edges put through queue
                edges_redundance =
                    ((double)edges_queued - edges_visited) / edges_visited;
            }
            nodes_redundance *= 100;
            edges_redundance *= 100;

            m_teps = (double)edges_visited / (elapsed * 1000.0);

            info["nodes_visited"] = nodes_visited;
            info["edges_visited"] = edges_visited;
            info["nodes_redundance"] = nodes_redundance;
            info["edges_redundance"] = edges_redundance;
            info["m_teps"] = m_teps;
        }
    }

    /**
     * @brief Compute statistics common to all traversal primitives.
     *
     * @param[in] enactor_stats
     * @param[in] elapsed
     * @param[in] labels
     */
    void ComputeTraversalStats(
        EnactorStats *enactor_stats,
        float elapsed,
        const VertexId *labels = NULL)
    {
        ComputeCommonStats(
            enactor_stats,
            elapsed,
            labels,
            true);
    }

    /**
     * @brief Display running statistics.
     *
     * @param[in] verbose Whether or not to print extra information.
     */
    void DisplayStats(bool verbose = true)
    {
        double elapsed        = info["elapsed"      ].get_real();
        int64_t nodes_visited = info["nodes_visited"].get_int();
        int64_t edges_visited = info["edges_visited"].get_int();
        double  m_teps        = info["m_teps"       ].get_real();
        int64_t search_depth  = info["search_depth" ].get_int();
        double  avg_duty      = info["average_duty" ].get_real();
        int64_t edges_queued  = info["edges_queued" ].get_int();
        int64_t nodes_queued  = info["nodes_queued" ].get_int();
        double  nodes_redundance = info["nodes_redundance"].get_real();
        double  edges_redundance = info["edges_redundance"].get_real();
        double  load_time        = info["load_time"       ].get_real();
        double  preprocess_time  = info["preprocess_time" ].get_real();
        double  postprocess_time = info["postprocess_time"].get_real();
        double  write_time       = info["write_time"      ].get_real();
        double  total_time       = info["total_time"      ].get_real();

        printf("\n [%s] finished.", info["algorithm"].get_str().c_str());
        printf("\n elapsed: %.4f ms\n iterations: %lld", elapsed, (long long)search_depth);

        if (verbose)
        {
            if (nodes_visited != 0 && nodes_visited < 5)
            {
                printf("Fewer than 5 vertices visited.\n");
            }
            else
            {
                if (m_teps > 0.01)
                {
                    printf("\n rate: %.4f MiEdges/s", m_teps);
                }
                if (avg_duty > 0.01)
                {
                    printf("\n average CTA duty: %.2f%%", avg_duty);
                }
                if (nodes_visited != 0 && edges_visited != 0)
                {
                    printf("\n src: %lld\n nodes_visited: %lld\n edges_visited: %lld",
                        source, (long long)nodes_visited, (long long)edges_visited);
                }
                if (nodes_queued > 0)
                {
                    printf("\n nodes queued: %lld", (long long)nodes_queued);
                }
                if (edges_queued > 0)
                {
                    printf("\n edges queued: %lld", (long long)edges_queued);
                }
                if (nodes_redundance > 0.01)
                {
                    printf("\n nodes redundance: %.2f%%", nodes_redundance);
                }
                if (edges_redundance > 0.01)
                {
                    printf("\n edges redundance: %.2f%%", edges_redundance);
                }
                printf("\n load time: %.4f ms", load_time);
                printf("\n preprocess time: %.4f ms", preprocess_time);
                printf("\n postprocess time: %.4f ms", postprocess_time);
                if (info["output_filename"].get_str() != "")
                    printf("\n write time: %.4f ms", write_time);
                printf("\n total time: %.4f ms", total_time);
           }
        }
        printf("\n");
    }
};

/**
 * @brief Structure for auxiliary variables used in frontier operations.
 */
template <typename SizeT>
struct FrontierAttribute
{
    SizeT        queue_length ;
    util::Array1D<SizeT,SizeT>
                 output_length;
    unsigned int queue_index  ;
    SizeT        queue_offset ;
    int          selector     ;
    bool         queue_reset  ;
    int          current_label;
    bool         has_incoming ;
    gunrock::oprtr::advance::TYPE
                 advance_type ;

    /*
     * @brief Default FrontierAttribute constructor
     */
    FrontierAttribute()
    {
        queue_length  = 0;
        queue_index   = 0;
        queue_offset  = 0;
        selector      = 0;
        queue_reset   = false;
        has_incoming  = false;
        output_length.SetName("output_length");
    }
};

/*
 * @brief Thread slice data structure
 */
class ThreadSlice
{
public:
    int           thread_num ;
    int           init_size  ;
    CUTThread     thread_Id  ;
    int           stats      ;
    void         *problem    ;
    void         *enactor    ;
    ContextPtr   *context    ;
    util::cpu_mt::CPUBarrier
                 *cpu_barrier;

    /*
     * @brief Default ThreadSlice constructor
     */
    ThreadSlice()
    {
        problem     = NULL;
        enactor     = NULL;
        context     = NULL;
        thread_num  = 0;
        init_size   = 0;
        stats       = -2;
        cpu_barrier = NULL;
    }

    /*
     * @brief Default ThreadSlice destructor
     */
    virtual ~ThreadSlice()
    {
        problem     = NULL;
        enactor     = NULL;
        context     = NULL;
        cpu_barrier = NULL;
    }
};

/*
 * @brief
 *
 * @tparam SizeT
 * @tparam DataSlice
 *
 * @param[in] enactor_stats Pointer to the enactor stats.
 * @param[in] frontier_attribute Pointer to the frontier attribute.
 * @param[in] data_slice Pointer to the data slice we process on.
 * @param[in] num_gpus Number of GPUs used for testing.
 */
template <typename SizeT, typename DataSlice>
bool All_Done(EnactorStats                    *enactor_stats,
              FrontierAttribute<SizeT>        *frontier_attribute,
              util::Array1D<SizeT, DataSlice> *data_slice,
              int                              num_gpus)
{
    for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
    if (enactor_stats[gpu].retval!=cudaSuccess)
    {
        printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu%num_gpus, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
        return true;
    }

    for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
    if (frontier_attribute[gpu].queue_length!=0 || frontier_attribute[gpu].has_incoming)
    {
        //printf("frontier_attribute[%d].queue_length = %d\n",gpu,frontier_attribute[gpu].queue_length);
        return false;
    }

    for (int gpu=0;gpu<num_gpus;gpu++)
    for (int peer=1;peer<num_gpus;peer++)
    for (int i=0;i<2;i++)
    if (data_slice[gpu]->in_length[i][peer]!=0)
    {
        //printf("data_slice[%d]->in_length[%d][%d] = %d\n", gpu, i, peer, data_slice[gpu]->in_length[i][peer]);
        return false;
    }

    for (int gpu=0;gpu<num_gpus;gpu++)
    for (int peer=1;peer<num_gpus;peer++)
    if (data_slice[gpu]->out_length[peer]!=0)
    {
        //printf("data_slice[%d]->out_length[%d] = %d\n", gpu, peer, data_slice[gpu]->out_length[peer]);
        return false;
    }

    return true;
}

/*
 * @brief Copy predecessor function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements Number of elements in the array.
 * @param[in] keys Pointer to the key array.
 * @param[in] in_preds Pointer to the input predecessor array.
 * @param[out] out_preds Pointer to the output predecessor array.
 */
template <typename VertexId, typename SizeT>
__global__ void Copy_Preds (
    const SizeT     num_elements,
    const VertexId* keys,
    const VertexId* in_preds,
          VertexId* out_preds)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x*blockDim.x+threadIdx.x;
    VertexId t;

    while (x<num_elements)
    {
        t = keys[x];
        out_preds[t] = in_preds[t];
        x+= STRIDE;
    }
}

/*
 * @brief Update predecessor function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements Number of elements in the array.
 * @param[in] nodes Number of nodes in graph.
 * @param[in] keys Pointer to the key array.
 * @param[in] org_vertexs
 * @param[in] in_preds Pointer to the input predecessor array.
 * @param[out] out_preds Pointer to the output predecessor array.
 */
template <typename VertexId, typename SizeT>
__global__ void Update_Preds (
    const SizeT     num_elements,
    const SizeT     nodes,
    const VertexId* keys,
    const VertexId* org_vertexs,
    const VertexId* in_preds,
          VertexId* out_preds)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x*blockDim.x + threadIdx.x;
    VertexId t, p;

    while (x<num_elements)
    {
        t = keys[x];
        p = in_preds[t];
        if (p<nodes) out_preds[t] = org_vertexs[p];
        x+= STRIDE;
    }
}

/*
 * @brief Assign marker function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements Number of elements in the array.
 * @param[in] num_gpus Number of GPUs used for testing.
 * @param[in] keys_in Pointer to the key array.
 * @param[in] partition_table Pointer to the partition table.
 * @param[out] marker
 */
template <typename VertexId, typename SizeT>
__global__ void Assign_Marker(
    const SizeT            num_elements,
    const int              num_gpus,
    const VertexId* const  keys_in,
    const int*      const  partition_table,
          SizeT**          marker)
{
    VertexId key;
    int gpu;
    extern __shared__ SizeT* s_marker[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < num_gpus)
        s_marker[threadIdx.x]=marker[threadIdx.x];
    __syncthreads();

    while (x < num_elements)
    {
        key = keys_in[x];
        gpu = partition_table[key];
        for (int i=0;i<num_gpus;i++)
            s_marker[i][x]=(i==gpu)?1:0;
        x+=STRIDE;
    }
}

/*
 * @brief Assign marker backward function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements Number of elements in the array.
 * @param[in] num_gpus Number of GPUs used for testing.
 * @param[in] keys_in Pointer to the key array.
 * @param[in] offsets Pointer to
 * @param[in] partition_table Pointer to the partition table.
 * @param[out] marker
 */
template <typename VertexId, typename SizeT>
__global__ void Assign_Marker_Backward(
    const SizeT            num_elements,
    const int              num_gpus,
    const VertexId* const  keys_in,
    const SizeT*    const  offsets,
    const int*      const  partition_table,
          SizeT**          marker)
{
    VertexId key;
    extern __shared__ SizeT* s_marker[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < num_gpus)
        s_marker[threadIdx.x]=marker[threadIdx.x];
    __syncthreads();

    while (x < num_elements)
    {
        key = keys_in[x];
        for (int gpu=0;gpu<num_gpus;gpu++)
            s_marker[gpu][x]=0;
        if (key!=-1) for (SizeT i=offsets[key];i<offsets[key+1];i++)
            s_marker[partition_table[i]][x]=1;
        x+=STRIDE;
    }
}

/*
 * @brief Make output function.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam num_vertex_associates
 * @tparam num_value__associates
 *
 * @param[in] num_elements Number of elements.
 * @param[in] num_gpus Number of GPUs used.
 * @param[in] keys_in Pointer to the key array.
 * @param[in] partition_table
 * @param[in] convertion_table
 * @param[in] array_size
 * @param[in] array
 */
template <typename VertexId, typename SizeT, typename Value,
          SizeT num_vertex_associates, SizeT num_value__associates>
__global__ void Make_Out(
   const  SizeT             num_elements,
   const  int               num_gpus,
   const  VertexId*   const keys_in,
   const  int*        const partition_table,
   const  VertexId*   const convertion_table,
   const  size_t            array_size,
          char*             array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    size_t     offset                  = 0;
    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
    offset+=sizeof(SizeT*   )*num_gpus;
    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus;
    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_vertex_associates;
    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_value__associates;
    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus*num_vertex_associates;
    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_gpus*num_value__associates;
    SizeT*     s_offset                = (SizeT*    )&(s_array[offset]);
    SizeT x= threadIdx.x;

    while (x<array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x= blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        VertexId key    = keys_in [x];
        int      target = partition_table[key];
        SizeT    pos    = s_marker[target][x]-1 + s_offset[target];

        if (target==0)
        {
            s_keys_outs[0][pos]=key;
        } else {
            s_keys_outs[target][pos]=convertion_table[key];
            #pragma unroll
            for (int i=0;i<num_vertex_associates;i++)
                s_vertex_associate_outss[target*num_vertex_associates+i][pos]
                    =s_vertex_associate_orgs[i][key];
            #pragma unroll
            for (int i=0;i<num_value__associates;i++)
                s_value__associate_outss[target*num_value__associates+i][pos]
                    =s_value__associate_orgs[i][key];
        }
        x+=STRIDE;
    }
}

/*
 * @brief Make output backward function.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam num_vertex_associates
 * @tparam num_value__associates
 *
 * @param[in] num_elements Number of elements.
 * @param[in] num_gpus Number of GPUs used.
 * @param[in] keys_in Pointer to the key array.
 * @param[in] partition_table
 * @param[in] convertion_table
 * @param[in] array_size
 * @param[in] array
 */
template <typename VertexId, typename SizeT, typename Value,
          SizeT num_vertex_associates, SizeT num_value__associates>
__global__ void Make_Out_Backward(
   const  SizeT             num_elements,
   const  int               num_gpus,
   const  VertexId*   const keys_in,
   const  SizeT*      const offsets,
   const  int*        const partition_table,
   const  VertexId*   const convertion_table,
   const  size_t            array_size,
          char*             array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    size_t     offset                  = 0;
    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
    offset+=sizeof(SizeT*   )*num_gpus;
    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus;
    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_vertex_associates;
    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_value__associates;
    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus*num_vertex_associates;
    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_gpus*num_value__associates;
    SizeT*     s_offset                = (SizeT*    )&(s_array[offset]);
    SizeT x= threadIdx.x;

    while (x<array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x= blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        VertexId key    = keys_in [x];
        if (key <0) {x+=STRIDE; continue;}
        for (SizeT j=offsets[key];j<offsets[key+1];j++)
        {
            int      target = partition_table[j];
            SizeT    pos    = s_marker[target][x]-1 + s_offset[target];

            if (target==0)
            {
                s_keys_outs[0][pos]=key;
            } else {
                s_keys_outs[target][pos]=convertion_table[j];
                #pragma unroll
                for (int i=0;i<num_vertex_associates;i++)
                    s_vertex_associate_outss[target*num_vertex_associates+i][pos]
                        =s_vertex_associate_orgs[i][key];
                #pragma unroll
                for (int i=0;i<num_value__associates;i++)
                    s_value__associate_outss[target*num_value__associates+i][pos]
                        =s_value__associate_orgs[i][key];
            }
        }
        x+=STRIDE;
    }
}

/*
 * @brief Mark_Queue function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements
 * @param[in] keys
 * @param[in] market
 */
template <typename VertexId, typename SizeT>
__global__ void Mark_Queue (
    const SizeT     num_elements,
    const VertexId* keys,
          unsigned int* marker)
{
    VertexId x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
    if (x< num_elements) marker[keys[x]]=1;
}

/*
 * @brief Check size function.
 *
 * @tparam SIZE_CHECK
 * @tparam SizeT
 * @tparam Type
 *
 * @param[in] name
 * @param[in] target_length
 * @param[in] array
 * @param[in] oversized
 * @param[in] thread_num
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] keep_content
 *
 * \return cudaError_t object Indicates the success of all CUDA calls.
 */
template <
    bool     SIZE_CHECK,
    typename SizeT,
    typename Type>
cudaError_t Check_Size(
    const char *name,
    SizeT       target_length,
    util::Array1D<SizeT, Type>
               *array,
    bool       &oversized,
    int         thread_num = -1,
    int         iteration  = -1,
    int         peer_      = -1,
    bool        keep_content = false)
{
    cudaError_t retval = cudaSuccess;

    if (target_length > array->GetSize())
    {
        printf("%d\t %d\t %d\t %s \t oversize :\t %d ->\t %d\n",
        thread_num, iteration, peer_, name, array->GetSize(), target_length);
        oversized=true;
        if (SIZE_CHECK)
        {
            if (array->GetSize() != 0) retval = array->EnsureSize(target_length, keep_content);
            else retval = array->Allocate(target_length, util::DEVICE);
        } else {
            char temp_str[]=" oversize", str[256];
            memcpy(str, name, sizeof(char) * strlen(name));
            memcpy(str + strlen(name), temp_str, sizeof(char) * strlen(temp_str));
            str[strlen(name)+strlen(temp_str)]='0';
            retval = util::GRError(cudaErrorLaunchOutOfResources, str, __FILE__, __LINE__);
        }
    }
    return retval;
}

/*
 * @brief Check size function.
 *
 * @tparam SIZE_CHECK
 * @tparam SizeT
 * @tparam VertexId
 * @tparam Value
 * @tparam GraphSlice
 * @tparam DataSlice
 * @tparam num_vertex_associate
 * @tparam num_value__associate
 *
 * @param[in] gpu
 * @param[in] peer
 * @param[in] array
 * @param[in] queue_length
 * @param[in] enactor_stats
 * @param[in] data_slice_l
 * @param[in] data_slice_p
 * @param[in] graph_slice_l Graph slice local
 * @param[in] graph_slice_p
 * @param[in] stream CUDA stream.
 */
template <
    bool     SIZE_CHECK,
    typename SizeT,
    typename VertexId,
    typename Value,
    typename GraphSlice,
    typename DataSlice,
    SizeT    num_vertex_associate,
    SizeT    num_value__associate>
void PushNeighbor(
    int gpu,
    int peer,
    SizeT             queue_length,
    EnactorStats      *enactor_stats,
    DataSlice         *data_slice_l,
    DataSlice         *data_slice_p,
    GraphSlice        *graph_slice_l,
    GraphSlice        *graph_slice_p,
    cudaStream_t      stream)
{
    if (peer == gpu) return;
    int gpu_  = peer<gpu? gpu : gpu+1;
    int peer_ = peer<gpu? peer+1 : peer;
    int i, t  = enactor_stats->iteration%2;
    bool to_reallocate = false;
    bool over_sized    = false;

    data_slice_p->in_length[enactor_stats->iteration%2][gpu_]
                  = queue_length;
    if (queue_length == 0) return;

    if (data_slice_p -> keys_in[t][gpu_].GetSize() < queue_length) to_reallocate=true;
    else {
        for (i=0;i<num_vertex_associate;i++)
            if (data_slice_p->vertex_associate_in[t][gpu_][i].GetSize() < queue_length) {to_reallocate=true;break;}
        if (!to_reallocate)
            for (i=0;i<num_value__associate;i++)
                if (data_slice_p->value__associate_in[t][gpu_][i].GetSize() < queue_length) {to_reallocate=true;break;}
    }

    if (to_reallocate)
    {
        if (SIZE_CHECK) util::SetDevice(data_slice_p->gpu_idx);
        if (enactor_stats->retval = Check_Size<SIZE_CHECK, SizeT, VertexId>(
            "keys_in", queue_length, &data_slice_p->keys_in[t][gpu_], over_sized,
            gpu, enactor_stats->iteration, peer)) return;

        for (i=0;i<num_vertex_associate;i++)
        {
            if (enactor_stats->retval = Check_Size<SIZE_CHECK, SizeT, VertexId>(
                "vertex_associate_in", queue_length, &data_slice_p->vertex_associate_in[t][gpu_][i], over_sized,
                gpu, enactor_stats->iteration, peer)) return;
            data_slice_p->vertex_associate_ins[t][gpu_][i] = data_slice_p->vertex_associate_in[t][gpu_][i].GetPointer(util::DEVICE);
        }
        for (i=0;i<num_value__associate;i++)
        {
            if (enactor_stats->retval = Check_Size<SIZE_CHECK, SizeT, Value>(
                "value__associate_in", queue_length, &data_slice_p->value__associate_in[t][gpu_][i], over_sized,
                gpu, enactor_stats->iteration, peer)) return;
            data_slice_p->value__associate_ins[t][gpu_][i] = data_slice_p->value__associate_in[t][gpu_][i].GetPointer(util::DEVICE);
        }
        if (SIZE_CHECK)
        {
            if (enactor_stats->retval = data_slice_p->vertex_associate_ins[t][gpu_].Move(util::HOST, util::DEVICE)) return;
            if (enactor_stats->retval = data_slice_p->value__associate_ins[t][gpu_].Move(util::HOST, util::DEVICE)) return;
            util::SetDevice(data_slice_l->gpu_idx);
        }
    }

    if (enactor_stats-> retval = util::GRError(cudaMemcpyAsync(
        data_slice_p -> keys_in[t][gpu_].GetPointer(util::DEVICE),
        data_slice_l -> keys_out[peer_].GetPointer(util::DEVICE),
        sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
        "cudaMemcpyPeer keys failed", __FILE__, __LINE__)) return;

    for (int i=0;i<num_vertex_associate;i++)
    {
        if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
            data_slice_p->vertex_associate_ins[t][gpu_][i],
            data_slice_l->vertex_associate_outs[peer_][i],
            sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
            "cudaMemcpyPeer vertex_associate_out failed", __FILE__, __LINE__)) return;
    }

    for (int i=0;i<num_value__associate;i++)
    {
        if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
            data_slice_p->value__associate_ins[t][gpu_][i],
            data_slice_l->value__associate_outs[peer_][i],
            sizeof(Value) * queue_length, cudaMemcpyDefault, stream),
                "cudaMemcpyPeer value__associate_out failed", __FILE__, __LINE__)) return;
    }
}

/*
 * @brief Show debug information function.
 *
 * @tparam Problem
 *
 * @param[in] thread_num
 * @param[in] peer_
 * @param[in] frontier_attribute
 * @param[in] enactor_stats
 * @param[in] data_slice
 * @param[in] graph_slice
 * @param[in] work_progress
 * @param[in] check_name
 * @param[in] stream CUDA stream.
 */
template <typename Problem>
void ShowDebugInfo(
    int           thread_num,
    int           peer_,
    FrontierAttribute<typename Problem::SizeT>
                 *frontier_attribute,
    EnactorStats *enactor_stats,
    typename Problem::DataSlice
                 *data_slice,
    GraphSlice<typename Problem::SizeT, typename Problem::VertexId, typename Problem::Value>
                 *graph_slice,
    util::CtaWorkProgressLifetime
                 *work_progress,
    std::string   check_name = "",
    cudaStream_t  stream = 0)
{
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value;
    SizeT queue_length;

    //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
    //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
    //if (frontier_attribute->queue_reset)
        queue_length = frontier_attribute->queue_length;
    //else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
    //util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
    printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, peer_, data_slice->stages[peer_], check_name.c_str(), queue_length);fflush(stdout);
    //printf("%d \t %d\t \t peer_ = %d, selector = %d, length = %d, p = %p\n",thread_num, enactor_stats->iteration, peer_, frontier_attribute->selector,queue_length,graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE));fflush(stdout);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>((check_name+" keys").c_str(), data_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE), queue_length, thread_num, enactor_stats->iteration,peer_, stream);
    //if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
    //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("degrees", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
    //if (BFSProblem::MARK_PREDECESSOR)
    //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred1", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
    //if (BFSProblem::ENABLE_IDEMPOTENCE)
    //    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask1", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
}

/*
 * @brief Set record function.
 *
 * @tparam DataSlice
 *
 * @param[in] data_slice
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] stage
 * @param[in] stream CUDA stream.
 */
template <typename DataSlice>
cudaError_t Set_Record(
    DataSlice *data_slice,
    int iteration,
    int peer_,
    int stage,
    cudaStream_t stream)
{
    cudaError_t retval = cudaEventRecord(data_slice->events[iteration%4][peer_][stage],stream);
    data_slice->events_set[iteration%4][peer_][stage]=true;
    return retval;
}

/*
 * @brief Set record function.
 *
 * @tparam DataSlice
 *
 * @param[in] data_slice
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] stage_to_check
 * @param[in] stage
 * @param[in] to_show
 */
template <typename DataSlice>
cudaError_t Check_Record(
    DataSlice *data_slice,
    int iteration,
    int peer_,
    int stage_to_check,
    int &stage,
    bool &to_show)
{
    cudaError_t retval = cudaSuccess;
    to_show = true;
    if (!data_slice->events_set[iteration%4][peer_][stage_to_check])
    {
        to_show = false;
        stage--;
    } else {
        retval = cudaEventQuery(data_slice->events[iteration%4][peer_][stage_to_check]);
        if (retval == cudaErrorNotReady)
        {
            to_show=false;
            stage--;
            retval = cudaSuccess;
        } else if (retval == cudaSuccess)
        {
            data_slice->events_set[iteration%4][peer_][stage_to_check]=false;
        }
    }
    return retval;
}

/*
 * @brief Iteration loop.
 *
 * @tparam NUM_VERTEX_ASSOCIATES
 * @tparam NUM_VALUE__ASSOCIATES
 * @tparam Enactor
 * @tparam Functor
 * @tparam Iteration
 *
 * @param[in] thread_data
 */
template <
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES,
    typename Enactor,
    typename Functor,
    typename Iteration>
void Iteration_Loop(
    ThreadSlice *thread_data)
{
    typedef typename Enactor::Problem     Problem   ;
    typedef typename Problem::SizeT       SizeT     ;
    typedef typename Problem::VertexId    VertexId  ;
    typedef typename Problem::Value       Value     ;
    typedef typename Problem::DataSlice   DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value>  GraphSlice;

    Problem      *problem              =  (Problem*) thread_data->problem;
    Enactor      *enactor              =  (Enactor*) thread_data->enactor;
    int           num_gpus             =   problem     -> num_gpus;
    int           thread_num           =   thread_data -> thread_num;
    DataSlice    *data_slice           =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    util::Array1D<SizeT, DataSlice>
                 *s_data_slice         =   problem     -> data_slices;
    GraphSlice   *graph_slice          =   problem     -> graph_slices       [thread_num] ;
    GraphSlice   **s_graph_slice       =   problem     -> graph_slices;
    FrontierAttribute<SizeT>
                 *frontier_attribute   = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    FrontierAttribute<SizeT>
                 *s_frontier_attribute = &(enactor     -> frontier_attribute [0         ]);
    EnactorStats *enactor_stats        = &(enactor     -> enactor_stats      [thread_num * num_gpus]);
    EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats      [0         ]);
    util::CtaWorkProgressLifetime
                 *work_progress        = &(enactor     -> work_progress      [thread_num * num_gpus]);
    ContextPtr   *context              =   thread_data -> context;
    int          *stages               =   data_slice  -> stages .GetPointer(util::HOST);
    bool         *to_show              =   data_slice  -> to_show.GetPointer(util::HOST);
    cudaStream_t *streams              =   data_slice  -> streams.GetPointer(util::HOST);
    SizeT         Total_Length         =   0;
    cudaError_t   tretval              =   cudaSuccess;
    int           grid_size            =   0;
    std::string   mssg                 =   "";
    int           pre_stage            =   0;
    size_t        offset               =   0;
    int           iteration            =   0;
    int           selector             =   0;
    util::DoubleBuffer<SizeT, VertexId, Value>
                 *frontier_queue_      =   NULL;
    FrontierAttribute<SizeT>
                 *frontier_attribute_  =   NULL;
    EnactorStats *enactor_stats_       =   NULL;
    util::CtaWorkProgressLifetime
                 *work_progress_       =   NULL;
    util::Array1D<SizeT, SizeT>
                 *scanned_edges_       =   NULL;
    int           peer, peer_, peer__, gpu_, i, iteration_, wait_count;
    bool          over_sized;

    if (Enactor::DEBUG)
    {
        printf("Iteration entered\n");fflush(stdout);
    }
    while (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
    {
        Total_Length             = 0;
        data_slice->wait_counter = 0;
        tretval                  = cudaSuccess;
        if (num_gpus>1 && enactor_stats[0].iteration>0)
        {
            frontier_attribute[0].queue_reset  = true;
            frontier_attribute[0].queue_offset = 0;
            for (i=1; i<num_gpus; i++)
            {
                frontier_attribute[i].selector     = frontier_attribute[0].selector;
                frontier_attribute[i].advance_type = frontier_attribute[0].advance_type;
                frontier_attribute[i].queue_offset = 0;
                frontier_attribute[i].queue_reset  = true;
                frontier_attribute[i].queue_index  = frontier_attribute[0].queue_index;
                frontier_attribute[i].current_label= frontier_attribute[0].current_label;
                enactor_stats     [i].iteration    = enactor_stats     [0].iteration;
            }
        } else {
            frontier_attribute[0].queue_offset = 0;
            frontier_attribute[0].queue_reset  = true;
        }
        for (peer=0; peer<num_gpus; peer++)
        {
            stages [peer         ] = 0   ;
            stages [peer+num_gpus] = 0   ;
            to_show[peer         ] = true;
            to_show[peer+num_gpus] = true;
            for (i=0; i<data_slice->num_stages; i++)
                data_slice->events_set[enactor_stats[0].iteration%4][peer][i]=false;
        }

        while (data_slice->wait_counter < num_gpus*2
           && (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)))
        {
            for (peer__=0; peer__<num_gpus*2; peer__++)
            {
                peer_               = (peer__%num_gpus);
                peer                = peer_<= thread_num? peer_-1   : peer_       ;
                gpu_                = peer <  thread_num? thread_num: thread_num+1;
                iteration           = enactor_stats[peer_].iteration;
                iteration_          = iteration%4;
                pre_stage           = stages[peer__];
                selector            = frontier_attribute[peer_].selector;
                frontier_queue_     = &(data_slice->frontier_queues[peer_]);
                scanned_edges_      = &(data_slice->scanned_edges  [peer_]);
                frontier_attribute_ = &(frontier_attribute         [peer_]);
                enactor_stats_      = &(enactor_stats              [peer_]);
                work_progress_      = &(work_progress              [peer_]);

                if (Enactor::DEBUG && to_show[peer__])
                {
                    mssg=" ";mssg[0]='0'+data_slice->wait_counter;
                    ShowDebugInfo<Problem>(
                        thread_num,
                        peer__,
                        frontier_attribute_,
                        enactor_stats_,
                        data_slice,
                        graph_slice,
                        work_progress_,
                        mssg,
                        streams[peer__]);
                }
                to_show[peer__]=true;

                switch (stages[peer__])
                {
                case 0: // Assign marker & Scan
                    if (peer_==0) {
                        if (peer__==num_gpus || frontier_attribute_->queue_length==0)
                        {
                            stages[peer__]=3;
                        } else if (!Iteration::HAS_SUBQ) {
                            stages[peer__]=2;
                        }
                        break;
                    } else if ((iteration==0 || data_slice->out_length[peer_]==0) && peer__>num_gpus) {
                        Set_Record(data_slice, iteration, peer_, 0, streams[peer__]);
                        stages[peer__]=3;
                        break;
                    }

                    if (peer__<num_gpus)
                    { //wait and expand incoming
                        if (!(s_data_slice[peer]->events_set[iteration_][gpu_][0]))
                        {   to_show[peer__]=false;stages[peer__]--;break;}

                        s_data_slice[peer]->events_set[iteration_][gpu_][0]=false;
                        frontier_attribute_->queue_length = data_slice->in_length[iteration%2][peer_];
                        data_slice->in_length[iteration%2][peer_]=0;
                        if (frontier_attribute_->queue_length ==0)
                        {   stages[peer__]=3;break;}

                        offset = 0;
                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                 data_slice -> vertex_associate_ins[iteration%2][peer_].GetPointer(util::HOST),
                                  sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES);
                        offset += sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES ;
                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                 data_slice -> value__associate_ins[iteration%2][peer_].GetPointer(util::HOST),
                                  sizeof(VertexId*) * NUM_VALUE__ASSOCIATES);
                        offset += sizeof(VertexId*) * NUM_VALUE__ASSOCIATES ;
                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                 data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
                                  sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
                        offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                 data_slice -> value__associate_orgs.GetPointer(util::HOST),
                                  sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
                        offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
                        data_slice->expand_incoming_array[peer_].Move(util::HOST, util::DEVICE, offset, 0, streams[peer_]);

                        grid_size = frontier_attribute_->queue_length/256+1;
                        if (grid_size>512) grid_size=512;
                        cudaStreamWaitEvent(streams[peer_],
                            s_data_slice[peer]->events[iteration_][gpu_][0], 0);
                        Iteration::template Expand_Incoming<NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            grid_size, 256,
                            offset,
                            streams[peer_],
                            frontier_attribute_->queue_length,
                            data_slice ->keys_in[iteration%2][peer_].GetPointer(util::DEVICE),
                            &frontier_queue_->keys[selector^1],
                            offset,
                            data_slice ->expand_incoming_array[peer_].GetPointer(util::DEVICE),
                            data_slice);
                        frontier_attribute_->selector^=1;
                        frontier_attribute_->queue_index++;
                        if (!Iteration::HAS_SUBQ) {
                            Set_Record(data_slice, iteration, peer_, 2, streams[peer__]);
                            stages[peer__]=2;
                        }
                    } else { //Push Neighbor
                        PushNeighbor <Enactor::SIZE_CHECK, SizeT, VertexId, Value, GraphSlice, DataSlice,
                                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            thread_num,
                            peer,
                            data_slice->out_length[peer_],
                            enactor_stats_,
                            s_data_slice  [thread_num].GetPointer(util::HOST),
                            s_data_slice  [peer]      .GetPointer(util::HOST),
                            s_graph_slice [thread_num],
                            s_graph_slice [peer],
                            streams       [peer__]);
                        Set_Record(data_slice, iteration, peer_, stages[peer__], streams[peer__]);
                        stages[peer__]=3;
                    }
                    break;

                case 1: //Comp Length
                    if (enactor_stats_->retval = Iteration::Compute_OutputLength(
                        frontier_attribute_,
                        graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
                        graph_slice    ->column_indices  .GetPointer(util::DEVICE),
                        graph_slice    ->column_offsets     .GetPointer(util::DEVICE),
                        graph_slice    ->row_indices  .GetPointer(util::DEVICE),
                        frontier_queue_->keys[selector]  .GetPointer(util::DEVICE),
                        scanned_edges_,
                        graph_slice    ->nodes,
                        graph_slice    ->edges,
                        context          [peer_][0],
                        streams          [peer_],
                        gunrock::oprtr::advance::V2V, true, false, false)) break;

                    if (!Enactor::SIZE_CHECK && 
                        (Iteration::AdvanceKernelPolicy::ADVANCE_MODE 
                            == oprtr::advance::TWC_FORWARD ||
                         Iteration::AdvanceKernelPolicy::ADVANCE_MODE 
                            == oprtr::advance::TWC_BACKWARD))
                    {}
                    else {
                        //printf("moving output_length\n");
                        frontier_attribute_ -> output_length.Move(
                            util::DEVICE, util::HOST,1,0,streams[peer_]);
                    }

                    if (Enactor::SIZE_CHECK)
                    {
                        Set_Record(data_slice, iteration, peer_, stages[peer_], streams[peer_]);
                    }
                    break;

                case 2: //SubQueue Core
                    if (Enactor::SIZE_CHECK)
                    {
                        if (enactor_stats_ -> retval = Check_Record (
                            data_slice, iteration, peer_,
                            stages[peer_]-1, stages[peer_], to_show[peer_])) break;
                        if (to_show[peer_]==false) break;
                        if (Iteration::AdvanceKernelPolicy::ADVANCE_MODE 
                            == oprtr::advance::TWC_FORWARD ||
                            Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            == oprtr::advance::TWC_BACKWARD)
                        {
                            frontier_attribute_->output_length[0] *= 1.1;
                        }
                        //printf("iteration = %lld, request_size = %d\n",
                        //    enactor_stats_ -> iteration, frontier_attribute_->output_length[0]);
                        Iteration::Check_Queue_Size(
                            thread_num,
                            peer_,
                            frontier_attribute_->output_length[0] + 2,
                            frontier_queue_,
                            frontier_attribute_,
                            enactor_stats_,
                            graph_slice);
                    }

                    Iteration::SubQueue_Core(
                        thread_num,
                        peer_,
                        frontier_queue_,
                        scanned_edges_,
                        frontier_attribute_,
                        enactor_stats_,
                        data_slice,
                        s_data_slice[thread_num].GetPointer(util::DEVICE),
                        graph_slice,
                        &(work_progress[peer_]),
                        context[peer_],
                        streams[peer_]);

                    if (enactor_stats_->retval = work_progress[peer_].GetQueueLength(
                        frontier_attribute_->queue_index,
                        frontier_attribute_->queue_length,
                        false,
                        streams[peer_],
                        true)) break;
                    if (num_gpus>1)
                        Set_Record(data_slice, iteration, peer_, stages[peer_], streams[peer_]);
                    break;

                case 3: //Copy
                    if (num_gpus <=1)
                    {
                        if (enactor_stats_-> retval = util::GRError(cudaStreamSynchronize(streams[peer_]), "cudaStreamSynchronize failed",__FILE__, __LINE__)) break;
                        Total_Length = frontier_attribute_->queue_length;
                        to_show[peer_]=false;break;
                    }
                    if (Iteration::HAS_SUBQ || peer_!=0) {
                        if (enactor_stats_-> retval = Check_Record(
                            data_slice, iteration, peer_,
                            stages[peer_]-1, stages[peer_], to_show[peer_])) break;
                        if (to_show[peer_] == false) break;
                    }

                    if (!Enactor::SIZE_CHECK /*&& Enactor::DEBUG*/)
                    {
                        if (Iteration::HAS_SUBQ)
                        {
                            if (enactor_stats_->retval =
                                Check_Size<false, SizeT, VertexId> ("queue3", frontier_attribute_->output_length[0]+2, &frontier_queue_->keys  [selector^1], over_sized, thread_num, iteration, peer_, false)) break;
                        }
                        if (frontier_attribute_->queue_length ==0) break;

                        if (enactor_stats_->retval =
                            Check_Size<false, SizeT, VertexId> ("total_queue", Total_Length + frontier_attribute_->queue_length, &data_slice->frontier_queues[num_gpus].keys[0], over_sized, thread_num, iteration, peer_, false)) break;

                        util::MemsetCopyVectorKernel<<<256,256, 0, streams[peer_]>>>(
                            data_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE) + Total_Length,
                            frontier_queue_->keys[selector].GetPointer(util::DEVICE),
                            frontier_attribute_->queue_length);
                        if (Problem::USE_DOUBLE_BUFFER)
                            util::MemsetCopyVectorKernel<<<256,256,0,streams[peer_]>>>(
                                data_slice->frontier_queues[num_gpus].values[0].GetPointer(util::DEVICE) + Total_Length,
                                frontier_queue_->values[selector].GetPointer(util::DEVICE),
                                frontier_attribute_->queue_length);
                    }

                    Total_Length += frontier_attribute_->queue_length;
                    break;

                case 4: //End
                    data_slice->wait_counter++;
                    to_show[peer__]=false;
                    break;
                default:
                    stages[peer__]--;
                    to_show[peer__]=false;
                }

                if (Enactor::DEBUG && !enactor_stats_->retval)
                {
                    mssg="stage 0 @ gpu 0, peer_ 0 failed";
                    mssg[6]=char(pre_stage+'0');
                    mssg[14]=char(thread_num+'0');
                    mssg[23]=char(peer__+'0');
                    enactor_stats_->retval = util::GRError(mssg, __FILE__, __LINE__);
                    if (enactor_stats_ -> retval) break;
                }
                stages[peer__]++;
                if (enactor_stats_->retval) break;
            }
        }

        if (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
        {
            for (peer_=0;peer_<num_gpus*2;peer_++)
                data_slice->wait_marker[peer_]=0;
            wait_count=0;
            while (wait_count<num_gpus*2-1 &&
                !Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
            {
                for (peer_=0;peer_<num_gpus*2;peer_++)
                {
                    if (peer_==num_gpus || data_slice->wait_marker[peer_]!=0)
                        continue;
                    tretval = cudaStreamQuery(streams[peer_]);
                    if (tretval == cudaSuccess)
                    {
                        data_slice->wait_marker[peer_]=1;
                        wait_count++;
                        continue;
                    } else if (tretval != cudaErrorNotReady)
                    {
                        enactor_stats[peer_%num_gpus].retval = tretval;
                        break;
                    }
                }
            }

            if (Enactor::DEBUG)
            {
                printf("%d\t %lld\t \t Subqueue finished. Total_Length= %d\n",
                    thread_num, enactor_stats[0].iteration, Total_Length);
                fflush(stdout);
            }

            grid_size = Total_Length/256+1;
            if (grid_size > 512) grid_size = 512;

            if (Enactor::SIZE_CHECK)
            {
                if (enactor_stats[0]. retval =
                    Check_Size<true, SizeT, VertexId> ("total_queue", Total_Length, &data_slice->frontier_queues[0].keys[frontier_attribute[0].selector], over_sized, thread_num, iteration, num_gpus, true)) break;
                if (Problem::USE_DOUBLE_BUFFER)
                    if (enactor_stats[0].retval =
                        Check_Size<true, SizeT, Value> ("total_queue", Total_Length, &data_slice->frontier_queues[0].values[frontier_attribute[0].selector], over_sized, thread_num, iteration, num_gpus, true)) break;

                offset=frontier_attribute[0].queue_length;
                for (peer_=1;peer_<num_gpus;peer_++)
                if (frontier_attribute[peer_].queue_length !=0) {
                    util::MemsetCopyVectorKernel<<<256,256, 0, streams[0]>>>(
                        data_slice->frontier_queues[0    ].keys[frontier_attribute[0    ].selector].GetPointer(util::DEVICE) + offset,
                        data_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                        frontier_attribute[peer_].queue_length);
                    if (Problem::USE_DOUBLE_BUFFER)
                        util::MemsetCopyVectorKernel<<<256,256,0,streams[0]>>>(
                            data_slice->frontier_queues[0       ].values[frontier_attribute[0    ].selector].GetPointer(util::DEVICE) + offset,
                            data_slice->frontier_queues[peer_   ].values[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                            frontier_attribute[peer_].queue_length);
                    offset+=frontier_attribute[peer_].queue_length;
                }
            }
            frontier_attribute[0].queue_length = Total_Length;
            if (!Enactor::SIZE_CHECK) frontier_attribute[0].selector = 0;
            frontier_queue_ = &(data_slice->frontier_queues[(Enactor::SIZE_CHECK || num_gpus == 1)?0:num_gpus]);
            if (Iteration::HAS_FULLQ)
            {
                peer_               = 0;
                frontier_queue_     = &(data_slice->frontier_queues[(Enactor::SIZE_CHECK || num_gpus==1)?0:num_gpus]);
                scanned_edges_      = &(data_slice->scanned_edges  [(Enactor::SIZE_CHECK || num_gpus==1)?0:num_gpus]);
                frontier_attribute_ = &(frontier_attribute[peer_]);
                enactor_stats_      = &(enactor_stats[peer_]);
                work_progress_      = &(work_progress[peer_]);
                iteration           = enactor_stats[peer_].iteration;
                frontier_attribute_->queue_offset = 0;
                frontier_attribute_->queue_reset  = true;
                if (!Enactor::SIZE_CHECK) frontier_attribute_->selector     = 0;

                Iteration::FullQueue_Gather(
                    thread_num,
                    peer_,
                    frontier_queue_,
                    scanned_edges_,
                    frontier_attribute_,
                    enactor_stats_,
                    data_slice,
                    s_data_slice[thread_num].GetPointer(util::DEVICE),
                    graph_slice,
                    work_progress_,
                    context[peer_],
                    streams[peer_]);
                selector            = frontier_attribute[peer_].selector;
                if (enactor_stats_->retval) break;

                if (frontier_attribute_->queue_length !=0)
                {
                    if (Enactor::DEBUG) {
                        mssg = "";
                        ShowDebugInfo<Problem>(
                            thread_num,
                            peer_,
                            frontier_attribute_,
                            enactor_stats_,
                            data_slice,
                            graph_slice,
                            work_progress_,
                            mssg,
                            streams[peer_]);
                    }

                    enactor_stats_->retval = Iteration::Compute_OutputLength(
                        frontier_attribute_,
                        graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
                        graph_slice    ->column_indices  .GetPointer(util::DEVICE),
                    graph_slice    ->column_offsets     .GetPointer(util::DEVICE),
                        graph_slice    ->row_indices  .GetPointer(util::DEVICE),
                        frontier_queue_->keys[selector].GetPointer(util::DEVICE),
                        scanned_edges_,
                        graph_slice    ->nodes,
                        graph_slice    ->edges,
                        context          [peer_][0],
                        streams          [peer_],
                        gunrock::oprtr::advance::V2V, true, false, false);
                    if (enactor_stats_->retval) break;

                    frontier_attribute_->output_length.Move(util::DEVICE, util::HOST, 1, 0, streams[peer_]);
                    if (Enactor::SIZE_CHECK)
                    {
                        tretval = cudaStreamSynchronize(streams[peer_]);
                        if (tretval != cudaSuccess) {enactor_stats_->retval=tretval;break;}

                        Iteration::Check_Queue_Size(
                            thread_num,
                            peer_,
                            frontier_attribute_->output_length[0] + 2,
                            frontier_queue_,
                            frontier_attribute_,
                            enactor_stats_,
                            graph_slice);

                    }

                    Iteration::FullQueue_Core(
                        thread_num,
                        peer_,
                        frontier_queue_,
                        scanned_edges_,
                        frontier_attribute_,
                        enactor_stats_,
                        data_slice,
                        s_data_slice[thread_num].GetPointer(util::DEVICE),
                        graph_slice,
                        work_progress_,
                        context[peer_],
                        streams[peer_]);
                    if (enactor_stats_->retval) break;
                    if (!Enactor::SIZE_CHECK)
                    {
                        if (enactor_stats_->retval =
                            Check_Size<false, SizeT, VertexId> ("queue3", frontier_attribute->output_length[0]+2, &frontier_queue_->keys[selector^1], over_sized, thread_num, iteration, peer_, false)) break;
                    }
                    selector = frontier_attribute[peer_].selector;
                    Total_Length = frontier_attribute[peer_].queue_length;
                } else {
                    Total_Length = 0;
                    for (peer__=0;peer__<num_gpus;peer__++)
                        data_slice->out_length[peer__]=0;
                }
                if (Enactor::DEBUG)
                {
                    printf("%d\t %lld\t \t Fullqueue finished. Total_Length= %d\n",
                        thread_num, enactor_stats[0].iteration, Total_Length);
                    fflush(stdout);
                }
                frontier_queue_ = &(data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpus]);
                if (num_gpus==1) data_slice->out_length[0]=Total_Length;
            }

            if (num_gpus > 1)
            {
                Iteration::Iteration_Update_Preds(
                    graph_slice,
                    data_slice,
                    &frontier_attribute[0],
                    &data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpus],
                    Total_Length,
                    streams[0]);
                Iteration::template Make_Output <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                    thread_num,
                    Total_Length,
                    num_gpus,
                    &data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpus],
                    &data_slice->scanned_edges[0],
                    &frontier_attribute[0],
                    enactor_stats,
                    &problem->data_slices[thread_num],
                    graph_slice,
                    &work_progress[0],
                    context[0],
                    streams[0]);
            }
            else
            {
                data_slice->out_length[0]= Total_Length;
            }

            for (peer_=0;peer_<num_gpus;peer_++)
                frontier_attribute[peer_].queue_length = data_slice->out_length[peer_];
        }
        Iteration::Iteration_Change(enactor_stats->iteration);
    }
}

/**
 * @brief Base class for graph problem enactor.
 *
 * @tparam SizeT
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template <
    typename SizeT,
    bool     _DEBUG,  // if DEBUG is set, print details to STDOUT
    bool     _SIZE_CHECK>
class EnactorBase
{
public:
    static const bool DEBUG = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;
    int           num_gpus;
    int          *gpu_idx;
    FrontierType  frontier_type;

    //Device properties
    util::Array1D<SizeT, util::CudaProperties>          cuda_props        ;

    // Queue size counters and accompanying functionality
    util::Array1D<SizeT, util::CtaWorkProgressLifetime> work_progress     ;
    util::Array1D<SizeT, EnactorStats>                  enactor_stats     ;
    util::Array1D<SizeT, FrontierAttribute<SizeT> >     frontier_attribute;

    FrontierType GetFrontierType() {return frontier_type;}

protected:

    /**
     * @brief Constructor
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] num_gpus
     * @param[in] gpu_idx
     */
    EnactorBase(
        FrontierType  frontier_type,
        int           num_gpus,
        int          *gpu_idx)
    {
        this->frontier_type = frontier_type;
        this->num_gpus      = num_gpus;
        this->gpu_idx       = gpu_idx;
        cuda_props        .SetName("cuda_props"        );
        work_progress     .SetName("work_progress"     );
        enactor_stats     .SetName("enactor_stats"     );
        frontier_attribute.SetName("frontier_attribute");
        cuda_props        .Init(num_gpus         , util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        work_progress     .Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        enactor_stats     .Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        frontier_attribute.Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            // Setup work progress (only needs doing once since we maintain
            // it in our kernel code)
            cuda_props   [gpu].Setup(gpu_idx[gpu]);
            for (int peer=0;peer<num_gpus;peer++)
            {
                work_progress     [gpu*num_gpus+peer].template Setup<SizeT>();
                //frontier_attribute[gpu*num_gpus+peer].output_length.Allocate(1, util::HOST | util::DEVICE);
                frontier_attribute[gpu*num_gpus + peer].output_length.Init(1, util::HOST | util::DEVICE, true);
            }
        }
    }

    /**
     * @brief EnactorBase destructor
     */
    virtual ~EnactorBase()
    {
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            for (int peer=0;peer<num_gpus;peer++)
            {
                enactor_stats     [gpu*num_gpus+peer].node_locks    .Release();
                enactor_stats     [gpu*num_gpus+peer].node_locks_out.Release();
                enactor_stats     [gpu*num_gpus+peer].edges_queued  .Release();
                enactor_stats     [gpu*num_gpus+peer].nodes_queued  .Release();
                frontier_attribute[gpu*num_gpus+peer].output_length .Release();
                if (work_progress [gpu*num_gpus+peer].HostReset()) return;
            }
        }
        work_progress     .Release();
        cuda_props        .Release();
        enactor_stats     .Release();
        frontier_attribute.Release();
    }

   /**
     * @brief Init function for enactor base class.
     *
     * @tparam Problem
     *
     * @param[in] problem The problem object for the graph primitive
     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
     * @param[in] advance_occupancy CTA Occupancy for Advance operator
     * @param[in] filter_occupancy CTA Occupancy for Filter operator
     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 256 by default.
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    template <typename Problem>
    cudaError_t Init(
        Problem *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            for (int peer=0;peer<num_gpus;peer++)
            {
                EnactorStats *enactor_stats_ = enactor_stats + gpu*num_gpus + peer;
                //initialize runtime stats
                enactor_stats_ -> advance_grid_size = MaxGridSize(gpu, advance_occupancy, max_grid_size);
                enactor_stats_ -> filter_grid_size  = MaxGridSize(gpu, filter_occupancy, max_grid_size);

                if (retval = enactor_stats_ -> advance_kernel_stats.Setup(enactor_stats_->advance_grid_size)) return retval;
                if (retval = enactor_stats_ ->  filter_kernel_stats.Setup(enactor_stats_->filter_grid_size)) return retval;
                if (retval = enactor_stats_ -> node_locks    .Allocate(node_lock_size, util::DEVICE)) return retval;
                if (retval = enactor_stats_ -> node_locks_out.Allocate(node_lock_size, util::DEVICE)) return retval;
                if (retval = enactor_stats_ -> nodes_queued  .Allocate(1, util::DEVICE | util::HOST)) return retval;
                if (retval = enactor_stats_ -> edges_queued  .Allocate(1, util::DEVICE | util::HOST)) return retval;
            }
        }
        return retval;
    }

    /*
     * @brief Reset function.
     */
    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            for (int peer=0; peer<num_gpus; peer++)
            {
                EnactorStats *enactor_stats_ = enactor_stats + gpu*num_gpus + peer;
                enactor_stats_ -> iteration             = 0;
                enactor_stats_ -> total_runtimes        = 0;
                enactor_stats_ -> total_lifetimes       = 0;
                enactor_stats_ -> nodes_queued[0]       = 0;
                enactor_stats_ -> edges_queued[0]       = 0;
                enactor_stats_ -> nodes_queued.Move(util::HOST, util::DEVICE);
                enactor_stats_ -> edges_queued.Move(util::HOST, util::DEVICE);
            }
        }
        return retval;
    }

    /**
     * @brief Setup function for enactor base class.
     *
     * @tparam Problem
     *
     * @param[in] problem The problem object for the graph primitive
     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
     * @param[in] advance_occupancy CTA Occupancy for Advance operator
     * @param[in] filter_occupancy CTA Occupancy for Filter operator
     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 256 by default.
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    template <typename Problem>
    cudaError_t Setup(
        Problem *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = Init(problem, max_grid_size, advance_occupancy, filter_occupancy, node_lock_size)) return retval;
        if (retval = Reset()) return retval;
        return retval;
    }

    /**
     * @brief Utility function for getting the max grid size.
     *
     * @param[in] gpu
     * @param[in] cta_occupancy CTA occupancy for current architecture
     * @param[in] max_grid_size Preset max grid size. If less or equal to 0, fully populate all SMs
     *
     * \return The maximum number of thread blocks this enactor class can launch.
     */
    int MaxGridSize(int gpu, int cta_occupancy, int max_grid_size = 0)
    {
        if (max_grid_size <= 0) {
            max_grid_size = this->cuda_props[gpu].device_props.multiProcessorCount * cta_occupancy;
        }

        return max_grid_size;
    }
};

/*
 * @brief IterationBase data structure.
 *
 * @tparam AdvanceKernelPolicy
 * @tparam FilterKernelPolicy
 * @tparam Enactor
 * @tparam _HAS_SUBQ
 * @tparam _HAS_FULLQ
 * @tparam _BACKWARD
 * @tparam _FORWARD
 * @tparam _UPDATE_PREDECESSORS
 */
template <
    typename _AdvanceKernelPolicy,
    typename _FilterKernelPolicy,
    typename _Enactor,
    bool     _HAS_SUBQ,
    bool     _HAS_FULLQ,
    bool     _BACKWARD,
    bool     _FORWARD,
    bool     _UPDATE_PREDECESSORS>
struct IterationBase
{
public:
    typedef _Enactor            Enactor   ;
    typedef _AdvanceKernelPolicy AdvanceKernelPolicy;
    typedef _FilterKernelPolicy  FilterKernelPolicy;
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    static const bool INSTRUMENT = Enactor::INSTRUMENT;
    static const bool DEBUG      = Enactor::DEBUG;
    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
    static const bool HAS_SUBQ   = _HAS_SUBQ;
    static const bool HAS_FULLQ  = _HAS_FULLQ;
    static const bool BACKWARD   = _BACKWARD;
    static const bool FORWARD    = _FORWARD;
    static const bool UPDATE_PREDECESSORS = _UPDATE_PREDECESSORS;

    /*
     * @brief SubQueue_Gather function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void SubQueue_Gather(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief SubQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void SubQueue_Core(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief FullQueue_Gather function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Gather(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief FullQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Core(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief Stop_Condition check function.
     *
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] num_gpus Number of GPUs used.
     */
    static bool Stop_Condition(
        EnactorStats                  *enactor_stats,
        FrontierAttribute<SizeT>      *frontier_attribute,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice,
        int                            num_gpus)
    {
        return All_Done(enactor_stats,frontier_attribute,data_slice,num_gpus);
    }

    /*
     * @brief Iteration_Change function.
     *
     * @param[in] iterations
     */
    static void Iteration_Change(long long &iterations)
    {
        iterations++;
    }

    /*
     * @brief Iteration_Update_Preds function.
     *
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] num_elements Number of elements.
     * @param[in] stream CUDA stream.
     */
    static void Iteration_Update_Preds(
        GraphSlice                    *graph_slice,
        DataSlice                     *data_slice,
        FrontierAttribute<SizeT>
                                      *frontier_attribute,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        SizeT                          num_elements,
        cudaStream_t                   stream)
    {
        if (num_elements == 0) return;
        int selector    = frontier_attribute->selector;
        int grid_size   = num_elements / 256;
        if ((num_elements % 256) !=0) grid_size++;
        if (grid_size > 512) grid_size = 512;

        if (Problem::MARK_PREDECESSORS && UPDATE_PREDECESSORS && num_elements>0 )
        {
            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, stream>>>(
                num_elements,
                frontier_queue->keys[selector].GetPointer(util::DEVICE),
                data_slice    ->preds         .GetPointer(util::DEVICE),
                data_slice    ->temp_preds    .GetPointer(util::DEVICE));

            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,stream>>>(
                num_elements,
                graph_slice   ->nodes,
                frontier_queue->keys[selector] .GetPointer(util::DEVICE),
                graph_slice   ->original_vertex.GetPointer(util::DEVICE),
                data_slice    ->temp_preds     .GetPointer(util::DEVICE),
                data_slice    ->preds          .GetPointer(util::DEVICE));//,
        }
    }

    /*
     * @brief Check frontier queue size function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] request_length Request frontier queue length.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     */
    static void Check_Queue_Size(
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        GraphSlice                    *graph_slice)
    {
        bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (Enactor::DEBUG)
            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                thread_num, iteration, peer_,
                frontier_queue->keys[selector^1].GetSize(),
                request_length);fflush(stdout);

        if (enactor_stats->retval =
            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector^1], over_sized, thread_num, iteration, peer_, false)) return;
        if (enactor_stats->retval =
            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector  ], over_sized, thread_num, iteration, peer_, true )) return;
        if (Problem::USE_DOUBLE_BUFFER)
        {
            if (enactor_stats->retval =
                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector^1], over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval =
                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector  ], over_sized, thread_num, iteration, peer_, true )) return;
        }
    }

    /*
     * @brief Make_Output function.
     *
     * @tparam NUM_VERTEX_ASSOCIATES
     * @tparam NUM_VALUE__ASSOCIATES
     *
     * @param[in] thread_num Number of threads.
     * @param[in] num_elements
     * @param[in] num_gpus Number of GPUs used.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    static void Make_Output(
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_gpus,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice_,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        if (num_gpus < 2) return;
        bool over_sized = false, keys_over_sized = false;
        int peer_ = 0, t=0, i=0;
        size_t offset = 0;
        SizeT *t_out_length = new SizeT[num_gpus];
        int selector = frontier_attribute->selector;
        int block_size = 256;
        int grid_size  = num_elements / block_size;
        if ((num_elements % block_size)!=0) grid_size ++;
        if (grid_size > 512) grid_size=512;
        DataSlice* data_slice=data_slice_->GetPointer(util::HOST);

        for (peer_ = 0; peer_<num_gpus; peer_++)
        {
            t_out_length[peer_] = 0;
            data_slice->out_length[peer_] = 0;
        }
        if (num_elements ==0) return;

        over_sized = false;
        for (peer_ = 0; peer_<num_gpus; peer_++)
        {
            if (enactor_stats->retval =
                Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> ("keys_marker", num_elements, &data_slice->keys_marker[peer_], over_sized, thread_num, enactor_stats->iteration, peer_)) break;
            if (over_sized) data_slice->keys_markers[peer_]=data_slice->keys_marker[peer_].GetPointer(util::DEVICE);
        }
        if (enactor_stats->retval) return;
        if (over_sized) data_slice->keys_markers.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);

        for (t=0; t<2; t++)
        {
            if (t==0 && !FORWARD) continue;
            if (t==1 && !BACKWARD) continue;

            if (BACKWARD && t==1)
                Assign_Marker_Backward<VertexId, SizeT>
                    <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
                    graph_slice   ->backward_offset   .GetPointer(util::DEVICE),
                    graph_slice   ->backward_partition.GetPointer(util::DEVICE),
                    data_slice    ->keys_markers      .GetPointer(util::DEVICE));
            else if (FORWARD && t==0)
                Assign_Marker<VertexId, SizeT>
                    <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
                    graph_slice   ->partition_table   .GetPointer(util::DEVICE),
                    data_slice    ->keys_markers      .GetPointer(util::DEVICE));

            for (peer_=0;peer_<num_gpus;peer_++)
            {
                Scan<mgpu::MgpuScanTypeInc>(
                    (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                    num_elements,
                    (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
                    (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                    context[0]);
            }

            if (num_elements>0) for (peer_=0; peer_<num_gpus;peer_++)
            {
                cudaMemcpyAsync(&(t_out_length[peer_]),
                    data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
                        + (num_elements -1),
                    sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
            } else {
                for (peer_=0;peer_<num_gpus;peer_++)
                    t_out_length[peer_]=0;
            }
            if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;

            keys_over_sized = true;
            for (peer_=0; peer_<num_gpus;peer_++)
            {
                if (enactor_stats->retval =
                    Check_Size <Enactor::SIZE_CHECK, SizeT, VertexId> (
                        "keys_out",
                        data_slice->out_length[peer_] + t_out_length[peer_],
                        peer_!=0 ? &data_slice->keys_out[peer_] :
                                   &data_slice->frontier_queues[0].keys[selector^1],
                        keys_over_sized, thread_num, enactor_stats[0].iteration, peer_),
                        data_slice->out_length[peer_]==0? false: true) break;
                if (keys_over_sized)
                    data_slice->keys_outs[peer_] = peer_==0 ?
                        data_slice->frontier_queues[0].keys[selector^1].GetPointer(util::DEVICE) :
                        data_slice->keys_out[peer_].GetPointer(util::DEVICE);
                if (peer_ == 0) continue;

                over_sized = false;
                for (i=0;i<NUM_VERTEX_ASSOCIATES;i++)
                {
                    if (enactor_stats[0].retval =
                        Check_Size <Enactor::SIZE_CHECK, SizeT, VertexId>(
                            "vertex_associate_outs",
                            data_slice->out_length[peer_] + t_out_length[peer_],
                            &data_slice->vertex_associate_out[peer_][i],
                            over_sized, thread_num, enactor_stats->iteration, peer_),
                            data_slice->out_length[peer_]==0? false: true) break;
                    if (over_sized) data_slice->vertex_associate_outs[peer_][i] = data_slice->vertex_associate_out[peer_][i].GetPointer(util::DEVICE);
                }
                if (enactor_stats->retval) break;
                if (over_sized) data_slice->vertex_associate_outs[peer_].Move(util::HOST, util::DEVICE, NUM_VERTEX_ASSOCIATES, 0, stream);

                over_sized = false;
                for (i=0;i<NUM_VALUE__ASSOCIATES;i++)
                {
                    if (enactor_stats->retval =
                        Check_Size<Enactor::SIZE_CHECK, SizeT, Value   >(
                            "value__associate_outs",
                            data_slice->out_length[peer_] + t_out_length[peer_],
                            &data_slice->value__associate_out[peer_][i],
                            over_sized, thread_num, enactor_stats->iteration, peer_,
                            data_slice->out_length[peer_]==0? false: true)) break;
                    if (over_sized) data_slice->value__associate_outs[peer_][i] = data_slice->value__associate_out[peer_][i].GetPointer(util::DEVICE);
                }
                if (enactor_stats->retval) break;
                if (over_sized) data_slice->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, NUM_VALUE__ASSOCIATES, 0, stream);
            }
            if (enactor_stats->retval) break;
            if (keys_over_sized) data_slice->keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);

            offset = 0;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> keys_markers         .GetPointer(util::HOST),
                      sizeof(SizeT*   ) * num_gpus);
            offset += sizeof(SizeT*   ) * num_gpus ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> keys_outs            .GetPointer(util::HOST),
                      sizeof(VertexId*) * num_gpus);
            offset += sizeof(VertexId*) * num_gpus ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
                      sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
            offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> value__associate_orgs.GetPointer(util::HOST),
                      sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
            offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
            for (peer_=0; peer_<num_gpus; peer_++)
            {
                memcpy(&(data_slice->make_out_array[offset]),
                         data_slice->vertex_associate_outs[peer_].GetPointer(util::HOST),
                          sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
                offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
            }
            for (peer_=0; peer_<num_gpus; peer_++)
            {
                memcpy(&(data_slice->make_out_array[offset]),
                        data_slice->value__associate_outs[peer_].GetPointer(util::HOST),
                          sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
                offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
            }
            memcpy(&(data_slice->make_out_array[offset]),
                     data_slice->out_length.GetPointer(util::HOST),
                      sizeof(SizeT) * num_gpus);
            offset += sizeof(SizeT) * num_gpus;
            data_slice->make_out_array.Move(util::HOST, util::DEVICE, offset, 0, stream);

            if (BACKWARD && t==1)
                Make_Out_Backward<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
                    graph_slice   -> backward_offset     .GetPointer(util::DEVICE),
                    graph_slice   -> backward_partition  .GetPointer(util::DEVICE),
                    graph_slice   -> backward_convertion .GetPointer(util::DEVICE),
                    offset,
                    data_slice    -> make_out_array      .GetPointer(util::DEVICE));
            else if (FORWARD && t==0)
                Make_Out<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
                    graph_slice   -> partition_table     .GetPointer(util::DEVICE),
                    graph_slice   -> convertion_table    .GetPointer(util::DEVICE),
                    offset,
                    data_slice    -> make_out_array      .GetPointer(util::DEVICE));
            for (peer_ = 0; peer_<num_gpus; peer_++)
            {
                data_slice->out_length[peer_] += t_out_length[peer_];
            }
        }
        if (enactor_stats->retval) return;
        if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;
        frontier_attribute->selector^=1;
        if (t_out_length!=NULL) {delete[] t_out_length; t_out_length=NULL;}
    }

};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
