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

namespace gunrock {
namespace util {

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
    Csr<VertexId, Value, SizeT> *csr_query_ptr; // pointer to CSR input query graph
    Csr<VertexId, Value, SizeT> *csr_data_ptr; // pointer to CSR input data graph
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
        info["compiler"]           = "";     // what compiled this program?
        info["compiler_version"]   = "";     // what version compiler?
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
        info["scaled"]             = false;  // default scaled for PageRank
        info["alpha"]              = 6.0f;   // default alpha for DOBFS
        info["beta"]               = 6.0f;   // default beta for DOBFS
        info["top_nodes"]          = 0;      // default number of nodes for top-k primitive
        info["normalized"]         = false;  // default normalized for PageRank
	info["multi_graphs"]       = false;  // default only one input graph
        info["node_value"]         = false;  // default don't load labels
        info["label"]              = "";     // label file name used in test
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
#if BOOST_COMP_CLANG
        info["compiler"] = BOOST_COMP_CLANG_NAME;
        info["compiler_version"] = BOOST_COMP_CLANG_DETECTION;
#elif BOOST_COMP_GNUC
        info["compiler"] = BOOST_COMP_GNUC_NAME;
        info["compiler_version"] = BOOST_COMP_GNUC_DETECTION;
#endif /* BOOST_COMP */
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
        info["scaled"    ] =  args.CheckCmdLineFlag("scaled"    ); // PR

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
	// Special initialization for SM problem
        if(algorithm_name == "SM") return Init_SM(args,csr_ref,csc_ref);

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

            std::ifstream fp(market_filename);
            if (market_filename == NULL||!fp.is_open())
            {
                fprintf(stderr, "Input graph file %s does not exist.\n",market_filename);
                exit (EXIT_FAILURE);
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
            info["rmat_scale"] = (int64_t)rmat_scale;
            info["rmat_nodes"] = (int64_t)rmat_nodes;
            info["rmat_edges"] = (int64_t)rmat_edges;
            info["rmat_edgefactor"] = (int64_t)rmat_edgefactor;

            util::CpuTimer cpu_timer;
            cpu_timer.Start();

            // generate R-MAT graph
            if (graphio::rmat::BuildRmatGraph<EDGE_VALUE>(
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
            info["rgg_scale"]       = (int64_t)rgg_scale;
            info["rgg_nodes"]       = (int64_t)rgg_nodes;
            info["rgg_thfactor"]    = rgg_thfactor;
            info["rgg_threshold"]   = rgg_threshold;
            info["rgg_vmultipiler"] = rgg_vmultipiler;

            util::CpuTimer cpu_timer;
            cpu_timer.Start();

            // generate random geometry graph
            if (graphio::rgg::BuildRggGraph<EDGE_VALUE>(
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
            exit(EXIT_FAILURE);
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
     * @brief SM Utility function to load input graph.
     *
     * @tparam NODE_VALUE
     *
     * @param[in] args Command line arguments.
     * @param[in] csr_ref Reference to the CSR graph.
     *
     * \return int whether successfully loaded the graph (0 success, 1 error).
     */
    template<bool NODE_VALUE>
    int LoadGraph_SM(
        util::CommandLineArgs &args,
        Csr<VertexId, Value, SizeT> &csr_ref,
        std::string type)
    {
        std::string graph_type = args.GetCmdLineArgvGraphType();
        if (graph_type == "market")  // Matrix-market graph
        {
            if (!args.CheckCmdLineFlag("quiet"))
            {
                printf("Loading Matrix-market coordinate-formatted graph ...\n");
            }
            char *market_filename = NULL;
            char *label_filename = NULL;

            if(type=="query"){
                market_filename = args.GetCmdLineArgvQueryDataset();
                if(NODE_VALUE)
                    label_filename = args.GetCmdLineArgvQueryLabel();
            }
            else
            {
                market_filename = args.GetCmdLineArgvDataDataset();
                if(NODE_VALUE)
                    label_filename = args.GetCmdLineArgvDataLabel();
            }

            if (market_filename == NULL)
            {
                printf("Log.");
                fprintf(stderr, "Input graph does not exist.\n");
                return 1;
            }

            if (NODE_VALUE && label_filename == NULL)
            {
                printf("Log.");
                fprintf(stderr, "Input graph labels does not exist.\n");
                return 1;
            }

            boost::filesystem::path market_filename_path(market_filename);
            file_stem = market_filename_path.stem().string();
            info["dataset"] = file_stem;
            if (graphio::BuildMarketGraph_SM<NODE_VALUE>(
                        market_filename,
                        label_filename,
                        csr_ref,
			 info["undirected"].get_bool(),
                        false,
                        args.CheckCmdLineFlag("quiet")) != 0)
            {
                return 1;
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
     * @brief SM Initialization process for Info.
     *
     * @param[in] algorithm_name Algorithm name.
     * @param[in] args Command line arguments.
     * @param[in] csr_query_ref Reference to the CSR structure.
     * @param[in] csr_data_ref Reference to the CSR structure.
     */
    void Init_SM(
        util::CommandLineArgs &args,
        Csr<VertexId, Value, SizeT> &csr_query_ref,
        Csr<VertexId, Value, SizeT> &csr_data_ref)
    {
        if(info["node_value"].get_bool()){
            LoadGraph_SM<true>(args,csr_query_ref, "query");
            LoadGraph_SM<true>(args,csr_data_ref, "data");
        }
        else{
        LoadGraph_SM<false>(args,csr_query_ref, "query");
        LoadGraph_SM<false>(args,csr_data_ref, "data");
        }
        csr_query_ptr = &csr_query_ref;
        csr_data_ptr = &csr_data_ref;

        InitBase("SM", args);
    }


    /**
     * @brief Compute statistics common to all primitives.
     *
     * @param[in] enactor_stats
     * @param[in] elapsed
     * @param[in] labels
     * @param[in] get_traversal_stats
     */
    template <typename EnactorStats>
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
    template <typename EnactorStats>
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
} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
