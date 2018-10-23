// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_pr.cu
 *
 * @brief Simple test driver program for PageRank.
 */

#include <gunrock/app/pr/pr_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

cudaError_t UseParameters(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(parameters.Use<int>(
        "num-elements",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        1024 * 1024,
        "number of elements per GPU to test on",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "for-size",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        1024 * 256,
        "number of operations to perform per repeat",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "num-repeats",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        100,
        "number of times to repeat the operations",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "device",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "the devices to run on",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "rand-seed",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        util::PreDefinedValues<int>::InvalidValue,
        "rand seed to generate random numbers; default is time(NULL)",
        __FILE__, __LINE__));

    return retval;
}

// Test routines

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<float> Distribution;

template <typename GraphT>
cudaError_t Test_BWL(
    util::Parameters &parameters, GraphT &graph)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    cudaError_t retval = cudaSuccess;
    SizeT num_elements = parameters.template Get<SizeT>("num-elements");
    SizeT for_size     = parameters.template Get<SizeT>("for-size");
    SizeT num_repeats  = parameters.template Get<SizeT>("num-repeats");
    int   rand_seed    = parameters.template Get<int  >("rand-seed");
    auto  devices      = parameters.template Get<std::vector<int>>("device");
    int   num_devices  = devices.size();
    cudaError_t *retvals = new cudaError_t[num_devices];
    util::Array1D<SizeT, VertexT>* gpu_elements
        = new util::Array1D<SizeT, VertexT>[num_devices];
    util::Array1D<SizeT, VertexT>* gpu_results
        = new util::Array1D<SizeT, VertexT>[num_devices];
    cudaStream_t* gpu_streams = new cudaStream_t[num_devices];
    if (!util::isValid(rand_seed))
        rand_seed = time(NULL);
    float ***timings   = new float**[num_devices];   
 
    util::PrintMsg("num_devices = " + std::to_string(num_devices));
    util::PrintMsg("rand-seed = " + std::to_string(rand_seed));
    #pragma omp parallel num_threads(num_devices)
    { do {
        int thread_num = omp_get_thread_num();
        auto  device_idx = devices   [thread_num];
        auto &retval   = retvals     [thread_num];
        auto &elements = gpu_elements[thread_num];
        auto &results  = gpu_elements[thread_num];
        auto &stream   = gpu_streams [thread_num];
        int  *peer_accessable = new int[num_devices + 1];
        timings[thread_num] = new float*[num_devices + 1];
        for (int i = 0; i <= num_devices; i++)
        {
            timings[thread_num][i] = new float[100];
            peer_accessable[i] = 1;
        }

        util::PrintMsg("using device[" + std::to_string(thread_num)
            + "] " + std::to_string(device_idx));
        retval = util::GRError(cudaSetDevice(device_idx),
            "cudaSetDevice failed.");
        if (retval) break;
        retval = util::GRError(cudaStreamCreateWithFlags(
            &stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags failed.");
        if (retval) break;
        
        for (int peer_offset = 1; peer_offset < num_devices; peer_offset++)
        {
            int peer = devices[(thread_num + peer_offset) % num_devices];
            int peer_access_avail = 0;
            retval = util::GRError(cudaDeviceCanAccessPeer(
                &peer_access_avail, device_idx, peer),
                "cudaDeviceCanAccessPeer failed");
            if (retval) break;
            if (peer_access_avail)
            {
                retval = util::GRError(cudaDeviceEnablePeerAccess(peer, 0),
                    "cudaDeviceEnablePeerAccess failed");
                if (retval) break;
            } else {
                peer_accessable[peer] = 0;
            }
            if (retval) break;
        }
        if (retval) break;
 
        elements.SetName("elements[" + std::to_string(thread_num) + "]");
        retval = elements.Allocate(num_elements, util::DEVICE | util::HOST);
        if (retval) break;
        results.SetName("results[" + std::to_string(thread_num) + "]");
        retval = results.Allocate(max(num_elements, for_size), util::DEVICE);
        if (retval) break;

        Engine engine(rand_seed + 11 * thread_num);
        Distribution distribution(0.0, 1.0);
        for (SizeT i = 0; i < num_elements; i++)
        {
            elements[i] = distribution(engine) * num_elements;
            if (elements[i] >= num_elements)
                elements[i] -= num_elements;
        }
        retval = elements.Move(util::HOST, util::DEVICE, 
            num_elements, 0, stream);
        if (retval) break;
        retval = util::GRError(cudaStreamSynchronize(stream),
            "cudaStreamSynchonorize failed");
        if (retval) break;
        #pragma omp barrier

        for (int peer_offset = 0; peer_offset < num_devices; peer_offset++)
        for (int small_large = 0; small_large <= 1; small_large++)
        for (int access_type = 0; access_type <= 1; access_type++) 
        for (int operation_type = 0; operation_type <= 2; operation_type++)
        {
            int peer = (thread_num + peer_offset) % num_devices;
            auto peer_elements = gpu_elements[peer].GetPointer(util::DEVICE);
            auto peer_results  = gpu_results [peer].GetPointer(util::DEVICE);  
            float elapsed = -1;

            if (peer_accessable[peer] != 0)
            {
                util::CpuTimer cpu_timer;
                cpu_timer.Start();

                retval = results.ForAll(
                    [elements, peer_elements, peer_results, num_elements,
                    small_large, access_type, operation_type, num_repeats] 
                    __host__ __device__ (VertexT *result, const SizeT &pos)
                    {
                        VertexT val = 0;
                        for (int i = 0; i < num_repeats; i++)
                        {
                            VertexT new_pos = pos + i * 16384;
                            new_pos = new_pos % num_elements;
                            if (access_type == 1)
                                new_pos = elements[pos];
                            
                            if (operation_type == 0)
                            {
                                result[pos] = peer_elements[new_pos];
                            } 

                            else if (operation_type == 1)
                            {
                                peer_results[new_pos] = new_pos;
                            }

                            else if (operation_type == 2)
                            {
                                peer_results[new_pos] += 1;
                            }
                        }
                    }, (small_large == 0) ? 1 : for_size, util::DEVICE, stream);
                if (retval) break; 
                retval = util::GRError(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize failed");
                cpu_timer.Stop();
                elapsed = cpu_timer.ElapsedMillis();
            }
            timings[thread_num][peer]
                [(small_large * 2 + access_type) * 3 + operation_type]
                = elapsed; 
            #pragma omp barrier
        }
        if (retval) break;

        #pragma omp barrier
        retval = elements.Release();
        if (retval) break;
        retval = results.Release();
        if (retval) break;
    } while (false); }
    return retval;
}


/******************************************************************************
 * Main
 ******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct
{
    /**
     * @brief the actual main function, after type switching
     * @tparam VertexT    Type of vertex identifier
     * @tparam SizeT      Type of graph size, i.e. type of edge identifier
     * @tparam ValueT     Type of edge values
     * @param  parameters Command line parameters
     * @param  v,s,val    Place holders for type deduction
     * \return cudaError_t error message(s), if any
     */
    template <
        typename VertexT, // Use int as the vertex identifier
        typename SizeT,   // Use int as the graph size type
        typename ValueT>  // Use int as the value type
    cudaError_t operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val)
    {
        typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_COO> GraphT;
        cudaError_t retval = cudaSuccess;

        GraphT graph;
        std::vector<std::string> switches{"num-elements", "for-size", "num-repeats"};

        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [](util::Parameters &parameters, GraphT &graph)
            {
                return Test_BWL(parameters, graph);
            }));
        return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test pr");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(UseParameters(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());

    return app::Switch_Types<
        app::VERTEXT_U32B | app::VERTEXT_U64B |
        app::SIZET_U32B | //app::SIZET_U64B |
        app::VALUET_F32B | //app::VALUET_F64B |
        app::DIRECTED | app::UNDIRECTED>
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
