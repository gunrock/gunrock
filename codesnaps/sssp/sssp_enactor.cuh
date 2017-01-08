// The enactor defines how a graph primitive runs. It calls traversal
// (advance and filter operators) and computation (functors).

// This SSSP example also illustrates what need to be supplied to the
// multi-GPU framework, by the implementor of a graph primitive, so that
// the primitive can be maily written in a single GPU manner, and the
// framework can expant it to utilize multiple GPUs in a single node.
//
// Different parts that fits into the multi-GPU framework are as following:
// 1) Core single-GPU primitive : 
//      the FullQueue_Core(...) function, line 160 to 204
// 2) Data to communicate : 
//      defined in the sssp_problem.cuh file in the same directory of this 
//      file, line 27, 31, and 100 to 106
// 3) Combining remote and local data : 
//      in Expand_Incoming_Kernel(...) GPU kernel, line 50 to 60
// 4) Stop condition : 
//      SSSP uses the default stop condition (all frontiers are empty, or 
//      any GPU encounters error), which is defined by the 
//      IterationBase::Stop_Condition(...) function (line 991 in 
//      gunrock/app/enactor_loop.cuh). Primitive implementator can overload 
//      this function, if non-default stop condition is required.


template<
    typename KernelPolicy,     // `type of kernelpolicy, includes defination of VertexId, SizeT and Value
    int NUM_VERTEX_ASSOCIATES, // number of vertex associative of type VertexId transmitted with remote sub-frontiers 
    int NUM_VALUE_ASSOCIATES>  // number of vertex associative of type Value transmitted with remote sub-frontiers
__global__ void Expand_Incoming_Kernel(
// Kernel to combine received and local data
             int                     thread_num,            // thread number
    typename KernelPolicy::VertexId  label,                 // label, equal to iteration number here
    typename KernelPolicy::SizeT     num_elements,          // size of recevied sub-frontier
    typename KernelPolicy::VertexId *d_keys_in,             // received sub-frontier
    typename KernelPolicy::VertexId *d_vertex_associate_in, // received associatives of VertexId type
    typename KernelPolicy::Value    *d_value__associate_in, // received associatives of Value type
    typename KernelPolicy::SizeT    *d_out_length,          // output frontier size counter, may not start with 0
    typename KernelPolicy::VertexId *d_keys_out,            // output frontier
    typename KernelPolicy::VertexId *d_preds,               // the local per-vertex predecessor marker
    typename KernelPolicy::Value    *d_distances,           // the local per-vertex distances marker
    typename KernelPolicy::VertexId *d_labels)              // the local pre-vertex label marker
{
    //Below is algorithmicly equal to gunrock::app::sssp::Expan_Incoming_Kernel,
    //but without optimizations.

    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    while (x < num_elements) // for each element in the received sub-frontier
    {
        VertexId key          = d_keys_in[x];                           // received vertex Id
        Value    distance     = d_value__associate_in[x];               // received distance
        Value    old_distance = atomicMin(d_distances + key, distance); // compared with local distance
        if (old_value > value && d_labels[key] != label)
        { // only when local distance got updated, and vertex not in local frontier
            d_labels[key] = label;
            if (NUM_VERTEX_ASSOCIATES == 1 && d_distances[key] == distance)
            // if need to mark predecessors, and the curent distance is still not changed yet
                d_preds[key] = d_vertex_associate_in[x];  // assign the received predecessor
            d_keys_out[atomicAdd(d_out_length, 1)] = key; // put vertex in output frontier
        }
        
        x += STRIDE; // presistant thread loop
    }
}

template<
    typename AdvanceKernelPolicy, // Kernel policy for advance
    typename FilterKernelPolicy,  // Kernel policy for filter
    typename Enactor>             // type of enactor
struct SSSPIteration : public IterationBase < // Based on IterationBase
// Main iteration loop for SSSP primitive
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    false,  //HAS_SUBQ = false, don't have SubQ_Core
    true,   //HAS_FULLQ = true, has FullQ_Core
    false,  //BACKWARD = false, communication is not backward
    true,   //FORWARD = true, communication is forward
    Enactor::Problem::MARK_PATHS> // MARK_PREDECESSORS, whether to mark predecessors
{
    // which Functor the iteration uses
    typedef SSSPFunctor<VertexId, SizeT, Value, Problem>
                                        Functor   ;

    template <
        int NUM_VERTEX_ASSOCIATES, // number of vertex associative of type VertexId transmitted with remote sub-frontiers 
        int NUM_VALUE__ASSOCIATES>  // number of vertex associative of type Value transmitted with remote sub-frontiers 
    static void Expand_Incoming(...)
    {
        // Check whether allocated memory space for the output array is sufficient first
        // set out_length[peer_] to be 0, when not using unified received
        Expand_Incoming_Kernel
            <AdvanceKernelPolicy, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<...>>>
            (gpu_idx,            // thread_num
            iteration,           // iteration_num
            num_elements,        // received sub-frontier size
            keys_in,             // received sub-frontier
            vertex_associate_in, // received associatives of VertexId type
            value__associate_in, // received associatives of Value type
            out_length,          // output frontier size
            keys_out,            // output frontier
            preds,               // local predecessor marker array
            distances,           // local distance array
            labels);             // local label array
        out_length.Move(DEVICE, HOST, ...);
    }

    static cudaError_t Compute_OutputLength(...)
    // Compute the resulted frontier size, for memory (re)allocation purpose
    {
        cudaError_t retval = cudaSuccess;
        if (!size_check // no need to check whether memory allocation is sufficient
            && !hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()) // advance does not require pre-scan
        {
            frontier_attribute -> output_length[0] = 0; // has no memory requirement
            return retval;
        } else {
            // check whether partitioned_scanned_edges array is sufficient
            if (retval = Size_Check<SizeT, SizeT>(
                size_check, "scanned_edges",
                frontir_atrribute -> queue_length + 2,
                partitioned_scanned_edges, ...))
                return retval;
            // perform a scan on the out-degrees of current frontier,
            // use the result for memory size check and possible reallocation
            // used for load balancing also
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <...>(frontier_attribute, d_in_key_queue, partitioned_scanned_edges, ...);
            // get the computed output size onto CPU
            frontier_attribute -> output_length.Move(DEVICE, HOST, 1, 0, stream);
            return retval;
        }
    }

    static void Check_Queue_Size(...)
    // Ensure output frontier allocation are sufficient
    {
        if (!size_check // no need to check whether memory allocation is sufficient
           && !hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()) // advance does not require pre-scan
        {
            frontier_attribute -> output_length[0] = 0;
            return retval; // no need to garentee sufficient memory
        } else if (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>()) // do not use kernel fusion
        {
            // make sure sufficient allocation for frontier between advance and filter
            if (retval = CheckSize<SizeT, VertexId>(
                ...,request_length, frontier_queue -> keys[selector^1], ...) 
                return;
            // make sure sufficient allocation for frontier after filter
            if (retval = CheckSize<SizeT, VertexId>(
                ...,graph_slice -> nodes * 1.2, frontier_queue -> keys[selector], ...) 
                return;
        } else { // when use kernel fusion
            // make sure sufficient allocation for frontier after filter
            if (retval = CheckSize<SizeT, VertexId>(
                ...,graph_slice -> nodes * 1.2, frontier_queue -> keys[selector^1], ...) 
                return;
        }
    }

    static void FullQueue_Core(...)
    // The main per-iteration computation step 
    {
        frontier_attribute -> queue_reset = true; // use frotnier_attribute -> queue_length as input frontier size
        
        // Edge Map
        // Traverse out going edges of vertices in the input frontier,
        // and for every such edge, evaluate Functor::CondEdge(...),
        // and if true, execute Functor::ApplyEdge(...)
        gunrock::oprtr::advance::LaunchKernel
            <..., Functor, ...>(
            frontier_attribute, // frontier attributes
            data_slice,         // problem associative data
            frontier_queue -> keys[frontier_attribute -> selector  ], // input frontier
            frontier_queue -> keys[frontier_attribute -> selector^1], // output frontier
            graph_slice, // row_offsets, column_indices, etc.
            ...);

        if (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>()) // if kernel fusion is not in effect
        {
            frontier_attribute -> queue_reset = false; // use outputed frontier length from advance, as input frontier length to filter
            frotnier_attribute -> queue_index ++;
            frontier_attribute -> selector ^= 1;       // aternate ping-pong queue

            // Vertex Map
            // for every vertex in the output frontier from advance,
            // evaluate Functor::CondFilter(...),
            // and if true, execute Funtor::ApplyFilter(...) and put such vertex in output frontier
            gunrock::oprtr::filter::LaunchKernel
                <..., Functor, ...>(
                frontier_attribute, // frontier_attributes,
                data_slice,         // problem associative data,
                frontier_queue -> keys[frontier_attribute -> selector  ], // input frontier
                frontier_queue -> keys[frontier_attribute -> selector^1], // output frontier
                graph_slice, // num_nodes, etc.
                ...); 
        }

        frotnier_attribute -> queue_index ++;
        frontier_attribute -> selector ^= 1;       // aternate ping-pong queue
        if (retval = work_progress -> GetQueueLength(
            frontier_attribute -> queue_index,
            frontier_attribute -> queue_length,
            ...)) return; // get the output queue length on CPU
    }

};

template<
    typename AdvanceKernelPolicy, // Kernel policy for advance
    typename FilterKernelPolicy,  // Kernel policy for filter
    typename Enactor>             // type of enactor
static CUT_THREADPROC SSSPThread(
// the per-GPU controlling thread on CPU
    void *thread_data_)           // data package to bypass the thread boundary
{
    // ... thread perparation
    
    if (retval = SetDevice(gpu_idx))
    { // if set device failed, quit
        thread_data -> status = ThreadSlice::Status::Ended;
        CUT_THREADEND;
    }

    thread_data -> status = ThreadSlice::Status::Idle; // ready, and waiting to enact
    while (thread_data -> status != ThreadSlice::Status::ToKill) // loop till instruct to kill
    {
        while (thread_data -> status == ThreadSlice::Status::Wait ||
               thread_data -> status == ThreadSlice::Status::Idle)
        {
            sleep(0); // wait until status changes
        }
        if (thread_data -> status == ThreadSlice::Status::ToKill)
            break; // end if instructed

        for (int peer_=0;peer_<num_gpus;peer_++)
        { // setup frontiers
            frontier_attribute[peer_].queue_index  = 0; // Work queue index
            frontier_attribute[peer_].queue_length = peer_==0?thread_data->init_size:0;
            frontier_attribute[peer_].selector     = 0;
            frontier_attribute[peer_].queue_reset  = true;
            enactor_stats     [peer_].iteration    = 0;
        }

        // Perform one SSSP iteration loop
        gunrock::app::Iteration_Loop
            <Enactor, Functor, SSSPIteration<...>, ...>
            (thread_data);

        thread_data -> status = ThreadSlice::Status::Idle; // signal work done
    }
    CUT_THREADEND;
}

template <typename Problem> // type of Problem
class SSSPEnactor : public EnactorBase<typename Problem::SizeT> 
{
    ThreadSlice *thread_slices; // thread data for CPU control threads
    CUTThread   *thread_Ids;    // thread Id for CPU control threads

public:

    // constructor of the Enactor
    SSSPEnactor(...)
    ...
    {}

    // destructor of the Eanctor
    virtual ~SSSPEnactor() 
    {
        Release(); // release allocated memory
    }

    // routine to release allocated memory
    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (thread_slices != NULL)
        {
            for (int gpu = 0; gpu < this->num_gpus; gpu++)
                thread_slices[gpu].status = ThreadSlice::Status::ToKill;
            cutWaitForThreads(thread_Ids, this->num_gpus); // Kill all GPU controling threads
            delete[] thread_Ids   ; thread_Ids    = NULL; 
            delete[] thread_slices; thread_slices = NULL; // deallocate thread data
        }
        if (retval = BaseEnactor::Release()) return retval; // release memory allocated by parent class
        problem = NULL;
        return retval;
    }

    template<
        typename AdvanceKernelPolicy, // kernel policy for advance operator
        typename FilterKernelPolicy>  // kernel policy for filter operator
    cudaError_t InitSSSP(             // initialize SSSP Enactor
        ContextPtr  *context,         // mGPU ContextPtr
        Problem     *problem,         // problem data
        int         max_grid_size = 0)
    {
        cudaError_t retval = cudaSuccess;

        // Initialize BaseEnactor
        if (retval = BaseEnactor::Init(
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
            return retval;

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus]; // GPU controlling thread data

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            // assign thread data
            thread_slices[gpu].thread_num    = gpu;
            thread_slices[gpu].problem       = (void*)problem;
            thread_slices[gpu].enactor       = (void*)this;
            thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
            thread_slices[gpu].status        = ThreadSlice::Status::Inited;
            // start the thread
            thread_slices[gpu].thread_Id     = cutStartThread(
                    (CUT_THREADROUTINE)&(SSSPThread<
                        AdvanceKernelPolicy, FilterKernelPolicy,
                        SSSPEnactor<Problem> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
        }

        // wait till all controlling threads ready
        for (int gpu=0; gpu < this->num_gpus; gpu++)
        {
            while (thread_slices[gpu].status != ThreadSlice::Status::Idle)
            {
                sleep(0);
            }
        }
        return retval;
    }

    // Routine to reset the Enactor
    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;
        // Reset the BaseEnactor
        if (retval = BaseEnactor::Reset())
            return retval;
        // Signal every controlling thread to wait
        for (int gpu=0; gpu < this -> num_gpus; gpu++)
            thread_slices[gpu].status = ThreadSlice::Status::Wait;
        return retval;
    }

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactSSSP(
        VertexId src)    // the source vertex
    {
        cudaError_t  retval     = cudaSuccess;

        // prepare initial frontier size for each GPU
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if ((this->num_gpus ==1) || (gpu==this->problem->partition_tables[0][src]))
                 thread_slices[gpu].init_size=1;
            else thread_slices[gpu].init_size=0;
            this->frontier_attribute[gpu*this->num_gpus].queue_length
                = thread_slices[gpu].init_size;
        }

        // singnal controlling threads to process
        for (int gpu=0; gpu< this->num_gpus; gpu++)
        {
            thread_slices[gpu].status = ThreadSlice::Status::Running;
        }

        // wait until all controling threads finish
        for (int gpu=0; gpu< this->num_gpus; gpu++)
        {
            while (thread_slices[gpu].status != ThreadSlice::Status::Idle)
            {
                sleep(0);
            }
        }

        // check whether has error
        for (int gpu=0; gpu<this->num_gpus * this -> num_gpus;gpu++)
        if (this->enactor_stats[gpu].retval!=cudaSuccess)
        {
            retval=this->enactor_stats[gpu].retval;
            return retval;
        }

        if (this -> debug) printf("\nGPU SSSP Done.\n");
        return retval;
    }

    //Define filter and advance KernelPolicy

    //Enact calling functions

    // Enact interface
    cudaError_t Enact(
        VertexId src,
        std::string traversal_mode = "LB")
    {
        // Select traversal mode, and SM version to call
        return EnactSSSP<...>(src);
    }

    // Init interface
    cudaError_t Init(...)
    {
        // Select traversal mode, and SM version to call
        return InitSSSP<...>(...);
    }
    
};
