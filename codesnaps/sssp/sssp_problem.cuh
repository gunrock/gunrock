// This data structure (the "Problem" struct) stores the graph
// topology in CSR format and the frontier. All Problem structs
// inherit from the ProblemBase struct. Algorithm-specific data is
// stored in a "DataSlice".

template<
    typename VertexId,
    typename SizeT,
    typename Value,
    bool _MARK_PATHS> // Whether to mark predecessor ID when advance
    struct SSSPProblem : public ProblemBase<VertexId, SizeT, Value,
        true,  // MARK_PREDECESSORS,
        false> // ENABLE_IDEMPOTENCE
{

    // MARK_PREDECESSORS sets the predecessor node ID during a
    // traversal for each node in the new frontier.
    static const bool MARK_PREDECESSORS     = true;

    // ENABLE_IDEMPOTENCE is an optimization when the operation
    // performed in parallel for all neighbor nodes/edges is
    // idempotent, meaning data races are benign.
    static const bool ENABLE_IDEMPOTENCE    = false;

    // Maximum number of vertex associative values of VertexId type during communication,
    // only used to transmit predecessors
    static const int MAX_NUM_VERTEX_ASSOCIATES = MARK_PATHS ? 1:0;

    // Maximum number of vertex associative values of Value type during communication,
    // only used to transmit distance
    static const int MAX_NUM_VALUE_ASSOCIATES = 1;

    // The DataSlice struct stores per-node or per-edge arrays and
    // global variables (if any) that are specific to this particular
    // algorithm. Here, we store the depth value and predecessor node
    // ID for each node.
    // Array1D is a data structure we build in Gunrock that used for
    // efficient 1D array operation and GPU-CPU data movement.
    // There are two additional device 1D arrays defined in BaseDataSlice,
    // they are:
    //
    // preds: that contains predecessor node ID for traversal-based primitives.
    // original_vertex: used for multi-GPU, where GPU i maps the original
    // vertex IDs to [0..|Vi|-1] and stores the original vertex IDs in this array.
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        //util::Array1D<SizeT, VertexId> labels     ; // node labels from source node, defined by BaseDataSlice
        util::Array1D<SizeT, Value   > distances  ; // Used for source distance
        util::Array1D<SizeT, Value   > weights    ; // Used for storing edge weights

        DataSlice() : BaseDataSlice()
        {
            // Name setup
        }

        viatual ~DataSlice()
        {
            Release(); // realse allocated memory
        }

        // Routine to release allocated memory
        cudaError_t Release()
        {
            cudaError_t retval = cudaSuccess;
            if (retval = util::SetDevice(this->gpu_idx)) return retval;
            if (retval = BaseDataSlice::Release()) return retval;
            if (retval = distances     .Release()) return retval;
            if (retval = weights       .Release()) return retval;
            return retval;
        }

        // The Init function initializes a data slice struct with a CSR
        // graph that's stored on the CPU. It also initializes the
        // algorithm-specific data, here labels, and preds.
        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            Csr<VertexId, SizeT, Value> *graph,
            ...)
        {
            cudaError_t retval = cudaSuccess;
            // Init BaseDataSlice
            if (retval = BaseDataSlice::Init(...)) return retval;

            // Allocate device memory
            if (retval = distances   .Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = weights     .Allocate(graph->edges, util::DEVICE)) return retval;
            if (retval = this->labels.Allocate(graph->nodes, util::DEVICE)) return retval;
            // Allocate and move edge weights
            weights.SetPointer(graph->edge_values, graph->edges, util::HOST);
            if (retval = weights.Move(util::HOST, util::DEVICE)) return retval;
            if (MARK_PATHS) {
                if (retval = this->preds.Allocate(graph->nodes, util::DEVICE)) return retval;
            }

            if (num_gpus >1)
            {
                // setup value__associate_orgs to point to distances, for communication
                this->value__associate_orgs[0] = distances.GetPointer(util::DEVICE);
                if (retval = this->value__associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
                if (MARK_PATHS)
                {
                    // setup vertex_associate_orgs to point to preds, for communication
                    this->vertex_associate_orgs[0] = this->preds.GetPointer(util::DEVICE);
                    if (retval = this->vertex_associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
                }
            }

            return retval;
        }

        // Reset data before each enact call
        cudaError_t Reset(
            FrontierType frontier_type,
            GraphSLice<VertexId, SizeT, Value> *graph_slice,
            ....)
        {
            cudaError_t retval = cudaSuccess;
            // Reset control status
            // (Re)allocate memory space for frontier queues and scanned_edges
            
            // If MARK_PATHS, set all predecessor node IDs to self ID
            if (MARK_PATHS)
                util::MemsetIdxKernel<<<128, 128>>>(
                    this->preds.GetPointer(util::DEVICE), nodes);

            // Set all distance values to maxvalue
            util::MemsetKernel<<<128, 128>>>(
                this->distances   .GetPointer(util::DEVICE),
                util::MaxValue<Value>(),
                nodes);

            // Set all labels values to maxvalue
            util::MemsetKernel<<<128, 128>>>(
                this -> labels .GetPointer(util::DEVICE),
                util::InvalidValue<VertexId>(),
                nodes);
            
            return retval;
        }
    };

    util::Array1D<SizeT, DataSlice> *data_slices;

    // Constructor
    SSSPProblem() : BaseProblem(
        false, // use_double_buffer
        false, // enable_backward
        false, // keep_order
        true,  // keep_node_num
        false, // skip_makeout_selection
        true)  // unified_receive
    {
        data_slices = NULL;
    }

    /**
     * @brief SSSPProblem default destructor
     */
    virtual ~SSSPProblem()
    {
        Release(); // release allocated memory
    }

    // Routine to release allocated memory
    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices==NULL) return retval;
        // Release allocated memory on each GPU
        for (int i = 0; i < this->num_gpus; ++i)
        {
            if (retval = util::SetDevice(this->gpu_idx[i])) return retval;
            if (retval = data_slices[i].Release()) return retval;
        }
        delete[] data_slices;data_slices=NULL;

        // Release allocated by parent class
        if (retval = BaseProblem::Release()) return retval;
        return retval;
    }

    // "Extract" copies distances and predecessors back to the CPU.
    cudaError_t Extract(VertexId *h_distances, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;

        if (this->num_gpus == 1) 
        {
            if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

            // move distances from GPU to CPU
            data_slices[0]->distances.SetPointer(h_distances);
            if (retval = data_slices[0]->distances.Move(util::DEVICE, util::HOST)) return retval;

            if (MARK_PATHS)
            {
                // move predecessors marker from GPU to CPU
                data_slices[0]->preds.SetPointer(h_preds);
                if (retval = data_slices[0]->preds.Move(util::DEVICE, util::HOST)) return retval;
            }
        } else {
            // tempary arrays to store pointers for each GPU's results
            Value    **temp_distances = new Value   *[this -> num_gpus];
            VertexId **temp_preds     = new VertexId*[this -> num_gpus];

            // Move each GPU's results to CPU
            for (int gpu = 0; gpu < this -> num_gpus; gpu++)
            {
                if (retval = util::SetDevice( this -> gpu_idx[gpu])) return;

                // move distances from GPU to CPU
                if (retval = data_slices[gpu] -> distances.Move(util::DEVICE, util::HOST)) return retval;
                temp_distances[gpu] = data_slices[gpu] -> distances.GetPointer(util::HOST);

                if (MARK_PATHS)
                {
                    // move predecessors marker from GPU to CPU
                    if (retval = data_slices[gpu]->preds.Move(util::DEVICE, util::HOST)) return retval;
                    temp_preds[gpu] = data_slice[gpu] -> preds.GetPointer(util::HOST);
                }
            }

            // Combine data from multiple GPUs
            for (VertexId v = 0; v < this -> nodes; v++)
            {
                int      gpu = this -> partition_table [0][v]; // get the host GPU
                VertexId v_  = this -> convertion_table[0][v]; // get the converted vertex Id on host GPU
                h_distances[v] = temp_distances[gpu][v_];
                if (MARK_PATHS)
                    h_preds[v] = temp_preds[gpu][v_];
            }

            // Cleanup
            delete[] temp_distances; temp_distances = NULL;
            delete[] temp_preds    ; temp_preds     = NULL;
        }
        return retval;
    } 

    cudaError_t Init(
            Csr<VertexId, SizeT, Value> *graph,
            int num_gpus,
            ...) 
    {
        cudaError_t retval = cudaSuccess;
        // Init BaseProblem, this will partition the graph also
        if (retval = BaseProblem::Init(graph, num_gpus, ...)) return retval;

        // Create pre-GPU dataslice
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];
        for (int gpu = 0; gpu < this->num_gpus; ++gpu)
        {
            if (retval = util::SetDevice(this -> gpu_idx[gpu])) return retval;
            // allocate data_slices[gpu] on CPU and GPU
            if (retval = data_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
            // init data_slices[gpu] based on sub_graph[gpu]
            if (retval = data_slice[gpu]->Init(sub_graph[gpu], ...)) return retval; 
        }
    }

    // The Reset function primes the graph data structure to an
    // untraversed state.
    cudaError_t Reset(
        VertexId src,
        FrontierType frontier_type,
        ...)
    {
        cudaError_t retval = cudaSuccess;

        // reset data_slice on each gpu
        for  (int gpu = 0; gpu < this->num_gpus; ++gpu)
        {
            if (retval = util::SetDevice(this -> gpu_idx[gpu])) return retval;
            if (retval = data_slice[gpu]->Reset(frontier_type, ...)) return retval; 
        }

        int      gpu  = 0;   // which GPU host src
        VertexId tsrc = src; // what is the vertex Id of src on its host GPU
        if (this -> num_gpus > 1)
        {
            gpu = this->partition_tables [0][src];
            tsrc= this->convertion_tables[0][src];
        }

        if (retval = util::SetDevice(this -> gpu_idx[gpu])) return retval;
        // Init distance (and pred) for src on its host GPU
        util::MemsetKernel<<<1,1>>>(data_slice[gpu] -> distances.GetPointer(util::DEVICE) + tsrc, 0, 1);
        if (MARK_PATHS)
            util::MemsetKernel<<<1,1>>>(data_slice[gpu] -> preds.GetPointer(util::DEVICE) + tsrc, 0, 1);

        // Put src into initial frontier
        util::MemsetKernel<<<1,1>>>(data_slice[gpu] -> frontir_queues[0].keys[0].GetPointer(util::DEVICE), tsrc, 1);
    }
};
