// This data structure (the "Problem" struct) stores the graph
// topology in CSR format and the frontier. All Problem structs
// inherit from the ProblemBase struct. Algorithm-specific data is
// stored in a "DataSlice".

template<
         typename VertexId,
         typename SizeT,
         typename Value,
         bool _MARK_PATHS> // Whether to mark predecessor ID when advance
         struct SSSPProblem : public ProblemBase<VertexId, SizeT, Value>
{

    // MARK_PREDECESSORS sets the predecessor node ID during a
    // traversal for each node in the new frontier.
    static const bool MARK_PREDECESSORS     = true;
    // ENABLE_IDEMPOTENCE is an optimization when the operation
    // performed in parallel for all neighbor nodes/edges is
    // idempotent, meaning data races are benign.
    static const bool ENABLE_IDEMPOTENCE    = false;
    // TODO: Needs YC to explain
    static const int MAX_NUM_VERTEX_ASSOCIATES = MARK_PATHS ? 1:0;
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
        util::Array1D<SizeT, Value> labels; // node labels from source node.

        // The Init function initializes a data slice struct with a CSR
        // graph that's stored on the CPU. It also initializes the
        // algorithm-specific data, here labels, and preds.
        cudaError_t Init(
                const Csr<VertexId, Value, SizeT> &graph)
        {
            cudaError_t retval = cudaSuccess;
            if (retval = this->labels.Allocate(graph->nodes, util::DEVICE)) break;

            if (MARK_PATHS) {
                if (retval = this->preds.Allocate(graph->nodes, util::DEVICE)) return retval;
            }

            return retval;
        }

        cudaError_t Reset(
            FrontierType frontier_type,
            GraphSLice<VertexId, SizeT, Value> *graph_slice)
        {
            cudaError_t retval = cudaSuccess;
            // TODO: YC, can I just suppose we have such a wrapper for all the mGPU related initialization?
            if (retval = util::GRError(ResetMultiGPUArrays(graph))) break;
            // Set all labels values to maxvalue. Set the
            // source node's label value to 0.
            // If MARK_PATHS, set all predecessor node IDs to self ID
            if (MARK_PATHS) {
                util::MemsetIdxKernel<<<BLOCK, THREAD>>>(this->preds.GetPointer(util::DEVICE), nodes);
            }
            util::MemsetKernel<<<BLOCK, THREAD>>>(this->labels.GetPointer(util::DEVICE), util::MaxValue<Value>(), nodes);
            if (retval = util::GRError(CopyGPU2CPU(this->labels.GetPointer(util::DEVICE)+src, 0, 1))) return retval;

            // Put the source node ID into the initial frontier.
            if (retval = util::GRError(CopyGPU2CPU(g_slices[0]->ping_pong_working_queue, src, 1))) return retval;
            return retval;
        }
    };

    SizeT nodes;
    SizeT edges;
    DataSlice *data_slices;

    // The constructor and destructor are ignored here.

    // "Extract" copies distances and predecessors back to the CPU.
    cudaError_t Extract(VertexId *h_distances, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;

        if (this->num_gpus == 1) {
            if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
            data_slices[0]->labels.SetPointer(h_distances);
            if (retval = data_slices[0]->labels.Move(util::DEVICE, util::HOST)) return retval;
            if (MARK_PATHS)
            {
                data_slices[0]->preds.SetPointer(h_preds);
                if (retval = data_slices[0]->preds.Move(util::DEVICE, util::HOST)) return retval;
            }
        } else {
            //TODO: YC, is it ok to just assume we call such a function?
            retval = ExtractMultiGPU(h_distance, h_preds);
        }
        return retval;
    } 

    cudaError_t Init(
            int num_gpus,
            const Csr<VertexId, Value, SizeT> &graph) {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseProblem::Init(num_gpus, graph)) return retval;
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];
        for (int gpu = 0; gpu < this->num_gpus; ++gpu)
        {
            // TODO: YC, is this part correct regarding mGPU?
            InitDataSlice(gpu, data_slices);
            if (retval = data_slice[gpu]->Init(graph)) return retval; 
        }
        if (num_gpus > 1)
        {
            this->value_associate_orgs[0] = labels.GetPointer(util::DEVICE);
            if (MARK_PATHS)
            {
                this->vertex_associate_orgs[0] = this->preds.GetPointer(util::DEVICE);
                if (retval = this->vertex_associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
            }
            if (retval = this->value_associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
        }
    }

    // The Reset function primes the graph data structure to an
    // untraversed state.
    cudaError_t Reset(
            FrontierType frontier_type,
            const Csr<VertexId, Value, SizeT> &graph, VertexId src)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = util::GRError(ProblemBase::Reset(graph))) break;
        for  (int gpu = 0; gpu < this->num_gpus; ++gpu)
        {
           if (retval = data_slice[gpu]->Reset(frontier_type, graph)) return retval; 
        }
    }
};
