// Problem struct is inherited from ProblemBase struct, which stores graph
// topology data in CSR format and frontier buffers.

template<
         typename VertexId,
         typename SizeT,
         typename Value,
         bool _MARK_PREDECESSORS, // Whether to mark predecessor ID when advance
         bool _ENABLE_IDEMPOTENCE, // Whether to enable idempotence when advance
         bool _USE_DOUBLE_BUFFER>
         struct BFSProblem : public ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER>
{

    // MARK_PREDECESSOR would be true when algorithm needs to know the predecessor node IDs
    // for a newly formed frontier during a certain step.
    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    // ENABLE_IDEMPOTENCE would be true when the operation performed in parallel
    // for all neighbor nodes/edges is idempotent, meaning data race is benign.
    static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE;

    // DataSlice sturct stores per-node or per-edge arrays and global variables (if any).
    struct DataSlice
    {
        VertexId *d_labels; // BFS depth value
        VertexId *d_preds; // Predecessor IDs
    };

    SizeT nodes;
    SizeT edges;
    DataSlice *d_data_slices;

    //Constructor, Destructor ignored here

    // Extract labels and predecessors back to CPU
    cudaError_t Extract(VertexId *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_labels, h_labels, nodes))) break;
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_preds, h_preds, nodes))) break;
        return retval;
    }

    // Init function takes a CSR graph stored on CPU, and initialize graph
    // topology data (by loading ProblemBase::Init(graph)) and DataSlice on
    // GPU.
    cudaError_t Init(
            const Csr<VertexId, Value, SizeT> &graph)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Init(graph))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_labels, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_preds, nodes))) break;
        return retval;  
    }

    // Reset function will be loaded before each BFS to reset problem related data.
    cudaError_t Reset(
            const Csr<VertexId, Value, SizeT> &graph, VertexId src)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Reset(graph))) break;
        // Reset depth values and predecessor values to invalid. Only set source node's depth value to 0.
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_labels, INVALID_NODE_VALUE, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_preds, INVALID_PREDECESSOR_ID, nodes);
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_labels+src, 0, 1)));
        // Put source node ID into the initial frontier
        if (retval = util::GRError(CopyGPU2CPU(g_slices[0]->ping_pong_working_queue, src, 1)));
        return retval;  
    }
};
