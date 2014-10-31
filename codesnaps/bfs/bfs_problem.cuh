// This data structure (the "Problem" struct) stores the graph
// topology in CSR format and the frontier. All Problem structs
// inherit from the ProblemBase struct. Algorithm-specific data is
// stored in a "DataSlice".

template<
         typename VertexId,
         typename SizeT,
         typename Value,
         bool _MARK_PREDECESSORS, // Whether to mark predecessor ID when advance
         bool _ENABLE_IDEMPOTENCE, // Whether to enable idempotence when advance
         bool _USE_DOUBLE_BUFFER>
         struct BFSProblem : public ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER>
{

    // MARK_PREDECESSORS sets the predecessor node ID during a
    // traversal for each node in the new frontier.
    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    // ENABLE_IDEMPOTENCE is an optimization when the operation
    // performed in parallel for all neighbor nodes/edges is
    // idempotent, meaning data races are benign.
    static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE;

    // The DataSlice struct stores per-node or per-edge arrays and
    // global variables (if any) that are specific to this particular
    // algorithm. Here, we store the depth value and predecessor node
    // ID for each node.
    struct DataSlice
    {
        VertexId *d_labels; // BFS depth value
        VertexId *d_preds; // Predecessor IDs
    };

    SizeT nodes;
    SizeT edges;
    DataSlice *d_data_slices;

    // The constructor and destructor are ignored here.

    // "Extract" copies labels and predecessors back to the CPU.
    cudaError_t Extract(VertexId *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_labels, h_labels, nodes))) break;
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_preds, h_preds, nodes))) break;
        return retval;
    }

    // The Init function initializes this Problem struct with a CSR
    // graph that's stored on the CPU. It also initializes the
    // algorithm-specific data, here depth and predecessor.
    cudaError_t Init(
            const Csr<VertexId, Value, SizeT> &graph)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Init(graph))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_labels, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_preds, nodes))) break;
        return retval;
    }

    // The Reset function primes the graph data structure to an
    // untraversed state.
    cudaError_t Reset(
            const Csr<VertexId, Value, SizeT> &graph, VertexId src)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Reset(graph))) break;
        // Set all depth and predecessor values to invalid. Set the
        // source node's depth value to 0.
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_labels, INVALID_NODE_VALUE, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_preds, INVALID_PREDECESSOR_ID, nodes);
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_labels+src, 0, 1)));
        // Put the source node ID into the initial frontier.
        if (retval = util::GRError(CopyGPU2CPU(g_slices[0]->ping_pong_working_queue, src, 1)));
        return retval;
    }
};
