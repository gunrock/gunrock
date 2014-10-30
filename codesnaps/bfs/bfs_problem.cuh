// Problem struct is inherited from ProblemBase struct, which stores graph
// topology data in CSR format. Each problem struct stores per-node or per-edge
// arrays and global variables (if any). It provides Init and Reset method as
// well as Extract method to get results from GPU to CPU.

template<
         typename VertexId,
         typename SizeT,
         typename Value,
         bool _MARK_PREDECESSORS, // Whether to mark predecessor ID when advance
         bool _ENABLE_IDEMPOTENCE, // Whether to enable idempotence when advance
         bool _USE_DOUBLE_BUFFER>
         struct BFSProblem : public ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER>
{

    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE;

    struct DataSlice
    {
        VertexId *d_labels; // Distance from source node labels
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

    cudaError_t Init(
            const Csr<VertexId, Value, SizeT> &graph)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Init(graph))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_labels, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_preds, nodes))) break;
        return retval;  
    }

    cudaError_t Reset(
            const Csr<VertexId, Value, SizeT> &graph, VertexId src)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Reset(graph))) break;
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_labels, -1, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_preds, -2, nodes);
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_labels+src, 0, 1)));  
        return retval;  
    }
};
