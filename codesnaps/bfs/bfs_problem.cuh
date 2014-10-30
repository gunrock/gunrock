template<
typename VertexId,
         typename SizeT,
         typename Value,
         bool _MARK_PREDECESSORS,
         bool _ENABLE_IDEMPOTENCE,
         bool _USE_DOUBLE_BUFFER>
         struct BFSProblem : public ProblemBase<VertexId, SizeT, _USE_DOUBLE_BUFFER>
{

    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE;

    struct DataSlice
    {
        VertexId *d_labels;
        VertexId *d_preds;
    };

    SizeT nodes;
    SizeT edges;
    DataSlice *d_data_slices;

    //Constructor, Destructor ignored here

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
