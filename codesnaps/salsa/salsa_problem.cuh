template<
typename VertexId,
         typename SizeT,
         typename Value>
         struct BFSProblem : public ProblemBase
{

    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;

    struct DataSlice
    {
        Value *d_hrank_curr;
        Value *d_arank_curr;
        Value *d_hrank_next;
        Value *d_arank_next;
        VertexId *d_in_degrees;
        VertexId *d_out_degrees;
        VertexId *d_hub_predecessors;
        VertexId *d_auth_predecessors;
        SizeT *d_labels;
    };

    SizeT nodes;
    SizeT edges;
    SizeT out_nodes;
    SizeT in_nodes;
    DataSlice *d_data_slices;

    //Constructor, Destructor ignored here

    cudaError_t Extract(VertexId *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_hrank_curr, h_hrank, nodes))) break;
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_arank_curr, h_arank, nodes))) break;
        return retval;
    }

    cudaError_t Init(
            const Csr<VertexId, Value, SizeT> &hub_graph,
            const Csr<VertexId, Value, SizeT> &auth_graph)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Init(hub_graph, auth_graph))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_hrank_curr, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_arank_curr, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_hrank_next, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_arank_next, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_in_degrees, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_out_degrees, nodes))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_hub_predecessors, edges))) break;
        if (retval = util::GRError(GPUMalloc(data_slices[0]->d_auth_predecessors, edges))) break;
        data_slices[0]->d_labels = NULL;
        return retval;  
    }

    cudaError_t Reset(
            const Csr<VertexId, Value, SizeT> &graph, VertexId src)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Reset(graph))) break;
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_hrank_curr, (Value)1.0/out_nodes, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_arank_curr, (Value)1.0/in_nodes, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_hrank_next, 0, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_arank_next, 0, nodes);

        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_out_degrees, 0, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_in_degrees, 0, nodes);
        util::MemsetMadVectorKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_out_degrees, BaseProblem::graph_slices[gpu]->d_row_offsets, &BaseProblem::graph_slices[gpu]->d_row_offsets[1], -1, nodes);
        util::MemsetMadVectorKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_in_degrees, BaseProblem::graph_slices[gpu]->d_column_offsets, &BaseProblem::graph_slices[gpu]->d_column_offsets[1], -1, nodes);

        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_hub_predecessors, -1, edges);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_auth_predecessors, -1, edges);

        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_labels+src, 0, 1)));  
        return retval;  
    }
};
