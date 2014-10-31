// This data structure (the "Problem" struct) stores the graph
// topology in CSR format and the frontier. All Problem structs
// inherit from the ProblemBase struct. Algorithm-specific data is
// stored in a "DataSlice".

template<
typename VertexId,
typename SizeT,
typename Value>
struct SALSAProblem : public ProblemBase
{

    static const bool MARK_PREDECESSORS     = true; // for SALSA algorithm, we need to track predecessors in Advance
    static const bool ENABLE_IDEMPOTENCE    = false; // In SALSA, data races during Advance are not allowed.

    struct DataSlice
    {
        Value *d_hrank_curr; // hub rank score for current iteration
        Value *d_arank_curr; // authority rank score for current iteration
        Value *d_hrank_next; // hub rank score for next iteration
        Value *d_arank_next; // authority rank score for next iteration
        VertexId *d_in_degrees; // in degrees for each node
        VertexId *d_out_degrees; // out degrees for each node
        VertexId *d_hub_predecessors; // hub graph predecessors (original graph)
        VertexId *d_auth_predecessors; // authority graph predecessors (reverse graph)
    };

    SizeT nodes; // node number of the graph
    SizeT edges; // edge number of the graph
    SizeT out_nodes; // number of nodes that have outgoing edges
    SizeT in_nodes; // number of nodes that have incoming edges
    DataSlice *d_data_slices;

    // The constructor and destructor are ignored here.

    // "Extract" copies final hub rank scores and authority rank
    // scores back to the CPU.

    cudaError_t Extract(VertexId *h_hrank, VertexId *h_arank)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_hrank_curr, h_hrank, nodes))) break;
        if (retval = util::GRError(CopyGPU2CPU(data_slices[0]->d_arank_curr, h_arank, nodes))) break;
        return retval;
    }

    // SALSA is an algorithm for bipartite graphs. So the Init
    // function takes two CSR graphs---the hub (original) graph and
    // the auth (reverse) graph stored on the CPU, and uses them to
    // initialize the graph topology data and the SALSA-specific
    // DataSlice on the GPU.
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
        return retval;
    }

    // The Reset function primes the graph data structure to an
    // initial state, which includes ...
    cudaError_t Reset(const Csr<VertexId, Value, SizeT> &graph)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::GRError(ProblemBase::Reset(graph))) break;

        // ... initializing the hub and authority rank scores ...
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_hrank_curr, (Value)1.0/out_nodes, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_arank_curr, (Value)1.0/in_nodes, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_hrank_next, 0, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_arank_next, 0, nodes);

        // ... accurate in and out degrees for each node ...
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_out_degrees, 0, nodes);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_in_degrees, 0, nodes);
        util::MemsetMadVectorKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_out_degrees, BaseProblem::graph_slices[gpu]->d_row_offsets, &BaseProblem::graph_slices[gpu]->d_row_offsets[1], -1, nodes);
        util::MemsetMadVectorKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_in_degrees, BaseProblem::graph_slices[gpu]->d_column_offsets, &BaseProblem::graph_slices[gpu]->d_column_offsets[1], -1, nodes);

        // ... and initializing predecessors with an INVALID_PREDECESSOR_ID.
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_hub_predecessors, INVALID_PREDECESSOR_ID, edges);
        util::MemsetKernel<<<BLOCK, THREAD>>>(data_slices[0]->d_auth_predecessors, INVALID_PREDECESSOR_ID, edges);

        return retval;
    }
};
