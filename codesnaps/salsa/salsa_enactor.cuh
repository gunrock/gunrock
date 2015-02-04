// The enactor defines how a graph primitive runs. It calls traversal
// (advance and filter operators) and computation (functors).

class SalsaEnactor : public EnactorBase {

    // For SALSA, Constructor, Destructor, and Setup functions are ignored

    // This user-defined function swaps current and next rank pointers
    template <typename ProblemData>
        void SwapRank(ProblemData *problem, int is_hub, int nodes)
        {
            typedef typename ProblemData::Value Value;
            Value *rank_curr;
            Value *rank_next;
            if (is_hub) {
                rank_curr = problem->data_slices[0]->d_hrank_curr;
                rank_next = problem->data_slices[0]->d_hrank_next;
            } else {
                rank_curr = problem->data_slices[0]->d_hrank_curr;
                rank_next = problem->data_slices[0]->d_hrank_next;
            }

            // copy next to curr and reset next
            util::MemsetCopyVectorKernel<<<128, 128>>>(rank_curr, rank_next, nodes);
            util::MemsetKernel<<<128, 128>>>(rank_next, (Value)0.0, nodes);
        }

    // This enactor defines the SALSA high-level algorithm.
    template<
        typename AdvancePolicy,
        typename FilterPolicy,
        typename SALSAProblem>
                     cudaError_t EnactSALSA(
                             CudaContext &context,
                             SALSAProblem *problem,
                             int max_iteration) {

                         typedef typename SALSAProblem::VertexId VertexId;
                         typedef typename SALSAProblem::SizeT SizeT;
                         typedef typename SALSAProblem::Value Value;

                         // Define SALSA functors.
                         typedef HFORWARDFunctor<
                             VertexId,
                             SizeT,
                             Value,
                             SALSAProblem> HForwardFunctor;

                         typedef AFORWARDFunctor<
                             VertexId,
                             SizeT,
                             Value,
                             SALSAProblem> AForwardFunctor;

                         typedef HBACKWARDFunctor<
                             VertexId,
                             SizeT,
                             Value,
                             SALSAProblem> HBackwardFunctor;

                         typedef ABACKWARDFunctor<
                             VertexId,
                             SizeT,
                             Value,
                             SALSAProblem> ABackwardFunctor;

                         // Load the Setup function.
                         cudaError_t retval = cudaSuccess;
                         if (retval = EnactorBase::Setup(problem)) break;

                        // Define the graph topology data pointer (g_slice) and
                        // the problem-specific data pointer (d_slice).
                         typename SALSAProblem::GraphSlice *g_slice = problem->d_graph_slices;
                         typename SALSAProblem::DataSlice *d_slice = problem->d_data_slices;

                         // Now let's do some computation.
                         SizeT queue_length = g_slice->nodes;
                         int selector = 0;
                         {
                             // First we'll do some initialization
                             // code that runs just once. Start by
                             // initializing the frontier with all
                             // node IDs.
                             util::MemsetIdxKernel<<<BLOCK, THREAD>>>(g_slice->ping_pong_working_queue[selector], g_slice->nodes);

                             // Set predecessor nodes for each edge in
                             // the original graph.
                             gunrock::oprtr::advance::Kernel
                                 <AdvancePolicy, SALSAProblem, HForwardFunctor>
                                 <<<advance_grid_size, AdvancePolicy::THREADS>>>(
                                         queue_length,
                                         g_slice->ping_pong_working_queue[selector],
                                         g_slice->ping_pong_working_queue[selector^1],
                                         g_slice->d_row_offsets,
                                         g_slice->d_column_indices, //advance on original graph
                                         d_slice
                                         context,
                                         gunrock::oprtr::advance::V2E);
                            // And set the predecessor nodes for each
                            // edge in the reverse graph.
                             gunrock::oprtr::advance::Kernel
                                 <AdvancePolicy, SALSAProblem, AForwardFunctor>
                                 <<<advance_grid_size, AdvancePolicy::THREADS>>>(
                                         queue_length,
                                         g_slice->ping_pong_working_queue[selector],
                                         g_slice->ping_pong_working_queue[selector^1],
                                         g_slice->d_column_offsets,
                                         g_slice->d_row_indices, //advance on reverse graph
                                         d_slice
                                         context,
                                         gunrock::oprtr::advance::V2E);
                         }

                         // Now we iterate between two Advance
                         // operators, which update (1) the hub rank
                         // and (2) the authority rank. We loop until
                         // we've reached the maximum iteration count.
                         int iteration = 0;
                         while (true) {
                             util::MemsetIdxKernel<<<BLOCK, THREAD>>>(g_slice->ping_pong_working_queue[selector], g_slice->edges);
                             SizeT queue_length = g_slice->edges;

                             // This Advance operator updates the hub rank ...
                             gunrock::oprtr::advance::Kernel
                                 <AdvancePolicy, SALSAProblem, ABackwardFunctor>
                                 <<<advance_grid_size, AdvancePolicy::THREADS>>>(
                                         queue_length,
                                         g_slice->ping_pong_working_queue[selector],
                                         g_slice->ping_pong_working_queue[selector^1],
                                         g_slice->d_column_offsets,
                                         g_slice->d_row_indices, //advance backward on reverse graph
                                         d_slice
                                         context,
                                         gunrock::oprtr::advance::E2V);

                             SwapRank<SALSAProblem>(problem, 0, g_slice->nodes);

                             // and here, the authority rank.
                             gunrock::oprtr::advance::Kernel
                                 <AdvancePolicy, SALSAProblem, ABackwardFunctor>
                                 <<<advance_grid_size, AdvancePolicy::THREADS>>>(
                                         queue_length,
                                         g_slice->ping_pong_working_queue[selector],
                                         g_slice->ping_pong_working_queue[selector^1],
                                         g_slice->d_row_offsets,
                                         g_slice->d_column_indices, //advance backward on original graph
                                         d_slice
                                         context,
                                         gunrock::oprtr::advance::E2V);

                             SwapRank<SALSAProblem>(problem, 0, g_slice->nodes);

                             iteration++;
                             if (iteration >= max_iteration) break;
                         }

                         return retval;
                     }

    // The entry point in the driver code to SALSA is this Enact call.
    template <typename SALSAProblem>
        cudaError_t Enact(
                CudaContext &context,
                SALSAProblem *problem,
                typename SALSAProblem::SizeT max_iteration)
        {
            // Gunrock provides recommended settings here for kernel
            // parameters, but they can be changed by end-users.
            typedef gunrock::oprtr::filter::KernelPolicy<
                SALSAProblem,
                300, //CUDA_ARCH
                8, //MIN_CTA_OCCUPANCY
                8> //LOG_THREAD_NUM
                FilterKernelPolicy;

            typedef gunrock::oprtr::advance::KernelPolicy<
                SALSAProblem,
                300,
                8, //MIN_CTA_OCCUPANCY
                10, //LOG_THREAD_NUM
                32*128> //THRESHOLD_TO_SWITCH_ADVANCE_MODE
                AdvanceKernelPolicy;

            return EnactSALSA<AdvanceKernelPolicy, FilterKernelPolicy, SALSAProblem>(
                    context, problem, max_iteration);
        }
};
