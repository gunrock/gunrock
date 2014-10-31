class SalsaEnactor : public EnactorBase {

    // Constructor, Destructor, and Setup functions are ignored

    // This is a user defined function to swap current and next rank pointers
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

            //swap rank_curr and rank_next
            util::MemsetCopyVectorKernel<<<128, 128>>>(rank_curr, rank_next, nodes); 
            util::MemsetKernel<<<128, 128>>>(rank_next, (Value)0.0, nodes);
        }

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

                         //Define SALSA Functors
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

                         //Load Setup function
                         cudaError_t retval = cudaSuccess;
                         if (retval = EnactorBase::Setup(problem)) break;

                        //Define graph topology data pointer (g_slice) and
                        //problem specific data pointer (d_slice)
                         typename SALSAProblem::GraphSlice *g_slice = problem->d_graph_slices;
                         typename SALSAProblem::DataSlice *d_slice = problem->d_data_slices;

                         //Prepare to do advance for each node. The purpose is to get information
                         //about predecessor node ID for each edge in both graphs.
                         SizeT queue_length = g_slice->nodes;
                         int selector = 0;
                         {
                             // Fill the frontier with all node IDs
                             util::MemsetIdxKernel<<<BLOCK, THREAD>>>(g_slice->ping_pong_working_queue[selector], g_slice->nodes);

                             // Advance use HForwardFunctor to set predecessor
                             // node for each edge in original graph
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
                            // Advance use AForwardFunctor to set predecessor
                            // node for each edge in reverse graph
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

                         //Update hub rank and authority ranks using two Advance operators until reach
                         //maximum iteration number
                         int iteration = 0;
                         while (true) {
                             util::MemsetIdxKernel<<<BLOCK, THREAD>>>(g_slice->ping_pong_working_queue[selector], g_slice->edges);
                             SizeT queue_length = g_slice->edges;

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

    // Kernel entry point to be loaded in the driver code. Details please refer
    // to annotations for bfs_enactor.cuh
    template <typename SALSAProblem>
        cudaError_t Enact(
                CudaContext &context,
                SALSAProblem *problem,
                typename SALSAProblem::SizeT max_iteration)
        {
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
