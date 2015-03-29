// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * andersen_enactor.cuh
 *
 * @brief Andersen Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/memset_kernel.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/andersen/andersen_problem.cuh>
#include <gunrock/app/andersen/andersen_functor.cuh>


namespace gunrock {
namespace app {
namespace andersen {

    template <typename Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK> class Enactor;

template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct AndersenIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    false, true, true, true, false>
{
public:
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    typedef RuleKernelFunctor            <VertexId, SizeT, Value, Problem> 
            RuleKernelFunctor            ;
    typedef RuleKernelFunctor2           <VertexId, SizeT, Value, Problem> 
            RuleKernelFunctor2           ;
    typedef MakeInverstFunctor           <VertexId, SizeT, Value, Problem>
            MakeInverstFunctor           ;
    typedef MakeInverstFunctor2          <VertexId, SizeT, Value, Problem>
            MakeInverstFunctor2          ;
    typedef MakeHashFunctor              <VertexId, SizeT, Value, Problem>
            MakeHashFunctor              ;

    static void GraphSlice_Union(
        GraphSlice               *a,
        GraphSlice               *b,
        GraphSlice               *c,
        EnactorStats             *enactor_stats,
        cudaStream_t              stream)
    {
        c->edges = a->edges + b->edges;
        if (enactor_stats->retval = c->column_indices.EnsureSize(c->edges)) return;
        //printf(" nodes = %d, a->row_offsets=%d,%p, a->column_indices=%d,%p, b->row_offsets=%d,%p, b->column_indices=%d,%p\n", a->nodes, 
        //    a->row_offsets   .GetSize(), a->row_offsets   .GetPointer(util::DEVICE),
        //    a->column_indices.GetSize(), a->column_indices.GetPointer(util::DEVICE),
        //    b->row_offsets   .GetSize(), b->row_offsets   .GetPointer(util::DEVICE),
        //    b->column_indices.GetSize(), b->column_indices.GetPointer(util::DEVICE));fflush(stdout);
        UnionGraphs <VertexId, SizeT>
           <<<128, 128, 0, stream>>> (
           a -> nodes,
           a -> row_offsets   .GetPointer(util::DEVICE), 
           a -> column_indices.GetPointer(util::DEVICE),
           b -> row_offsets   .GetPointer(util::DEVICE),
           b -> column_indices.GetPointer(util::DEVICE),
           c -> row_offsets   .GetPointer(util::DEVICE),
           c -> column_indices.GetPointer(util::DEVICE));
    }

    static void FullQueue_Gather(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        if (enactor_stats->iteration>0)
        {
            printf("GraphUnion : pts_add <- pts + pts_inc\n");fflush(stdout);
            GraphSlice_Union(data_slice->pts_graphslice, data_slice->pts_inc_graphslice, data_slice->pts_add_graphslice, enactor_stats, stream);
            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;

            printf("Assigment  : t <- pts; pts <- pts_add; pts_add <- pts_inc; pts_inc <- t(0)\n");fflush(stdout);
            //util::Array1D<SizeT, SizeT   > t_offsets;//       = data_slice->    pts_graphslice->row_offsets;
            //util::Array1D<SizeT, VertexId> t_indices;//       = data_slice->    pts_graphslice->column_indices;
            data_slice->                   t_offsets       = data_slice->    pts_graphslice->row_offsets;
            data_slice->                   t_indices       = data_slice->    pts_graphslice->column_indices;
            data_slice->    pts_graphslice->row_offsets    = data_slice->pts_add_graphslice->row_offsets;
            data_slice->    pts_graphslice->column_indices = data_slice->pts_add_graphslice->column_indices;
            data_slice->    pts_graphslice->edges          = data_slice->pts_add_graphslice->edges;
            data_slice->pts_add_graphslice->row_offsets    = data_slice->pts_inc_graphslice->row_offsets;
            data_slice->pts_add_graphslice->column_indices = data_slice->pts_inc_graphslice->column_indices;
            data_slice->pts_add_graphslice->edges          = data_slice->pts_inc_graphslice->edges;
            data_slice->pts_inc_graphslice->row_offsets    = data_slice->t_offsets;
            data_slice->pts_inc_graphslice->column_indices = data_slice->t_indices;
            data_slice->pts_inc_graphslice->edges          = 0;

            printf("GraphUnion : copyIad <- copyInv + copyIin\n");fflush(stdout);
            GraphSlice_Union(data_slice->copyInv_graphslice, data_slice->copyIin_graphslice, data_slice->copyIad_graphslice, enactor_stats, stream);
            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;

            printf("Assigment  : t <- copyInv; copyInv <- copyIad; copyIad <- copyIin; copyIin <- t(0)\n");fflush(stdout);
            data_slice->                   t_offsets       = data_slice->copyInv_graphslice->row_offsets;
            data_slice->                   t_indices       = data_slice->copyInv_graphslice->column_indices;
            data_slice->copyInv_graphslice->row_offsets    = data_slice->copyIad_graphslice->row_offsets;
            data_slice->copyInv_graphslice->column_indices = data_slice->copyIad_graphslice->column_indices;
            data_slice->copyInv_graphslice->edges          = data_slice->copyIad_graphslice->edges;
            data_slice->copyIad_graphslice->row_offsets    = data_slice->copyIin_graphslice->row_offsets;
            data_slice->copyIad_graphslice->column_indices = data_slice->copyIin_graphslice->column_indices;
            data_slice->copyIad_graphslice->edges          = data_slice->copyIin_graphslice->edges;
            data_slice->copyIin_graphslice->row_offsets    = data_slice->t_offsets;
            data_slice->copyIin_graphslice->column_indices = data_slice->t_indices;
            data_slice->copyIin_graphslice->edges          = 0;
            data_slice->t_offsets                          = util::Array1D<SizeT, SizeT   >();
            data_slice->t_indices                          = util::Array1D<SizeT, VertexId>();
        }

        printf("Inverting  : ptsIadd <- (pts_add)^-1\n");fflush(stdout);
        GraphSlice* r_graphslice = enactor_stats->iteration==0 ? data_slice->pts_graphslice : data_slice->pts_add_graphslice;
        GraphSlice* t_graphslice = data_slice->ptsIadd_graphslice;
        if (enactor_stats->retval = t_graphslice->column_indices.EnsureSize(r_graphslice->edges)) return;
        if (enactor_stats->retval = frontier_queue->keys[1].EnsureSize(r_graphslice->edges)) return;
        if (enactor_stats->retval = frontier_queue->keys[0].EnsureSize(r_graphslice->nodes)) return;
        if (enactor_stats->retval = scanned_edges->EnsureSize(r_graphslice->nodes + 1)) return;
        util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->t_length.GetPointer(util::DEVICE), 0, r_graphslice->nodes); 
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        //printf("0_1\n");fflush(stdout);
        util::MemsetIdxKernel<<<128, 128, 0, stream>>>(frontier_queue->keys[0].GetPointer(util::DEVICE), r_graphslice->nodes);
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        //printf("0_2\n");fflush(stdout);
        t_graphslice->edges     = r_graphslice->edges;
        data_slice  ->r_offsets = r_graphslice->row_offsets;
        data_slice  ->r_indices = r_graphslice->column_indices;
        data_slice  ->t_offsets = t_graphslice->row_offsets;
        data_slice  ->t_indices = t_graphslice->column_indices;
        if (enactor_stats->retval = data_slice  ->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream)) return;
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        //printf("0_3\n");fflush(stdout);
        frontier_attribute->queue_index = 0;
        frontier_attribute->queue_reset = true;
        frontier_attribute->queue_length= r_graphslice->nodes;
        frontier_attribute->selector    = 0;
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, MakeInverstFunctor> (
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            (VertexId*)NULL,
            data_slice->r_offsets.GetPointer(util::DEVICE),
            data_slice->r_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            r_graphslice->nodes,
            r_graphslice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            true);
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        //printf("0_4\n");fflush(stdout);

        util::MemsetKernel<<<1,1,0,stream>>>(t_graphslice->row_offsets.GetPointer(util::DEVICE), 0, 1);
        Scan<mgpu::MgpuScanTypeInc>(
            (SizeT*)data_slice->t_length.GetPointer(util::DEVICE),
            r_graphslice->nodes,
            (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
            (SizeT*)(t_graphslice->row_offsets.GetPointer(util::DEVICE)+1),
            context[0]);
        if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;
        //printf("0_5\n");fflush(stdout);

        util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->t_length.GetPointer(util::DEVICE), 0, r_graphslice->nodes);
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        frontier_attribute->queue_index = 0;
        frontier_attribute->queue_reset = true;
        frontier_attribute->queue_length= r_graphslice->nodes;
        frontier_attribute->selector    = 0;
        /*gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, MakeInverstFunctor2> (
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            (VertexId*)NULL,
            data_slice->r_offsets.GetPointer(util::DEVICE),
            data_slice->r_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            r_graphslice->nodes,
            r_graphslice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            true);*/
        InverstGraph <VertexId, SizeT, Value>
            <<<128, 128, 0, stream>>> (
            r_graphslice->nodes,
            r_graphslice->row_offsets   .GetPointer(util::DEVICE),
            r_graphslice->column_indices.GetPointer(util::DEVICE),
            t_graphslice->row_offsets   .GetPointer(util::DEVICE),
            data_slice  ->t_length      .GetPointer(util::DEVICE),
            t_graphslice->column_indices.GetPointer(util::DEVICE));
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        //printf("0_6\n");fflush(stdout);
        t_graphslice->edges = r_graphslice->edges;
       
        r_graphslice = NULL;
        t_graphslice = NULL;
        data_slice->r_offsets = util::Array1D<SizeT, SizeT   >();
        data_slice->r_indices = util::Array1D<SizeT, VertexId>();
        data_slice->t_offsets = util::Array1D<SizeT, SizeT   >();
        data_slice->t_indices = util::Array1D<SizeT, VertexId>();
    }

    static void FullQueue_Core(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        GraphSlice*     pts_graphslice = data_slice->    pts_graphslice;
        //GraphSlice*  ptsInv_graphslice = data_slice-> ptsInv_graphslice;
        GraphSlice* ptsIadd_graphslice = data_slice->ptsIadd_graphslice;
        GraphSlice* pts_add_graphslice = data_slice->pts_add_graphslice;
        GraphSlice* pts_inc_graphslice = data_slice->pts_inc_graphslice;
        GraphSlice* copyInv_graphslice = data_slice->copyInv_graphslice;
        GraphSlice* copyIad_graphslice = data_slice->copyIad_graphslice;
        GraphSlice* copyIin_graphslice = data_slice->copyIin_graphslice;
        GraphSlice* loadInv_graphslice = data_slice->loadInv_graphslice;
        GraphSlice*   store_graphslice = data_slice->  store_graphslice;
        GraphSlice*  gepInv_graphslice = data_slice-> gepInv_graphslice;
        GraphSlice*       r_graphslice = NULL;
        GraphSlice*       s_graphslice = NULL;
        GraphSlice*       t_graphslice = NULL;
        SizeT                    nodes = pts_graphslice->nodes;
        SizeT                 max_size = 0;
        //bool               to_continue = false;

        if (enactor_stats->iteration > 0)
        {
            if (pts_add_graphslice->edges == 0 && copyIad_graphslice->edges == 0)
            {
                data_slice -> to_continue = false;
                return ;
            } else data_slice -> to_continue = true;
        }

        max_size = max(max_size, enactor_stats->iteration==0? copyInv_graphslice->edges : copyIad_graphslice->edges);
        max_size = max(max_size, copyInv_graphslice->edges);
        max_size = max(max_size, loadInv_graphslice->edges);
        max_size = max(max_size, ptsIadd_graphslice->edges);
        max_size = max(max_size,  gepInv_graphslice->edges);
        if (enactor_stats->retval = frontier_queue->keys[frontier_attribute->selector^1].EnsureSize(max_size)) return;
        if (enactor_stats->retval = scanned_edges->EnsureSize(max_size+1)) return;
        if (enactor_stats->retval = frontier_queue->keys[frontier_attribute->selector  ].EnsureSize(nodes)) return;
        //util::MemsetKernel   <<<128, 128, 0, stream>>>(pts_inc_graphslice->row_offsets.GetPointer(util::DEVICE), 0, nodes);
        //util::MemsetKernel   <<<128, 128, 0, stream>>>(copyIin_graphslice->row_offsets.GetPointer(util::DEVICE), 0, nodes);
        util::MemsetIdxKernel<<<128, 128, 0, stream>>>(frontier_queue->keys[0].GetPointer(util::DEVICE), nodes);
        t_graphslice = pts_inc_graphslice;
        data_slice -> t_offsets    = t_graphslice -> row_offsets;
        data_slice -> t_indices    = t_graphslice -> column_indices;
        data_slice -> t_hash       = data_slice   -> pts_hash;
        bool to_repeat = true;
        while (to_repeat)
        {
            to_repeat = false;
            printf("RuleKernel : pts_inc <- copyIad, pts\n");fflush(stdout);
            util::MemsetKernel   <<<128, 128, 0, stream>>>(data_slice->t_length.GetPointer(util::DEVICE), 0, nodes);
            r_graphslice = enactor_stats->iteration == 0? copyInv_graphslice : copyIad_graphslice;
            s_graphslice = pts_graphslice;
            data_slice -> r_offsets    = r_graphslice -> row_offsets;
            data_slice -> r_offsets2   = NULL;//util::Array1D<SizeT, SizeT>();
            data_slice -> r_indices    = r_graphslice -> column_indices;
            data_slice -> s_offsets    = s_graphslice -> row_offsets;
            data_slice -> s_indices    = s_graphslice -> column_indices;
            data_slice -> h_stride     = nodes;
            data_slice -> h_size       = data_slice   -> t_hash.GetSize();
            data_slice -> t_conflict   = false;
            data_slice ->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
            
            //util::cpu_mt::PrintGPUArray("r_offsets", data_slice->r_offsets.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);
            //util::cpu_mt::PrintGPUArray("s_offsets", data_slice->s_offsets.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);
            //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
            //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
            //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
            //fflush(stdout);
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, RuleKernelFunctor> (
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
                data_slice->r_offsets.GetPointer(util::DEVICE),
                data_slice->r_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                r_graphslice->nodes,
                r_graphslice->edges,
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V,
                false,
                true);
            data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.", __FILE__, __LINE__)) break;

            printf("RuleKernel : pts_inc += copyInv, pts_add\n");fflush(stdout);
            r_graphslice = copyInv_graphslice;
            s_graphslice = enactor_stats->iteration==0? pts_graphslice : pts_add_graphslice;
            data_slice -> r_offsets    = r_graphslice -> row_offsets;
            data_slice -> r_indices    = r_graphslice -> column_indices;
            data_slice -> s_offsets    = s_graphslice -> row_offsets;
            data_slice -> s_indices    = s_graphslice -> column_indices;
            data_slice->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
            //util::cpu_mt::PrintGPUArray("r_offsets", data_slice->r_offsets.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);
            //util::cpu_mt::PrintGPUArray("s_offsets", data_slice->s_offsets.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);
            //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
            //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
            //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
            //fflush(stdout);
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, RuleKernelFunctor> (
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
                data_slice->r_offsets.GetPointer(util::DEVICE),
                data_slice->r_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                r_graphslice->nodes,
                r_graphslice->edges,
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V,
                false,
                true);
            data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.", __FILE__, __LINE__)) break;

            printf("RuleKernel : pts_inc += gepInv, pts_add\n");fflush(stdout);
            r_graphslice = gepInv_graphslice;
            s_graphslice = enactor_stats->iteration==0? pts_graphslice : pts_add_graphslice;
            data_slice -> r_offsets2   = data_slice   -> gepInv_offset.GetPointer(util::DEVICE);
            data_slice -> r_offsets    = r_graphslice -> row_offsets;
            data_slice -> r_indices    = r_graphslice -> column_indices;
            data_slice -> s_offsets    = s_graphslice -> row_offsets;
            data_slice -> s_indices    = s_graphslice -> column_indices;
            data_slice->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
            //util::cpu_mt::PrintGPUArray("r_offsets", data_slice->r_offsets.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);
            //util::cpu_mt::PrintGPUArray("s_offsets", data_slice->s_offsets.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);
            //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
            //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
            //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
            //fflush(stdout);
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, RuleKernelFunctor> (
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
                data_slice->r_offsets.GetPointer(util::DEVICE),
                data_slice->r_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                r_graphslice->nodes,
                r_graphslice->edges,
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V,
                false,
                true);
            data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.", __FILE__, __LINE__)) break;
            //util::cpu_mt::PrintGPUArray("t_graph->length", data_slice->t_length.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);

            if (data_slice->t_conflict) {
                to_repeat = true;
                printf("pts_hash: %d -> %d\n", data_slice->h_size, data_slice->h_size*2);fflush(stdout);
                if (enactor_stats->retval = data_slice->t_hash.EnsureSize(data_slice->h_size*2)) break;
                util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->t_hash.GetPointer(util::DEVICE), 0, data_slice->t_hash.GetSize());
                data_slice -> h_size = data_slice->t_hash.GetSize();
                data_slice -> t_conflict = false;
                data_slice -> data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);

                gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, MakeHashFunctor> (
                    enactor_stats[0],
                    frontier_attribute[0],
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    scanned_edges->GetPointer(util::DEVICE),
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    pts_graphslice->row_offsets.GetPointer(util::DEVICE),
                    pts_graphslice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    pts_graphslice->nodes,
                    pts_graphslice->edges,
                    work_progress[0],
                    context[0],
                    stream,
                    gunrock::oprtr::advance::V2V,
                    false,
                    true); 
            }
        }
        if (enactor_stats->retval) return; 

        util::MemsetKernel<<<1,1,0,stream>>>(data_slice->t_offsets.GetPointer(util::DEVICE), (SizeT)0, 1);
        Scan<mgpu::MgpuScanTypeInc>(
            (SizeT*)data_slice->t_length .GetPointer(util::DEVICE),
            nodes,
            (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
            (SizeT*)(data_slice->t_offsets.GetPointer(util::DEVICE)+1),
            context[0]);
        data_slice->t_offsets.Move(util::DEVICE, util::HOST, 1, nodes, stream);
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.", __FILE__, __LINE__)) return;
        printf("pts: t_graphslice->edges = %d\n", data_slice->t_offsets[nodes]);fflush(stdout);
        t_graphslice->edges = data_slice->t_offsets[nodes];
        if (enactor_stats->retval = data_slice->t_indices.EnsureSize(t_graphslice->edges)) return;
        if (enactor_stats->retval = data_slice->t_marker .EnsureSize(data_slice->t_hash.GetSize())) return;
        util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->t_marker.GetPointer(util::DEVICE), 0, data_slice->t_hash.GetSize());
        util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->t_length.GetPointer(util::DEVICE), 0, nodes);

        r_graphslice = enactor_stats->iteration == 0? copyInv_graphslice : copyIad_graphslice;
        s_graphslice = pts_graphslice;
        data_slice -> r_offsets    = r_graphslice -> row_offsets;
        data_slice -> r_indices    = r_graphslice -> column_indices;
        data_slice -> r_offsets2   = NULL;//util::Array1D<SizeT, SizeT>();
        data_slice -> s_offsets    = s_graphslice -> row_offsets;
        data_slice -> s_indices    = s_graphslice -> column_indices;
        data_slice -> h_stride     = nodes;
        data_slice -> h_size       = data_slice   -> t_hash.GetSize();
        data_slice -> t_conflict   = false;
        data_slice ->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
        printf("RuleKernelFunctor2 pts_inc <- copyIad, pts\n");
        //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
        //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
        //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
        //fflush(stdout);
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, RuleKernelFunctor2> (
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            (VertexId*)NULL,
            data_slice->r_offsets.GetPointer(util::DEVICE),
            data_slice->r_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            r_graphslice->nodes,
            r_graphslice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            true);
        //data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
        //if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;

        r_graphslice = copyInv_graphslice;
        s_graphslice = enactor_stats->iteration==0? pts_graphslice : pts_add_graphslice;
        data_slice -> r_offsets    = r_graphslice -> row_offsets;
        data_slice -> r_indices    = r_graphslice -> column_indices;
        data_slice -> s_offsets    = s_graphslice -> row_offsets;
        data_slice -> s_indices    = s_graphslice -> column_indices;
        data_slice->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
        printf("RuleKernelFunctor2 pts_inc += copyInv, pts_add\n");
        //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
        //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
        //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
        fflush(stdout);
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, RuleKernelFunctor2> (
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            (VertexId*)NULL,
            data_slice->r_offsets.GetPointer(util::DEVICE),
            data_slice->r_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            r_graphslice->nodes,
            r_graphslice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            true);
        //data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
        //if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;

        r_graphslice = gepInv_graphslice;
        s_graphslice = enactor_stats->iteration==0? pts_graphslice : pts_add_graphslice;
        data_slice -> r_offsets2   = data_slice   -> gepInv_offset.GetPointer(util::DEVICE);
        data_slice -> r_offsets    = r_graphslice -> row_offsets;
        data_slice -> r_indices    = r_graphslice -> column_indices;
        data_slice -> s_offsets    = s_graphslice -> row_offsets;
        data_slice -> s_indices    = s_graphslice -> column_indices;
        data_slice->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
        printf("RuleKernelFunctor2 :  pts_inc += gepInv, pts_add\n");
        //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
        //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
        //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
        fflush(stdout);
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, RuleKernelFunctor2> (
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            (VertexId*)NULL,
            data_slice->r_offsets.GetPointer(util::DEVICE),
            data_slice->r_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            r_graphslice->nodes,
            r_graphslice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            true);
        //data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
        //if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;
        t_graphslice -> column_indices = data_slice -> t_indices;
        t_graphslice -> row_offsets    = data_slice -> t_offsets;
        //util::cpu_mt::PrintGPUArray("pts_graph->offsets", t_graphslice->row_offsets.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);
        data_slice->pts_hash = data_slice->t_hash;

        t_graphslice = copyIin_graphslice;
        data_slice -> t_offsets    = t_graphslice -> row_offsets;
        data_slice -> t_indices    = t_graphslice -> column_indices;
        data_slice -> t_hash       = data_slice   -> copyInv_hash;
        util::MemsetKernel   <<<128, 128, 0, stream>>>(data_slice->t_length.GetPointer(util::DEVICE), 0, nodes);
        to_repeat = true;
        while (to_repeat)
        {
            to_repeat = false;
            printf("RuleKernel : copyIin <- loadInv, pts_add\n");fflush(stdout);
            r_graphslice = loadInv_graphslice;
            s_graphslice = enactor_stats->iteration == 0? pts_graphslice : pts_add_graphslice;
            data_slice -> r_offsets    = r_graphslice -> row_offsets;
            data_slice -> r_indices    = r_graphslice -> column_indices;
            data_slice -> r_offsets2   = NULL;//util::Array1D<SizeT, SizeT>();
            data_slice -> s_offsets    = s_graphslice -> row_offsets;
            data_slice -> s_indices    = s_graphslice -> column_indices;
            data_slice -> h_stride     = nodes;
            data_slice -> h_size       = data_slice   -> t_hash.GetSize();
            data_slice -> t_conflict   = false;
            data_slice ->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
            //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
            //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
            //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
            //fflush(stdout);
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, RuleKernelFunctor> (
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
                data_slice->r_offsets.GetPointer(util::DEVICE),
                data_slice->r_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                r_graphslice->nodes,
                r_graphslice->edges,
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V,
                false,
                true);
            data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) break;

            printf("RuleKernel : copyIin += ptsIadd, store\n");fflush(stdout);
            r_graphslice = ptsIadd_graphslice;
            s_graphslice = store_graphslice;
            data_slice -> r_offsets    = r_graphslice -> row_offsets;
            data_slice -> r_indices    = r_graphslice -> column_indices;
            data_slice -> s_offsets    = s_graphslice -> row_offsets;
            data_slice -> s_indices    = s_graphslice -> column_indices;
            data_slice->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
            //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
            //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
            //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
            //fflush(stdout);
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, RuleKernelFunctor> (
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
                data_slice->r_offsets.GetPointer(util::DEVICE),
                data_slice->r_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                r_graphslice->nodes,
                r_graphslice->edges,
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V,
                false,
                true);
            data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) break;

            if (data_slice->t_conflict) {
                to_repeat = true;
                printf("copyInv_hash: %d -> %d\n", data_slice->h_size, data_slice->h_size*2);fflush(stdout);
                if (enactor_stats->retval = data_slice->t_hash.EnsureSize(data_slice->h_size*2)) break;
                util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->t_hash.GetPointer(util::DEVICE), 0, data_slice->t_hash.GetSize());
                data_slice -> h_size = data_slice->t_hash.GetSize();
                data_slice -> t_conflict = false;
                data_slice -> data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);

                gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, MakeHashFunctor> (
                    enactor_stats[0],
                    frontier_attribute[0],
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    scanned_edges->GetPointer(util::DEVICE),
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    t_graphslice->row_offsets.GetPointer(util::DEVICE),
                    t_graphslice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    t_graphslice->nodes,
                    t_graphslice->edges,
                    work_progress[0],
                    context[0],
                    stream,
                    gunrock::oprtr::advance::V2V,
                    false,
                    true); 
            }
        }
        if (enactor_stats->retval) return; 
        
        util::MemsetKernel<<<1,1,0,stream>>>(data_slice->t_offsets.GetPointer(util::DEVICE), 0, 1);
        Scan<mgpu::MgpuScanTypeInc>(
            (SizeT*)data_slice->t_length .GetPointer(util::DEVICE),
            nodes,
            (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
            (SizeT*)(data_slice->t_offsets.GetPointer(util::DEVICE)+1),
            context[0]);
        data_slice->t_offsets.Move(util::DEVICE, util::HOST, 1, nodes, stream);
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        t_graphslice->edges = data_slice->t_offsets[nodes];
        printf("copyInv: edges += %d\n", t_graphslice->edges);fflush(stdout);
        if (enactor_stats->retval = data_slice->t_indices.EnsureSize(t_graphslice->edges)) return;
        if (enactor_stats->retval = data_slice->t_marker .EnsureSize(data_slice->t_hash.GetSize())) return;
        util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->t_marker.GetPointer(util::DEVICE), 0, data_slice->t_hash.GetSize());
        util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->t_length.GetPointer(util::DEVICE), 0, nodes);

        r_graphslice = loadInv_graphslice;
        s_graphslice = enactor_stats->iteration==0? pts_graphslice : pts_add_graphslice;
        data_slice -> r_offsets    = r_graphslice -> row_offsets;
        data_slice -> r_indices    = r_graphslice -> column_indices;
        data_slice -> s_offsets    = s_graphslice -> row_offsets;
        data_slice -> s_indices    = s_graphslice -> column_indices;
        data_slice -> h_stride     = nodes;
        data_slice -> h_size       = data_slice   -> copyInv_hash.GetSize();
        data_slice -> t_conflict   = false;
        data_slice ->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
        printf("RuleKernelFunctor2 : copyInc <- loadInv, pts_add\n");
        //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
        //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
        //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
        fflush(stdout);
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, RuleKernelFunctor2> (
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            (VertexId*)NULL,
            data_slice->r_offsets.GetPointer(util::DEVICE),
            data_slice->r_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            r_graphslice->nodes,
            r_graphslice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            true);
        //data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
        //if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;
        printf("RuleKernelFunctor2 copyIin += ptsIadd, store\n");fflush(stdout);
        r_graphslice = ptsIadd_graphslice;
        s_graphslice = store_graphslice;
        data_slice -> r_offsets    = r_graphslice -> row_offsets;
        data_slice -> r_indices    = r_graphslice -> column_indices;
        data_slice -> s_offsets    = s_graphslice -> row_offsets;
        data_slice -> s_indices    = s_graphslice -> column_indices;
        data_slice->data_slice->Move(util::HOST, util::DEVICE, -1, 0, stream);
        //printf("r_offset = %d,%p r_indices = %d,%p\n", data_slice->r_offsets.GetSize(), data_slice->r_offsets.GetPointer(util::DEVICE), data_slice->r_indices.GetSize(), data_slice->r_indices.GetPointer(util::DEVICE));
        //printf("s_offset = %d,%p s_indices = %d,%p\n", data_slice->s_offsets.GetSize(), data_slice->s_offsets.GetPointer(util::DEVICE), data_slice->s_indices.GetSize(), data_slice->s_indices.GetPointer(util::DEVICE));
        //printf("t_offset = %d,%p t_indices = %d,%p\n", data_slice->t_offsets.GetSize(), data_slice->t_offsets.GetPointer(util::DEVICE), data_slice->t_indices.GetSize(), data_slice->t_indices.GetPointer(util::DEVICE));
        //fflush(stdout);
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, RuleKernelFunctor2> (
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            (VertexId*)NULL,
            data_slice->r_offsets.GetPointer(util::DEVICE),
            data_slice->r_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            r_graphslice->nodes,
            r_graphslice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            true);
        //data_slice->data_slice->Move(util::DEVICE, util::HOST, -1, 0, stream);
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        
        t_graphslice -> column_indices = data_slice->t_indices;
        t_graphslice -> row_offsets    = data_slice->t_offsets;
        //util::cpu_mt::PrintGPUArray("copyInv_graph->offsets", t_graphslice->row_offsets.GetPointer(util::DEVICE), t_graphslice->nodes, -1, -1, -1, stream);
        data_slice->copyInv_hash = data_slice->t_hash;

        r_graphslice = NULL;
        s_graphslice = NULL;
        t_graphslice = NULL;
        data_slice -> r_offsets = util::Array1D<SizeT, SizeT   >();
        data_slice -> r_indices = util::Array1D<SizeT, VertexId>();
        data_slice -> r_offsets2= NULL;//util::Array1D<SizeT, SizeT   >();
        data_slice -> s_offsets = util::Array1D<SizeT, SizeT   >();
        data_slice -> s_indices = util::Array1D<SizeT, VertexId>();
        data_slice -> t_offsets = util::Array1D<SizeT, SizeT   >();
        data_slice -> t_indices = util::Array1D<SizeT, VertexId>();
        data_slice -> t_hash    = util::Array1D<SizeT, VertexId>();
    }

    static cudaError_t Compute_OutputLength(
        FrontierAttribute<SizeT> *frontier_attribute,
        SizeT       *d_offsets,
        VertexId    *d_indices,
        VertexId    *d_in_key_queue,
        SizeT       *partitioned_scanned_edges,
        SizeT        max_in,
        SizeT        max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false)
    {
        return cudaSuccess;  
    }

    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
              int             grid_size,
              int             block_size,
              size_t          shared_size,
              cudaStream_t    stream,
              SizeT           &num_elements,
        const VertexId* const keys_in,
              VertexId*       keys_out,
        const size_t          array_size,
              char*           array)
    {
    }

    static bool Stop_Condition(
        EnactorStats   *enactor_stats,
        FrontierAttribute<SizeT> *frontier_attribute,
        util::Array1D<SizeT, DataSlice> *data_slice,
        int num_gpus)
    {
        //printf("Andersen Stop checked\n");fflush(stdout);
        for (int gpu = 0; gpu < num_gpus*num_gpus; gpu++)
        if (enactor_stats[gpu].retval != cudaSuccess)
        {
            printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu%num_gpus, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
            return true;
        }

        if (num_gpus < 2)
        {
            //printf("checking to_continue = %s\n", data_slice[0]->to_continue? "true" : "false");
            //fflush(stdout);
            return !data_slice[0]->to_continue;
        }
        
        for (int gpu=0; gpu<num_gpus; gpu++)
            if (data_slice[gpu]->to_continue)
        {
            //printf("data_slice[%d]->turn==0\n", gpu);fflush(stdout);
            return false;
        }
        
        for (int gpu=0; gpu<num_gpus; gpu++)
        for (int peer=1; peer<num_gpus; peer++)
        for (int i=0; i<2; i++)
        if (data_slice[gpu]->in_length[i][peer]!=0)
        {
            //printf("data_slice[%d]->in_length[%d][%d] = %d\n", gpu, i, peer, data_slice[gpu]->in_length[i][peer]);fflush(stdout);
            return false;
        }

        for (int gpu=0; gpu<num_gpus; gpu++)
        for (int peer=0; peer<num_gpus; peer++)
        if (data_slice[gpu]->out_length[peer]!=0) 
        {
            //printf("data_slice[%d]->out_length[%d] = %d\n", gpu, peer, data_slice[gpu]->out_length[peer]); fflush(stdout);
            return false;
        }
        //printf("Andersen to stop\n");fflush(stdout);
        return true;
    }

    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    static void Make_Output(
        GraphSlice                    *graph_slice,
        DataSlice                     *data_slice,
        EnactorStats                  *enactor_stats,
        FrontierAttribute<SizeT>      *frontier_attribute,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        SizeT                          num_elements,
        int                            num_gpus,
        int                            thread_num,
        cudaStream_t                   stream,
        ContextPtr                     context)
    {
    }
};

    /**
     * @brief Enacts a connected component computing on the specified graph.
     *
     * @tparam FilterPolicy Kernel policy for vertex mapping.
     * @tparam AndersenProblem Andersen Problem type.
     * @param[in] problem AndersenProblem object.
     * @param[in] max_grid_size Max grid size for Andersen kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename AndersenEnactor>
    static CUT_THREADPROC AndersenThread(
        void * thread_data_)
    {
        typedef typename AndersenEnactor::Problem    Problem;
        typedef typename AndersenEnactor::SizeT      SizeT;
        typedef typename AndersenEnactor::VertexId   VertexId;
        typedef typename AndersenEnactor::Value      Value;
        typedef typename Problem::DataSlice    DataSlice;
        typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
        typedef RuleKernelFunctor<VertexId, SizeT, Value, Problem> AndersenFunctor;
        ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
        Problem      *problem            =  (Problem*)     thread_data->problem;
        AndersenEnactor    *enactor            =  (AndersenEnactor*)   thread_data->enactor;
        int           num_gpus           =   problem     -> num_gpus;
        int           thread_num         =   thread_data -> thread_num;
        int           gpu_idx            =   problem     -> gpu_idx            [thread_num] ;
        DataSlice    *data_slice         =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
        FrontierAttribute<SizeT>
                     *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
        EnactorStats *enactor_stats      = &(enactor     -> enactor_stats      [thread_num * num_gpus]);

        do {
            printf("AndersenThread entered\n");fflush(stdout);
            if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
            thread_data->stats = 1;
            while (thread_data->stats !=2) sleep(0);
            thread_data->stats = 3;

            for (int peer_=0; peer_<num_gpus; peer_++)
            {
                frontier_attribute[peer_].queue_index  = 0;
                frontier_attribute[peer_].selector     = 0;
                frontier_attribute[peer_].queue_length = 0;
                frontier_attribute[peer_].queue_reset  = true;
                enactor_stats     [peer_].iteration    = 0;
            }
            if (num_gpus>1)
            {
                data_slice->vertex_associate_orgs.Move(util::HOST, util::DEVICE);
            }
            
            gunrock::app::Iteration_Loop
                <1,0, AndersenEnactor, AndersenFunctor, AndersenIteration<AdvanceKernelPolicy, FilterKernelPolicy, AndersenEnactor> > (thread_data);

            printf("Andersen_Thread finished\n");fflush(stdout);
        } while (0);
        thread_data->stats = 4;
        CUT_THREADEND;
    }


/**
 * @brief BC problem enactor class.
 *
 * @tparam INSTRUMENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <
    typename _Problem,
    bool _INSTRUMENT,                           // Whether or not to collect per-CTA clock-count statistics
    bool _DEBUG,
    bool _SIZE_CHECK>
class AndersenEnactor : public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>
{
    // Members
    _Problem    *problem      ;
    ThreadSlice *thread_slices;    
    CUTThread   *thread_Ids   ;

    // Methods
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;

public:

    /**
     * @brief AndersenEnactor default constructor
     */
    AndersenEnactor(int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase<SizeT, _DEBUG, _SIZE_CHECK>(EDGE_FRONTIERS, num_gpus, gpu_idx)
    {
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
    }

    /**
     * @brief AndersenEnactor default destructor
     */
    ~AndersenEnactor()
    {
        cutWaitForThreads(thread_Ids, this->num_gpus);
        delete[] thread_Ids   ; thread_Ids    = NULL;
        delete[] thread_slices; thread_slices = NULL;
        problem = NULL;        
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        double &avg_duty)
    {
        unsigned long long total_lifetimes = 0;
        unsigned long long total_runtimes  = 0;
        total_queued = 0;
        
        for (int gpu=0; gpu<this->num_gpus; gpu++)
        {
            if (util::SetDevice(this->gpu_idx[gpu])) return;
            cudaThreadSynchronize();

            total_queued    += this->enactor_stats[gpu].total_queued;
            total_lifetimes += this->enactor_stats[gpu].total_lifetimes;
            total_runtimes  += this->enactor_stats[gpu].total_runtimes;
        }

        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    template<
        typename AdvanceKernelPolity,
        typename FilterKernelPolicy>
    cudaError_t InitAndersen(
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size = 512,
        bool        size_check    = true)
    {
        cudaError_t retval = cudaSuccess;
        //cpu_barrier = new util::cpu_mt::CPUBarrier[2];
        //cpu_barrier[0]=util::cpu_mt::CreateBarrier(this->num_gpus);
        //cpu_barrier[1]=util::cpu_mt::CreateBarrier(this->num_gpus);
        // Lazy initialization
        if (retval = EnactorBase <SizeT, DEBUG, SIZE_CHECK> ::Init(
            problem,
            max_grid_size,
            AdvanceKernelPolity::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY)) return retval;

        if (DEBUG) {
            printf("Andersen vertex map occupancy %d, level-grid size %d\n",
                        FilterKernelPolicy::CTA_OCCUPANCY, this->enactor_stats[0].filter_grid_size);
        }

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            //thread_slices[gpu].cpu_barrier  = cpu_barrier;
            thread_slices[gpu].thread_num   = gpu;
            thread_slices[gpu].problem      = (void*)problem;
            thread_slices[gpu].enactor      = (void*)this;
            thread_slices[gpu].context      =&(context[gpu*this->num_gpus]);
            thread_slices[gpu].stats        = -1;
            thread_slices[gpu].thread_Id = cutStartThread(
                (CUT_THREADROUTINE)&(AndersenThread<
                    AdvanceKernelPolity, FilterKernelPolicy,
                    AndersenEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
        }
        return retval;
    }

    cudaError_t Reset()
    {
        return EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Reset();
    }
 
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactAndersen()
    {
        cudaError_t              retval         = cudaSuccess;

        do {
            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                while (thread_slices[gpu].stats!=1) sleep(0);
                thread_slices[gpu].stats=2;
            }
            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                while (thread_slices[gpu].stats!=4) sleep(0);
            }

            for (int gpu=0;gpu< this->num_gpus;gpu++)
                if (this->enactor_stats[gpu].retval!=cudaSuccess)
                {retval=this->enactor_stats[gpu].retval;break;}
        } while (0);
        if (this->DEBUG) printf("\nGPU Andersen Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Enact Kernel Entry, specify KernelPolicy
     *
     * @tparam AndersenProblem Andersen Problem type. @see AndersenProblem
     * @param[in] problem Pointer to AndersenProblem object.
     * @param[in] max_grid_size Max grid size for Andersen kernel calls. 
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename AndersenProblem>
    cudaError_t Enact()
        //AndersenProblem                      *problem,
        //int                             max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
            if (min_sm_version == -1 || this->cuda_props[gpu].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[gpu].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::advance::KernelPolicy<
                Problem,                            //Problem data type
                300,                                //CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY,
                7,                                  // LOG_THREADS,
                8,                                  // LOG_BLOCKS,
                32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE,
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS,
                32,                                 // WART_GATHER_THRESHOLD,
                128 * 4,                            // CTA_GATHER_THRESHOLD,
                7,                                  // LOG_SCHEDULE_GRANULARITY,
                gunrock::oprtr::advance::LB>
                    AdvancePolicy;

            typedef gunrock::oprtr::filter::KernelPolicy<
                Problem,                            // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                7,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                0,                                  // END_BITMASK (no bitmask for andersen)
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterPolicy;

            return EnactAndersen<AdvancePolicy, FilterPolicy>();
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    cudaError_t Init(
            ContextPtr *context,
            Problem    *problem,
            int         max_grid_size = 512,
            bool        size_check    = true)
    {
        int min_sm_version = -1;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
            if (min_sm_version == -1 || this->cuda_props[gpu].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[gpu].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::advance::KernelPolicy<
                Problem,                            //Problem data type
                300,                                //CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY,
                7,                                  // LOG_THREADS,
                8,                                  // LOG_BLOCKS,
                32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE,
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS,
                32,                                 // WART_GATHER_THRESHOLD,
                128 * 4,                            // CTA_GATHER_THRESHOLD,
                7,                                  // LOG_SCHEDULE_GRANULARITY,
                gunrock::oprtr::advance::LB>
                    AdvancePolicy;

            typedef gunrock::oprtr::filter::KernelPolicy<
                Problem,                            // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                7,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                0,                                  // END_BITMASK (no bitmask for andersen)
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterPolicy;

            return InitAndersen<AdvancePolicy, FilterPolicy>(
                    context, problem, max_grid_size, size_check);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;

    }
    /** @} */
};

} // namespace andersen
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
