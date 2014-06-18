// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * partitioner_base.cuh
 *
 * @brief Base struct for all the partitioner types
 */

#pragma once

#include <gunrock/util/basic_utils.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/multithreading.cuh>

#include <vector>

namespace gunrock {
namespace app {

/**
 * @brief Base partitioner structure.
 *
 */

template <
    typename   _VertexId,
    typename   _SizeT,
    typename   _Value>
struct PartitionerBase
{
    typedef _VertexId  VertexId;
    typedef _SizeT     SizeT;
    typedef _Value     Value;
    typedef Csr<VertexId,Value,SizeT> GraphT;

    // Members
public:
    // Number of GPUs to be partitioned
    int        num_gpus;
    int        Status;

    // Original graph
    const GraphT *graph;

    // Partioned graphs
    GraphT *sub_graphs;

    int       **partition_tables;
    VertexId  **convertion_tables;
    VertexId  **original_vertexes;
    SizeT     **in_offsets;
    SizeT     **out_offsets;
    //Mthods

    template <
        typename VertexId,
        typename SizeT,
        typename Value>
    struct ThreadSlice
    {
    public:
        const GraphT     *graph;
        GraphT     *sub_graph;
        int        thread_num,num_gpus;
        util::cpu_mt::CPUBarrier* cpu_barrier;
        CUTThread  thread_Id;
        int        *partition_table0,**partition_table1;
        VertexId   *convertion_table0,**convertion_table1;
        VertexId   **original_vertexes;
        SizeT      **in_offsets,**out_offsets;
    };

    /**
     * @brief PartitionerBase default constructor
     */
    PartitionerBase()
    {
        Status            = 0;
        num_gpus          = 0;
        graph             = NULL;
        sub_graphs        = NULL;
        partition_tables  = NULL;
        convertion_tables = NULL;
        original_vertexes = NULL;
        in_offsets        = NULL;
        out_offsets       = NULL;
    }

    virtual ~PartitionerBase()
    {
        if (Status == 0) return;        
        Status   = 0;
        num_gpus = 0;
    } 

    cudaError_t Init(
        const GraphT &graph,
        int   num_gpus)
    {   
        cudaError_t retval= cudaSuccess;
        this->num_gpus    = num_gpus;
        this->graph       = &graph;
        sub_graphs        = new GraphT   [num_gpus  ];
        partition_tables  = new int*     [num_gpus+1];
        convertion_tables = new VertexId*[num_gpus+1];
        original_vertexes = new VertexId*[num_gpus  ];
        in_offsets        = new SizeT*   [num_gpus  ];
        out_offsets       = new SizeT*   [num_gpus  ];
        
        for (int i=0;i<num_gpus+1;i++)
        {
            partition_tables [i] = NULL;
            convertion_tables[i] = NULL;
            if (i!=num_gpus) original_vertexes[i] = NULL;
        }
        partition_tables [0] = new int     [graph.nodes];
        convertion_tables[0] = new VertexId[graph.nodes];
        memset(partition_tables [0], 0, sizeof(int     ) * graph.nodes);
        memset(convertion_tables[0], 0, sizeof(VertexId) * graph.nodes);
        for (int i=0;i<num_gpus;i++)
        {
            in_offsets [i] = new SizeT [num_gpus+1];
            out_offsets[i] = new SizeT [num_gpus+1];
            memset(in_offsets [i], 0, sizeof(SizeT) * (num_gpus+1));
            memset(out_offsets[i], 0, sizeof(SizeT) * (num_gpus+1)); 
        }
        Status = 1;

        return retval;
    }
    
    static CUT_THREADPROC MakeSubGraph_Thread(void *thread_data_)
    {
        ThreadSlice<VertexId,SizeT,Value> *thread_data = (ThreadSlice<VertexId,SizeT,Value> *) thread_data_;
        const GraphT* graph           = thread_data->graph;
        GraphT*     sub_graph         = thread_data->sub_graph;
        int         gpu               = thread_data->thread_num;
        util::cpu_mt::CPUBarrier* cpu_barrier = thread_data->cpu_barrier;
        int         num_gpus          = thread_data->num_gpus;
        int*        partition_table0  = thread_data->partition_table0;
        VertexId*   convertion_table0 = thread_data->convertion_table0;
        int**       partition_table1  = thread_data->partition_table1;
        VertexId**  convertion_table1 = thread_data->convertion_table1;
        VertexId**  original_vertexes = thread_data->original_vertexes;
        SizeT**     out_offsets       = thread_data->out_offsets;
        SizeT**     in_offsets        = thread_data->in_offsets;
        SizeT       num_nodes         = 0, node_counter;
        SizeT       num_edges         = 0, edge_counter;
        int*        marker            = new int[graph->nodes];
        SizeT*      cross_counter     = new SizeT[num_gpus];
        VertexId*   tconvertion_table = new VertexId[graph->nodes];

        memset(marker, 0, sizeof(int)*graph->nodes);
        memset(cross_counter, 0, sizeof(SizeT) * num_gpus);

        for (SizeT node=0; node<graph->nodes; node++)
        if (partition_table0[node] == gpu)
        {
            convertion_table0[node] = cross_counter[gpu];
            tconvertion_table[node] = cross_counter[gpu];
            marker[node] =1;
            for (SizeT edge=graph->row_offsets[node]; edge<graph->row_offsets[node+1]; edge++)
            {
                SizeT neibor = graph->column_indices[edge];
                int peer  = partition_table0[neibor];
                if ((peer != gpu) && (marker[neibor] == 0))
                {
                    tconvertion_table[neibor]=cross_counter[peer];
                    cross_counter[peer]++;
                    marker[neibor]=1;
                    num_nodes++;
                }
            }
            cross_counter[gpu]++;
            num_nodes++;
            num_edges+= graph->row_offsets[node+1] - graph->row_offsets[node];
        }
        delete[] marker;marker=NULL;
        out_offsets[gpu][0]=0;
        node_counter=cross_counter[gpu];
        for (int peer=0;peer<num_gpus;peer++)
        {
            if (peer==gpu) continue;
            int peer_=peer < gpu? peer+1 : peer;
            out_offsets[gpu][peer_]=node_counter;
            node_counter+=cross_counter[peer];
        }
        out_offsets[gpu][num_gpus]=node_counter;

        util::cpu_mt::IncrementnWaitBarrier(cpu_barrier,gpu);
        
        in_offsets[gpu][0]=0;
        node_counter=0;
        for (int peer=0;peer<num_gpus;peer++)
        {
            if (peer==gpu) continue;
            int peer_ = peer < gpu ? peer+1 : peer;
            int gpu_  = gpu  < peer? gpu +1 : gpu ; 
            in_offsets[gpu][peer_]=node_counter;
            node_counter+=out_offsets[peer][gpu_+1]-out_offsets[peer][gpu_];
        }
        in_offsets[gpu][num_gpus]=node_counter;
        
        if      (graph->node_values == NULL && graph->edge_values == NULL) 
             sub_graph->template FromScratch < false , false  >(num_nodes,num_edges);
        else if (graph->node_values != NULL && graph->edge_values == NULL) 
             sub_graph->template FromScratch < false , true   >(num_nodes,num_edges);
        else if (graph->node_values == NULL && graph->edge_values != NULL) 
             sub_graph->template FromScratch < true  , false  >(num_nodes,num_edges);
        else sub_graph->template FromScratch < true  , true   >(num_nodes,num_edges);

        if (convertion_table1[0] != NULL) free(convertion_table1[0]);
        if (partition_table1 [0] != NULL) free(partition_table1[0]);
        convertion_table1[0]= (VertexId*) malloc (sizeof(VertexId) * num_nodes);//new VertexId[num_nodes];
        partition_table1 [0]= (int*)      malloc (sizeof(int)      * num_nodes);//new int     [num_nodes];
        original_vertexes[0]= (VertexId*) malloc (sizeof(VertexId) * num_nodes);//new VertexId[num_nodes];
        edge_counter=0;
        for (SizeT node=0; node<graph->nodes; node++)
        if (partition_table0[node] == gpu)
        {
            VertexId node_ = tconvertion_table[node];
            sub_graph->row_offsets[node_]=edge_counter;
            if (graph->node_values != NULL) sub_graph->node_values[node_]=graph->node_values[node];
            partition_table1 [0][node_] = 0;
            convertion_table1[0][node_] = node_;
            original_vertexes[0][node_] = node;
            for (SizeT edge=graph->row_offsets[node]; edge<graph->row_offsets[node+1]; edge++)
            {
                SizeT    neibor  = graph->column_indices[edge];
                int      peer    = partition_table0[neibor];
                int      peer_   = peer < gpu ? peer+1 : peer;
                if (peer == gpu) peer_ = 0;
                VertexId neibor_ = tconvertion_table[neibor] + out_offsets[gpu][peer_];
                
                sub_graph->column_indices[edge_counter] = neibor_;
                if (graph->edge_values !=NULL) sub_graph->edge_values[edge_counter]=graph->edge_values[edge];
                if (peer != gpu)
                {
                    sub_graph->row_offsets[neibor_]=num_edges;
                    partition_table1 [0][neibor_] = peer_;
                    convertion_table1[0][neibor_] = convertion_table0[neibor];
                    original_vertexes[0][neibor_] = neibor;
                }
                edge_counter++;
            }   
        }
        sub_graph->row_offsets[num_nodes]=num_edges;

        delete[] cross_counter;     cross_counter     = NULL;
        delete[] tconvertion_table; tconvertion_table = NULL;
        CUT_THREADEND;
    }

    cudaError_t MakeSubGraph()
    {
        cudaError_t retval = cudaSuccess;
        ThreadSlice<VertexId,SizeT,Value>* thread_data = new ThreadSlice<VertexId,SizeT,Value>[num_gpus];
        CUTThread*   thread_Ids  = new CUTThread  [num_gpus];
        util::cpu_mt::CPUBarrier   cpu_barrier; //= cutCreateBarrier(num_gpus);
        cpu_barrier = util::cpu_mt::CreateBarrier(num_gpus);

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            thread_data[gpu].graph             = graph;
            thread_data[gpu].sub_graph         = &(sub_graphs[gpu]);
            thread_data[gpu].thread_num        = gpu;
            thread_data[gpu].cpu_barrier       = &cpu_barrier;
            thread_data[gpu].num_gpus          = num_gpus;
            thread_data[gpu].partition_table0  = partition_tables [0];
            thread_data[gpu].convertion_table0 = convertion_tables[0];
            thread_data[gpu].partition_table1  = &(partition_tables[gpu+1]);
            thread_data[gpu].convertion_table1 = &(convertion_tables[gpu+1]);
            thread_data[gpu].original_vertexes = &(original_vertexes[gpu]);
            thread_data[gpu].in_offsets        = in_offsets;
            thread_data[gpu].out_offsets       = out_offsets;
            thread_data[gpu].thread_Id         = cutStartThread((CUT_THREADROUTINE)&(MakeSubGraph_Thread), (void*)(&(thread_data[gpu])));
            thread_Ids[gpu]=thread_data[gpu].thread_Id;
        }

        cutWaitForThreads(thread_Ids,num_gpus);
        util::cpu_mt::DestoryBarrier(&cpu_barrier);
        delete[] thread_Ids ;thread_Ids =NULL;
        delete[] thread_data;thread_data=NULL;
        Status = 2;
        return retval;
    }

    virtual cudaError_t Partition(
        GraphT*    &sub_graphs,
        int**      &partition_tables,
        VertexId** &convertion_tables,
        VertexId** &original_vertexes,
        SizeT**    &in_offsets,
        SizeT**    &out_offsets)
    {
        return util::GRError("PartitionBase::Partition is undefined", __FILE__, __LINE__);
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
