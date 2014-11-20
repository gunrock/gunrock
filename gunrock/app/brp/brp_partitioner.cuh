// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * brp_partitioner.cuh
 *
 * @brief Implementation of biased random partitioner
 */

#pragma once

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

#include <gunrock/app/partitioner_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace app {
namespace brp {
    
    template <typename SizeT>
    struct sort_node
    {   
    public:
        SizeT posit;
        int   value;
    
        bool operator==(const sort_node& node) const
        {   
            return (node.value == value);
        }   

        bool operator<(const sort_node& node) const
        {   
            return (node.value < value);
        }   
    
        sort_node & operator=(const sort_node &rhs)
        {   
            this->posit=rhs.posit;
            this->value=rhs.value;
            return *this;
        }   
    };
    
    template <typename SizeT>
    bool compare_sort_node(sort_node<SizeT> A, sort_node<SizeT> B)
    {
        return (A.value<B.value);
    }

template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool     ENABLE_BACKWARD = false>
struct BiasRandomPartitioner : PartitionerBase<VertexId,SizeT,Value,ENABLE_BACKWARD>
{
    typedef Csr<VertexId,Value,SizeT> GraphT;

    // Members
    float *weitage;

    // Methods
    BiasRandomPartitioner()
    {
        weitage=NULL;
    }

    BiasRandomPartitioner(const GraphT &graph,
                      int   num_gpus,
                      float *weitage = NULL)
    {
        Init2(graph,num_gpus,weitage);
    }

    void Init2(
        const GraphT &graph,
        int num_gpus,
        float *weitage)
    {
        this->Init(graph,num_gpus);
        this->weitage=new float[num_gpus+1];
        if (weitage==NULL)
            for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu]=1.0f/num_gpus;
        else {
            float sum=0;
            for (int gpu=0;gpu<num_gpus;gpu++) sum+=weitage[gpu];
            for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu]=weitage[gpu]/sum;
        }
        //for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu+1]+=this->weitage[gpu];
    }

    ~BiasRandomPartitioner()
    {
        if (weitage!=NULL)
        {
            delete[] weitage;weitage=NULL;
        }
    }

    cudaError_t Partition(
        GraphT*    &sub_graphs,
        int**      &partition_tables,
        VertexId** &convertion_tables,
        VertexId** &original_vertexes,
        //SizeT**    &in_offsets,
        SizeT**    &in_counter,
        SizeT**    &out_offsets,
        SizeT**    &out_counter,
        SizeT**    &backward_offsets,
        int**      &backward_partitions,
        VertexId** &backward_convertions)
    {
        cudaError_t retval = cudaSuccess;
        int*        tpartition_table=this->partition_tables[0];
        SizeT       nodes  = this->graph->nodes;
        sort_node<SizeT> *sort_list = new sort_node<SizeT>[nodes];
        VertexId    *t_queue = new VertexId[nodes];
        VertexId    *marker = new VertexId[nodes];
        SizeT       total_count = 0, current=0, tail = 0, level = 0;
        SizeT       *counter = new SizeT[this->num_gpus+1];
        SizeT       n1 = 1;//, n2 = 1;
        SizeT       target_level = n1;
        SizeT       *level_tail = new SizeT[target_level+1];
        float       f1 = 0.5;//1.0/this->num_gpus;
        float       *gpu_percentage=new float[this->num_gpus+1];
        SizeT       *current_count = new SizeT[this->num_gpus];
        VertexId    StartId, EndId;
        VertexId    *row_offsets=this->graph->row_offsets;
        VertexId    *column_indices=this->graph->column_indices;
        int         seed = time(NULL);

        srand(seed);
        printf("seed = %d\n",seed);fflush(stdout);
        target_level = n1;//(n1<n2? n2:n1);
        for (SizeT node=0;node<nodes;node++)
        {
            sort_list[node].value=this->graph->row_offsets[node+1]-this->graph->row_offsets[node];
            sort_list[node].posit=node;
            tpartition_table[node]=this->num_gpus;
        }
        for (int i=0;i<this->num_gpus;i++) current_count[i]=0;
        memset(marker,0,sizeof(VertexId)*nodes);
        std::vector<sort_node<SizeT> > sort_vector(sort_list, sort_list+nodes);
        std::sort(sort_vector.begin(),sort_vector.end());

        //printf("1");fflush(stdout);
        for (SizeT pos=0;pos<nodes;pos++)
        {
            VertexId node = sort_vector[pos].posit;
            if (tpartition_table[node]!=this->num_gpus) continue;
            //printf("node = %d, value =%d\t",node, sort_vector[pos].value);fflush(stdout);
            current = 0; tail = 0; level = 0; total_count = 0;
            t_queue[current] = node;
            marker[node]=node;
            //tpartition_table[node]=this->num_gpus+1;
            for (SizeT i=0;i<=this->num_gpus;i++) counter[i]=0;
            //counter[this->num_gpus]=1;
            //memset(marker,0,sizeof(int)*nodes);
            //while (level < (n1<n2? n2:n1))
            for (level=0;level<target_level;level++)
            {
                level_tail[level] = tail;
                //printf("level = %d\t",level);fflush(stdout);
                //while (current <= level_tail[level])
                for (;current<=level_tail[level];current++)
                {
                    VertexId t_node=t_queue[current];
                    StartId = row_offsets[t_node];
                    EndId   = row_offsets[t_node+1];
                    //printf("t_node = %d\t",t_node);fflush(stdout);
                    for (VertexId i=StartId;i<EndId;i++)
                    {
                        VertexId neibor=column_indices[i];
                        if (marker[neibor]==node) continue;
                        if (tpartition_table[neibor]<this->num_gpus)
                        {
                            if (level < n1) 
                            {
                                counter[tpartition_table[neibor]]++;
                                //total_count++;
                            }
                        } else {
                            /*if (level < n2) 
                            {
                                counter[this->num_gpus]++;
                                tpartition_table[neibor]=this->num_gpus+1;
                                //printf("%d\t",neibor);
                            }*/
                            //if (level < n1) total_count++;
                        }
                        //if (level < n1) total_count++;
                        marker[neibor]=node;
                        tail++;t_queue[tail]=neibor;
                    }
                    //current ++;
                }
                //level++;
            }
            level_tail[level]=tail;

            total_count=0;
            for (int i=0;i<this->num_gpus;i++) 
            {
                total_count+=counter[i];
                //printf("c%d = %d ",i,counter[i]);
            }
            for (int i=0;i<this->num_gpus;i++) 
            {
                gpu_percentage[i]=(total_count==0?0:(f1*counter[i]/total_count));
                //printf("g%d = %f ",i,gpu_percentage[i]);
            }
            total_count=0;
            for (int i=0;i<this->num_gpus;i++)
            {
                SizeT e=nodes*weitage[i]-current_count[i]; 
                total_count+=(e>=0?e:0);
            }
            for (int i=0;i<this->num_gpus;i++)
            {
                SizeT e=nodes*weitage[i]-current_count[i];
                gpu_percentage[i]+=(e>0?((1-f1)*e/total_count):0);
                //printf("e%d = %d ",i,e);
            }
            float total_percentage=0;
            for (int i=0;i<this->num_gpus;i++)
                total_percentage+=gpu_percentage[i];
            for (int i=0;i<this->num_gpus;i++)
            {
                gpu_percentage[i]=gpu_percentage[i]/total_percentage;
                //printf("p%d = %f ",i,gpu_percentage[i]);
            }
            gpu_percentage[this->num_gpus]=1;
            for (int i=this->num_gpus-1;i>=0;i--)
                gpu_percentage[i]=gpu_percentage[i+1]-gpu_percentage[i];
            float x=1.0f*rand()/RAND_MAX;
            //printf("%f %f %f %f : %f\t",gpu_percentage[0],gpu_percentage[1],gpu_percentage[2],gpu_percentage[3],x);
            for (int i=0;i<this->num_gpus;i++)
                if (x>=gpu_percentage[i] && x<gpu_percentage[i+1]) 
                {
                    current_count[i]++;
                    tpartition_table[node]=i;
                    //printf("%d -> %d",node,i);
                    break;
                }
            if (tpartition_table[node]>= this->num_gpus)
            {
                tpartition_table[node]=(rand()%(this->num_gpus));
                current_count[tpartition_table[node]]++;
            }
            //printf("\n");
        }

        delete[] sort_list; sort_list = NULL;
        delete[] t_queue  ; t_queue   = NULL;
        delete[] counter  ; counter   = NULL;
        delete[] marker   ; marker    = NULL;
        delete[] level_tail; level_tail = NULL;
        delete[] current_count; current_count = NULL;
        delete[] gpu_percentage; gpu_percentage = NULL;
        retval = this->MakeSubGraph
                 ();
        sub_graphs          = this->sub_graphs;
        partition_tables    = this->partition_tables;
        convertion_tables   = this->convertion_tables;
        original_vertexes   = this->original_vertexes;
        //in_offsets          = this->in_offsets;
        in_counter          = this->in_counter;
        out_offsets         = this->out_offsets;
        out_counter         = this->out_counter;
        backward_offsets    = this->backward_offsets;
        backward_partitions = this->backward_partitions;
        backward_convertions= this->backward_convertions;
        return retval;
    }
};

} //namespace cp
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
