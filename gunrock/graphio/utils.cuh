// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * utils.cuh
 *
 * @brief General graph-building utility routines
 */

#pragma once

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <omp.h>

#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/random_bits.cuh>

#include <gunrock/coo.cuh>
#include <gunrock/csr.cuh>

namespace gunrock {
namespace graphio {

/**
 * @brief Generates a random node-ID in the range of [0, num_nodes)
 *
 * @param[in] num_nodes Number of nodes in Graph
 *
 * \return random node-ID
 */
template <typename SizeT>
SizeT RandomNode (SizeT num_nodes)
{
    SizeT node_id;
    util::RandomBits(node_id);
    if (node_id < 0) node_id *= -1;
    return node_id % num_nodes;
}

double Sprng (struct drand48_data *rand_data)
{
    double result;
    drand48_r(rand_data, &result);
    return result;
}

bool Flip (struct drand48_data *rand_data)
{
    return Sprng(rand_data) >= 0.5;
}

template <typename VertexId>
void ChoosePartition (
    VertexId *u, VertexId *v, VertexId step,
    double a, double b, double c, double d, struct drand48_data *rand_data)
{
    double p;
    p = Sprng(rand_data);

    if (p < a)
    {
        // do nothing
    }
    else if ((a < p) && (p < a + b))
    {
        *v = *v + step;
    }
    else if ((a + b < p) && (p < a + b + c))
    {
        *u = *u + step;
    }
    else if ((a + b + c < p) && (p < a + b + c + d))
    {
        *u = *u + step;
        *v = *v + step;
    }
}

void VaryParams (double *a, double *b, double *c, double *d, drand48_data *rand_data)
{
    double v, S;

    // Allow a max. of 5% variation
    v = 0.05;

    if (Flip(rand_data))
    {
        *a += *a * v * Sprng(rand_data);
    }
    else
    {
        *a -= *a * v * Sprng(rand_data);
    }
    if (Flip(rand_data))
    {
        *b += *b * v * Sprng(rand_data);
    }
    else
    {
        *b -= *b * v * Sprng(rand_data);
    }
    if (Flip(rand_data))
    {
        *c += *c * v * Sprng(rand_data);
    }
    else
    {
        *c -= *c * v * Sprng(rand_data);
    }
    if (Flip(rand_data))
    {
        *d += *d * v * Sprng(rand_data);
    }
    else
    {
        *d -= *d * v * Sprng(rand_data);
    }

    S = *a + *b + *c + *d;

    *a = *a / S;
    *b = *b / S;
    *c = *c / S;
    *d = *d / S;
}

template <typename VertexId, typename Value, typename SizeT>
void RemoveStandaloneNodes(
    Csr<VertexId, Value, SizeT>* graph)
{
    //typedef Graph::SizeT    SizeT;
    //typedef Graph::VertexId VertexId;
    //typedef Graph::Value    Value;

    SizeT nodes = graph->nodes;
    SizeT edges = graph->edges;
    int *marker = new int[nodes];
    memset(marker, 0, sizeof(int) * nodes);
    VertexId *column_indices = graph->column_indices;
    SizeT    *row_offsets    = graph->row_offsets;
    SizeT    *displacements  = new SizeT   [nodes];
    SizeT    *new_offsets    = new SizeT   [nodes+1];
    SizeT    *block_offsets  = NULL;
    VertexId *new_nodes      = new VertexId[nodes];
    Value    *new_values     = new Value   [nodes];
    Value    *values         = graph->node_values;
    int       num_threads    = 0;

    #pragma omp parallel
    {
            num_threads  = omp_get_num_threads();
        int thread_num   = omp_get_thread_num ();
        SizeT edge_start = (long long)(edges) * thread_num / num_threads;
        SizeT edge_end   = (long long)(edges) * (thread_num+1) / num_threads;
        SizeT node_start = (long long)(nodes) * thread_num / num_threads;
        SizeT node_end   = (long long)(nodes) * (thread_num+1) / num_threads;

        for (SizeT    edge = edge_start; edge < edge_end; edge++)
            marker[column_indices[edge]] = 1;
        for (VertexId node = node_start; node < node_end; node++)
            if (row_offsets[node] != row_offsets[node+1])
            marker[node] = 1;
        if (thread_num == 0) block_offsets = new SizeT[num_threads+1];
        #pragma omp barrier
        
        //SizeT counter = 0;
        displacements[node_start] = 0;
        for (VertexId node = node_start; node < node_end-1; node++)
            displacements[node+1] = displacements[node] + 1 - marker[node];
        if (node_end != 0)
            block_offsets[thread_num + 1] = displacements[node_end -1] + 1 - marker[node_end-1];
        else block_offsets[thread_num + 1] = 1 - marker[0];

        #pragma omp barrier
        #pragma omp single
        {
            block_offsets[0] = 0;
            for (int i=0; i<num_threads; i++)
                block_offsets[i+1] += block_offsets[i];
            //util::cpu_mt::PrintCPUArray("block_offsets", block_offsets, num_threads+1);
        }

        for (VertexId node = node_start; node < node_end; node++)
        {
            if (marker[node] == 0) continue;
            VertexId node_ = node - block_offsets[thread_num] - displacements[node];
            new_nodes  [node ] = node_;
            new_offsets[node_] = row_offsets[node];
            if (values != NULL) new_values[node_] = values[node];
        }
        //#pragma omp barrier
        //printf("thread %d\n", thread_num);fflush(stdout);
    }

    for (SizeT edge = 0; edge < edges; edge++)
    {
        column_indices[edge] = new_nodes[column_indices[edge]];
    }

    //printf("num_threads = %d\n", num_threads);
    nodes = nodes - block_offsets[num_threads];
    memcpy(row_offsets, new_offsets, sizeof(SizeT) * (nodes + 1));
    if (values!=NULL) memcpy(values, new_values, sizeof(Value) * nodes);
    printf("graph #nodes : %lld -> %lld \n", (long long)graph->nodes, (long long)nodes);
    graph->nodes = nodes;
    row_offsets[nodes] = graph->edges;

    delete[] new_offsets  ; new_offsets   = NULL;
    delete[] new_values   ; new_values    = NULL;
    delete[] new_nodes    ; new_nodes     = NULL;
    delete[] marker       ; marker        = NULL;
    delete[] displacements; displacements = NULL;
    delete[] block_offsets; block_offsets = NULL;
}

} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
