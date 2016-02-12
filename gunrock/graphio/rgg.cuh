// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.

/**
 * @file
 * rgg.cuh
 *
 * @brief RGG Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <list>
#include <random>
#include <gunrock/graphio/utils.cuh>
#include <gunrock/util/sort_omp.cuh>

namespace gunrock {
namespace graphio {
namespace rgg {

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<double> Distribution;

template <typename T>
inline T SqrtSum(T x, T y)
{
    return sqrt(x*x + y*y);
}

template <typename T>
T P2PDistance(T co_x0, T co_y0, T co_x1, T co_y1)
{
    return SqrtSum(co_x0 - co_x1, co_y0 - co_y1);
}

class RggPoint {
public:
    double x, y;
    long long node;

    RggPoint() {}
    RggPoint(double x, double y, long long node)
    {
        this->x = x;
        this->y = y;
        this->node = node;
    }
};

//inline bool operator< (const RggPoint& lhs, const RggPoint& rhs)
template <typename Point>
bool XFirstPointCompare (
    Point lhs,
    Point rhs)
{
    if (lhs.x < rhs.x) return true;
    if (lhs.x > rhs.x) return false;
    if (lhs.y < rhs.y) return true;
    return false;
}

template <typename T>
bool PureTwoFactor(T x)
{
    if (x<3) return true;
    while (x > 0)
    {
        if ((x%2) != 0) return false;
        x /= 2;
    }
    return true;
}

/*
 * @brief Build random geometry graph (RGG).
 *
 * @tparam WITH_VALUES Whether or not associate with per edge weight values.
 * @tparam VertexId Vertex identifier.
 * @tparam Value Value type.
 * @tparam SizeT Graph size type.
 *
 * @param[in] nodes
 * @param[in] graph
 * @param[in] threshould
 * @param[in] undirected
 * @param[in] value_multipiler
 * @param[in] value_min
 * @param[in] seed
 */
template <bool WITH_VALUES, typename VertexId, typename SizeT, typename Value>
int BuildRggGraph(
    SizeT nodes,
    Csr<VertexId, SizeT, Value> &graph,
    double threshold  = -1,
    bool   undirected = true,
    double value_multipiler = 1,
    double value_min        = 1,
    int    seed             = -1,
    bool   quiet = false)
{
    typedef Coo<VertexId, Value> EdgeTupleType;

    if (nodes < 0)
    {
        fprintf(stderr, "Invalid graph size: nodes = %lld\n", (long long)nodes);
        return -1;
    }

    int       reserved_size = 50;
    RggPoint *points        = new RggPoint[nodes+1];
    SizeT    *row_offsets   = new SizeT[nodes+1];
    VertexId *col_index_    = new VertexId[reserved_size * nodes];
    Value    *values_       = WITH_VALUES ? new Value[reserved_size * nodes] : NULL;
    SizeT    *offsets       = NULL;
    if (threshold < 0)
              threshold     = 0.55 * sqrt(log(nodes)/nodes);
    SizeT     edges         = 0;
    long long row_length    = 1.0 / threshold + 1;
    VertexId **blocks       = new VertexId* [row_length * row_length + 1];
    SizeT    *block_size    = new SizeT     [row_length * row_length + 1];
    SizeT    *block_length  = new SizeT     [row_length * row_length + 1];
    EdgeTupleType *coo      = NULL;
    long long reserved_factor2 = 8;
    long long initial_length   = reserved_factor2 * nodes / row_length / row_length;

    if (seed == -1) seed = time(NULL);
    if (!quiet) { printf("rgg seed = %lld\n", (long long)seed); }
    if (initial_length <4) initial_length = 4;
    for (SizeT i=0; i< row_length * row_length +1; i++)
    {
        block_size  [i] = 0;
        block_length[i] = 0;
        blocks      [i] = NULL;
    }

    #pragma omp parallel
    {
        int       thread_num  = omp_get_thread_num();
        int       num_threads = omp_get_num_threads();
        SizeT     node_start  = (long long)(nodes) * thread_num / num_threads;
        SizeT     node_end    = (long long)(nodes) * (thread_num + 1) / num_threads;
        SizeT     counter     = 0;
        VertexId *col_index   = col_index_ + reserved_size * node_start;
        Value    *values      = WITH_VALUES ? values_ + reserved_size * node_start : NULL;
        unsigned int seed_    = seed + 805 * thread_num;
        Engine    engine(seed_);
        Distribution distribution(0.0, 1.0);

        #pragma omp single
            offsets           = new SizeT[num_threads+1];

        for (VertexId node = node_start; node < node_end; node++)
        {
            points[node].x = distribution(engine);
            points[node].y = distribution(engine);
            points[node].node = node;
        }

        #pragma omp barrier
        #pragma omp single
        {
            std::stable_sort(points, points+nodes, XFirstPointCompare<RggPoint>);
        }

        for (VertexId node = node_start; node < node_end; node++)
        {
            SizeT x_index = points[node].x / threshold;
            SizeT y_index = points[node].y / threshold;
            SizeT block_index = x_index * row_length + y_index;
            #pragma omp atomic
                block_size[block_index]++;
        }

        #pragma omp barrier
        #pragma omp single
        {
            for (SizeT i=0; i<row_length * row_length; i++)
            if (block_size[i] != 0)
                blocks[i] = new VertexId[block_size[i]];
        }

        for (VertexId node = node_start; node < node_end; node++)
        {
            double co_x0 = points[node].x; //co_x[node];
            double co_y0 = points[node].y; //co_y[node];
            //RggPoint point(co_x0, co_y0, node);
            SizeT x_index = co_x0 / threshold;
            SizeT y_index = co_y0 / threshold;
            SizeT block_index = x_index * row_length + y_index;
            SizeT pos = 0;

            #pragma omp atomic capture
            {
                pos = block_length[block_index];
                block_length[block_index] += 1;
            }
            blocks[block_index][pos] = node;
        }

        #pragma omp barrier

        for (VertexId node = node_start; node < node_end; node++)
        {
            row_offsets[node] = counter;
            double co_x0 = points[node].x;
            double co_y0 = points[node].y;
            SizeT x_index = co_x0 / threshold;
            SizeT y_index = co_y0 / threshold;

            for (SizeT x1 = x_index-2; x1 <= x_index+2; x1++)
            for (SizeT y1 = y_index-2; y1 <= y_index+2; y1++)
            {
                if (x1 < 0 || y1 < 0 || x1 >= row_length || y1 >= row_length)
                    continue;

                SizeT block_index = x1*row_length + y1;
                VertexId *block = blocks[block_index];
                for (SizeT i = 0; i< block_length[block_index]; i++)
                {
                    VertexId peer = block[i];
                    if (node >= peer) continue;
                    double   co_x1 = points[peer].x;//co_x[peer];
                    double   co_y1 = points[peer].y;//co_y[peer];
                    double   dis_x = co_x0 - co_x1;
                    double   dis_y = co_y0 - co_y1;
                    if (fabs(dis_x) > threshold || fabs(dis_y) > threshold) continue;
                    double   dis   = SqrtSum(dis_x, dis_y);
                    if (dis > threshold) continue;

                    col_index[counter] = peer;
                    if (WITH_VALUES)
                    {
                        values[counter] = distribution(engine) * value_multipiler + value_min;
                    }
                    counter++;
                }
            }
        }
        offsets[thread_num+1] = counter;

        #pragma omp barrier
        #pragma omp single
        {
            offsets[0] = 0;
            for (int i=0; i<num_threads; i++)
                offsets[i+1] += offsets[i];
            edges = offsets[num_threads] * (undirected ? 2 : 1);
            coo = (EdgeTupleType*) malloc (sizeof(EdgeTupleType) * edges);
        }

        SizeT offset = offsets[thread_num] * (undirected ? 2 : 1);
        for (VertexId node = node_start; node < node_end; node++)
        {
            SizeT end_edge = (node != node_end-1 ? row_offsets[node+1] : counter );
            for (SizeT edge = row_offsets[node]; edge < end_edge; edge++)
            {
                VertexId peer = col_index[edge];
                if (undirected)
                {
                    EdgeTupleType *coo_p = coo + offset + edge*2;
                    coo_p -> row = node;
                    coo_p -> col = peer;
                    coo_p -> val = WITH_VALUES ? values[edge] : 1;
                    coo_p = coo_p + 1;
                    coo_p -> row = peer;
                    coo_p -> col = node;
                    coo_p -> val = WITH_VALUES ? values[edge] : 1;
                } else {
                    EdgeTupleType *coo_p = coo + offset + edge;
                    coo_p -> row = node;
                    coo_p -> col = peer;
                    coo_p -> val = WITH_VALUES ? values[edge] : 1;
                }
            }
        }

        col_index = NULL;
        values    = NULL;
    }

    SizeT counter = 0;
    for (SizeT i=0;  i < row_length * row_length; i++)
    if (block_size[i] != 0)
    {
        counter += block_length[i];
        if (blocks[i] != NULL) delete[] blocks[i]; blocks[i] = NULL;
    }

    char *out_file = NULL;
    graph.template FromCoo<WITH_VALUES, EdgeTupleType>(
        out_file, coo, nodes, edges, false, undirected, false, quiet);

    delete[] row_offsets; row_offsets = NULL;
    delete[] offsets    ; offsets     = NULL;
    delete[] points     ; points      = NULL;
    delete[] blocks     ; blocks      = NULL;
    delete[] block_size ; block_size  = NULL;
    delete[] block_length; block_length = NULL;
    delete[] col_index_ ; col_index_  = NULL;
    if (WITH_VALUES) { delete[] values_; values_ = NULL; }
    free(coo); coo=NULL;

    return 0;
}

} // namespace rgg
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
