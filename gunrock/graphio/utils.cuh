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

} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
