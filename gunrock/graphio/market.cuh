// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * market.cuh
 *
 * @brief MARKET Graph Construction Routines
 */


#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>

#include <gunrock/graphio/utils.cuh>

namespace gunrock {
namespace graphio {


/**
 * @brief Reads a MARKET graph from an input-stream into a CSR sparse format
 *
 * Here is an example of the matrix market format
 * +----------------------------------------------+
 * |%%MatrixMarket matrix coordinate real general | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comment lines
 * |%                                             | <--+
 * |  M N L                                       | <--- rows, columns, entries
 * |  I1 J1 A(I1, J1)                             | <--+
 * |  I2 J2 A(I2, J2)                             |    |
 * |  I3 J3 A(I3, J3)                             |    |-- L lines
 * |     . . .                                    |    |
 * |  IL JL A(IL, JL)                             | <--+
 * +----------------------------------------------+
 *
 * Indices are 1-based i.2. A(1,1) is the first element.
 *
 * @param[in] f_in          Input MARKET graph file
 * @param[in] csr_graph     Csr graph object to store the graph data
 * @param[in] undirected    Is the graph undirected or not?
 *
 * \return If there is any File I/O error along the way.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int ReadMarketStream(
    FILE *f_in,
    Csr<VertexId, Value, SizeT> &csr_graph,
    bool undirected,
    bool reversed)
{
    typedef Coo<VertexId, Value> EdgeTupleType;

    SizeT edges_read = -1;
    SizeT nodes = 0;
    SizeT edges = 0;
    EdgeTupleType *coo = NULL;          // read in COO format

    time_t mark0 = time(NULL);
    printf("  Parsing MARKET COO format ");
    fflush(stdout);

    char line[1024];

    bool ordered_rows = true;

    while(true) {

        if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
            break;
        }

        if (line[0] == '%') {

            // Comment

        } else if (edges_read == -1) {

            // Problem description
            long long ll_nodes_x, ll_nodes_y, ll_edges;
            if (sscanf(line, "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges) != 3) {
                fprintf(stderr, "Error parsing MARKET graph: invalid problem description\n");
                return -1;
            }

            if (ll_nodes_x != ll_nodes_y) {
                fprintf(stderr, "Error parsing MARKET graph: not square (%lld, %lld)\n", ll_nodes_x, ll_nodes_y);
                return -1;
            }

            nodes = ll_nodes_x;
            edges = (undirected) ? ll_edges * 2 : ll_edges;

            printf(" (%lld nodes, %lld directed edges)... ",
                   (unsigned long long) ll_nodes_x, (unsigned long long) ll_edges);
            fflush(stdout);

            // Allocate coo graph
            coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * edges);

            edges_read++;

        } else {

            // Edge description (v -> w)
            if (!coo) {
                fprintf(stderr, "Error parsing MARKET graph: invalid format\n");
                return -1;
            }
            if (edges_read >= edges) {
                fprintf(stderr, "Error parsing MARKET graph: encountered more than %d edges\n", edges);
                if (coo) free(coo);
                return -1;
            }

            long long ll_row, ll_col;
            if (sscanf(line, "%lld %lld", &ll_col, &ll_row) != 2) {
                fprintf(stderr,
                        "Error parsing MARKET graph: badly formed edge\n"
                        /*, edges */
                        /* JDO commented this out; it wasn't used and
                         * generated a warning */);
                if (coo) free(coo);
                return -1;
            }

            if (reversed && !undirected) {
                coo[edges_read].col = ll_row - 1;   // zero-based array
                coo[edges_read].row = ll_col - 1;   // zero-based array
                ordered_rows = false;
            } else {
                coo[edges_read].row = ll_row - 1;   // zero-based array
                coo[edges_read].col = ll_col - 1;   // zero-based array
            }

            edges_read++;

            if (undirected) {
                // Go ahead and insert reverse edge
                coo[edges_read].row = ll_col - 1;       // zero-based array
                coo[edges_read].col = ll_row - 1;       // zero-based array

                ordered_rows = false;
                edges_read++;
            }
        }
    }

    if (coo == NULL) {
        fprintf(stderr, "No graph found\n");
        return -1;
    }

    if (edges_read != edges) {
        fprintf(stderr, "Error parsing MARKET graph: only %d/%d edges read\n", edges_read, edges);
        if (coo) free(coo);
        return -1;
    }

    time_t mark1 = time(NULL);
    printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
    fflush(stdout);

    // Convert COO to CSR
    csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, edges, ordered_rows);
    free(coo);

    fflush(stdout);

    return 0;
}

/**
 * \defgroup PublicInterface Gunrock Public Interface
 * @{
 */

/**
 * @brief Loads a MARKET-formatted CSR graph from the specified file.
 *
 * @param[in] mm_filename Graph file name, if empty, then it is loaded from stdin.
 * @param[in] csr_graph Reference to CSR graph object. @see Csr
 * @param[in] undirected Is the graph undirected or not?
 *
 * \return If there is any File I/O error along the way. 0 for no error.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildMarketGraph(
    char *mm_filename,
    Csr<VertexId, Value, SizeT> &csr_graph,
    bool undirected,
    bool reversed)
{
    if (mm_filename == NULL) {

        // Read from stdin
        printf("Reading from stdin:\n");
        if (ReadMarketStream<LOAD_VALUES>(stdin, csr_graph, undirected, reversed) != 0) {
            return -1;
        }

    } else {

        // Read from file
        FILE *f_in = fopen(mm_filename, "r");
        if (f_in) {
            printf("Reading from %s:\n", mm_filename);
            if (ReadMarketStream<LOAD_VALUES>(f_in, csr_graph, undirected, reversed) != 0) {
                fclose(f_in);
                return -1;
            }
            fclose(f_in);
        } else {
            perror("Unable to open file");
            return -1;
        }
    }

    return 0;
}

/**@}*/

} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
