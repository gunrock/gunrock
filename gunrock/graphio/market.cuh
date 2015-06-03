// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

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
#include <libgen.h>
#include <iostream>

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
    char *output_file,
    Csr<VertexId, Value, SizeT> &csr_graph,
    bool undirected,
    bool reversed)
{
    typedef Coo<VertexId, Value> EdgeTupleType;

    SizeT edges_read = -1;
    SizeT nodes = 0;
    SizeT edges = 0;
    EdgeTupleType *coo = NULL; // read in COO format

    time_t mark0 = time(NULL);
    printf("  Parsing MARKET COO format");
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
            if (sscanf(line, "%lld %lld %lld",
                       &ll_nodes_x, &ll_nodes_y, &ll_edges) != 3) {
                fprintf(stderr, "Error parsing MARKET graph:"
                        " invalid problem description.\n");
                return -1;
            }

            if (ll_nodes_x != ll_nodes_y) {
                fprintf(stderr,
                        "Error parsing MARKET graph: not square (%lld, %lld)\n",
                        ll_nodes_x, ll_nodes_y);
                return -1;
            }

            nodes = ll_nodes_x;
            edges = (undirected) ? ll_edges * 2 : ll_edges;

            printf(" (%lld nodes, %lld directed edges)... ",
                   (unsigned long long) ll_nodes_x,
                   (unsigned long long) ll_edges);
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
              fprintf(stderr,
                      "Error parsing MARKET graph:"
                      "encountered more than %d edges\n",
                      edges);
              if (coo) free(coo);
              return -1;
            }

            long long ll_row, ll_col, ll_value;
            int num_input;
            if (LOAD_VALUES) {
                if ((num_input = sscanf(
                         line, "%lld %lld %lld",
                         &ll_col, &ll_row, &ll_value)) < 2) {
                    fprintf(stderr,
                            "Error parsing MARKET graph: badly formed edge\n");
                    if (coo) free(coo);
                    return -1;
                } else if (num_input == 2) {
                    ll_value = 1;
                }
            } else {
                if (sscanf(line, "%lld %lld", &ll_col, &ll_row) != 2) {
                    fprintf(stderr,
                            "Error parsing MARKET graph: badly formed edge\n");
                    if (coo) free(coo);
                    return -1;
                }
            }

            if (LOAD_VALUES) {
                coo[edges_read].val = ll_value;
            }
            if (reversed && !undirected) {
                coo[edges_read].col = ll_row - 1;   // zero-based array
                coo[edges_read].row = ll_col - 1;   // zero-based array
                ordered_rows = false;
            } else {
                coo[edges_read].row = ll_row - 1;   // zero-based array
                coo[edges_read].col = ll_col - 1;   // zero-based array
                ordered_rows = false;
            }

            edges_read++;

            if (undirected) {
                // Go ahead and insert reverse edge
                coo[edges_read].row = ll_col - 1;       // zero-based array
                coo[edges_read].col = ll_row - 1;       // zero-based array

                if (LOAD_VALUES) {
                    coo[edges_read].val = ll_value;
                }

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
        fprintf(stderr,
                "Error parsing MARKET graph: only %d/%d edges read\n",
                edges_read, edges);
        if (coo) free(coo);
        return -1;
    }

    time_t mark1 = time(NULL);
    printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
    fflush(stdout);

    // Convert COO to CSR
    csr_graph.template FromCoo<LOAD_VALUES>(output_file, coo,
                                            nodes, edges, ordered_rows,
                                            undirected, reversed);

    free(coo);

    fflush(stdout);

    return 0;
}

/**
 * @brief Read csr arrays directly instead of transfer from coo format
 *
 */
template <bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int ReadCsrArrays(
    char *f_in,
    Csr<VertexId, Value, SizeT> &csr_graph,
    bool undirected,
    bool reversed)
{
    csr_graph.template FromCsr<LOAD_VALUES>(f_in, undirected, reversed);
    return 0;
}


/**
 * \defgroup PublicInterface Gunrock Public Interface
 * @{
 */

/**
 * @brief Loads a MARKET-formatted CSR graph from the specified file.
 *
 * @param[in] mm_filename Graph file name, if empty, it is loaded from stdin.
 * @param[in] output_file Output file to store the computed graph topology info.
 * @param[in] csr_graph Reference to CSR graph object. @see Csr
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed Is the graph reversed or not?
 *
 * \return If there is any File I/O error along the way. 0 for no error.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildMarketGraph(
    char *mm_filename,
    char *output_file,
    Csr<VertexId, Value, SizeT> &csr_graph,
    bool undirected,
    bool reversed)
{
    FILE *_file = fopen(output_file, "r");
    if (_file)
    {
        fclose(_file);
        if (ReadCsrArrays<LOAD_VALUES>(
                output_file, csr_graph, undirected, reversed) != 0) {
            return -1;
        }
    }
    else {
        if (mm_filename == NULL) {
            // Read from stdin
            printf("Reading from stdin:\n");
            if (ReadMarketStream<LOAD_VALUES>(
                    stdin, output_file, csr_graph, undirected, reversed) != 0) {
                return -1;
            }
        }
        else {
            // Read from file
            FILE *f_in = fopen(mm_filename, "r");
            if (f_in) {
                printf("Reading from %s:\n", mm_filename);
                if (ReadMarketStream<LOAD_VALUES>(
                        f_in, output_file, csr_graph,
                        undirected, reversed) != 0) {
                    fclose(f_in);
                    return -1;
                }
                fclose(f_in);
            } else  {
                perror("Unable to open file");
                return -1;
            }
        }
    }
    return 0;
}

/**
 * @brief read in graph function read in graph according to it's type
 *
 */
template <bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildMarketGraph(
    char *file_in,
    Csr<VertexId, Value, SizeT> &graph,
    bool undirected,
    bool reversed)
{
    // seperate the graph path and the file name
    char *temp1 = strdup(file_in);
    char *temp2 = strdup(file_in);
    char *file_path = dirname (temp1);
    char *file_name = basename(temp2);

    if (undirected)
    {
        char ud[256];
        sprintf(ud, "%s/.%s_undirected_csr", file_path, file_name);
        if (BuildMarketGraph<true>(file_in, ud, graph, true, false) != 0)
            return 1;
    }
    else if (!undirected && reversed)
    {
        char rv[256];
        sprintf(rv, "%s/.%s_reversed_csr", file_path, file_name);
        if (BuildMarketGraph<true>(file_in, rv, graph, false, true) != 0)
            return 1;
    }
    else if (!undirected && !reversed)
    {
        char nr[256];
        sprintf(nr, "%s/.%s_nonreversed_csr", file_path, file_name);
        if (BuildMarketGraph<true>(file_in, nr, graph, false, false) != 0)
            return 1;
    }
    else
    {
        fprintf(stderr, "Unspecified Graph Type.\n");
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
