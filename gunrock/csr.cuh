// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * csr.cuh
 *
 * @brief CSR (Compressed Sparse Row) Graph Data Structure
 */

#pragma once

#include <time.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/error_utils.cuh>

namespace gunrock {

/**
 * @brief CSR data structure which uses Compressed Sparse Row
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 */
template<typename VertexId, typename Value, typename SizeT>
struct Csr {
    SizeT nodes;     /**< Number of nodes in the graph. */
    SizeT edges;     /**< Number of edges in the graph. */
    SizeT out_nodes; /**< Number of nodes which have outgoing edges. */
    SizeT average_degree;

    VertexId *column_indices; /**< Column indices corresponding to all the non-zero values in the sparse matrix. */
    SizeT    *row_offsets;    /**< List of indices where each row of the sparse matrix starts. */
    Value    *edge_values;    /**< List of values attached to edges in the graph. */
    Value    *node_values;    /**< List of values attached to nodes in the graph. */

    Value average_edge_value;
    Value average_node_value;

    bool  pinned;        /**< Whether to use pinned memory */

    /**
     * @brief CSR Constructor
     *
     * @param[in] pinned Use pinned memory for CSR data structure
     * (default: do not use pinned memory)
     */
    Csr(bool pinned = false) {
        nodes = 0;
        edges = 0;
        average_degree = 0;
        average_edge_value = 0;
        average_node_value = 0;
        out_nodes = -1;
        row_offsets = NULL;
        column_indices = NULL;
        edge_values = NULL;
        node_values = NULL;
        this->pinned = pinned;
    }

    /**
     * @brief Allocate memory for CSR graph.
     *
     * @param[in] nodes Number of nodes in COO-format graph
     * @param[in] edges Number of edges in COO-format graph
     */
    template <bool LOAD_EDGE_VALUES, bool LOAD_NODE_VALUES>
    void FromScratch(SizeT nodes, SizeT edges) {
        this->nodes = nodes;
        this->edges = edges;

        if (pinned) {

            // Put our graph in pinned memory
            int flags = cudaHostAllocMapped;
            if (gunrock::util::GRError(
                        cudaHostAlloc((void **)&row_offsets,
                                      sizeof(SizeT) * (nodes + 1), flags),
                        "Csr cudaHostAlloc row_offsets failed", __FILE__, __LINE__))
                exit(1);
            if (gunrock::util::GRError(
                        cudaHostAlloc((void **)&column_indices,
                                      sizeof(VertexId) * edges, flags),
                        "Csr cudaHostAlloc column_indices failed",
                        __FILE__, __LINE__))
                exit(1);

            if (LOAD_NODE_VALUES) {
                if (gunrock::util::GRError(
                            cudaHostAlloc((void **)&node_values,
                                          sizeof(Value) * nodes, flags),
                            "Csr cudaHostAlloc node_values failed",
                            __FILE__, __LINE__))
                    exit(1);
            }

            if (LOAD_EDGE_VALUES) {
                if (gunrock::util::GRError(
                            cudaHostAlloc((void **)&edge_values,
                                          sizeof(Value) * edges, flags),
                            "Csr cudaHostAlloc edge_values failed",
                            __FILE__, __LINE__))
                    exit(1);
            }

        } else {

            // Put our graph in regular memory
            row_offsets = (SizeT*) malloc(sizeof(SizeT) * (nodes + 1));
            column_indices = (VertexId*) malloc(sizeof(VertexId) * edges);
            node_values = (LOAD_NODE_VALUES) ?
                          (Value*) malloc(sizeof(Value) * nodes) : NULL;
            edge_values = (LOAD_EDGE_VALUES) ?
                          (Value*) malloc(sizeof(Value) * edges) : NULL;
        }
    }

    /**
     *
     * @brief Store graph information into a file
     *
     */
    void WriteToFile(char  *file_name, SizeT v, SizeT e, SizeT *row,
                     VertexId *col, Value *edge_values = NULL) {
        std::ofstream fout(file_name);
        if (fout.is_open()) {
            fout.write(reinterpret_cast<const char*>(&v), sizeof(SizeT));
            fout.write(reinterpret_cast<const char*>(&e), sizeof(SizeT));
            fout.write(reinterpret_cast<const char*>(row), (v+1)*sizeof(SizeT));
            fout.write(reinterpret_cast<const char*>(col), e*sizeof(VertexId));
            if (edge_values != NULL) {
                fout.write(reinterpret_cast<const char*>(edge_values),
                           e * sizeof(Value));
            }
            fout.close();
        }
    }

    void WriteToLigraFile(char  *file_name, SizeT v, SizeT e, SizeT *row,
                     VertexId *col, Value *edge_values = NULL) {
        char adj_name[256];
        sprintf(adj_name, "%s.adj", file_name);
        printf("writing to ligra .adj file.\n");

        std::ofstream fout3(adj_name);
        if (fout3.is_open()) {
            fout3 << v << " " << v << " " << e << std::endl;
            for (int i = 0; i < v; ++i)
                fout3 << row[i] << std::endl;
            for (int i = 0; i < e; ++i)
                fout3 << col[i] << std::endl;
            if (edge_values != NULL) {
                for (int i = 0; i < e; ++i)
                    fout3 << edge_values[i] << std::endl;
            }
            fout3.close();
        }
    }

    /**
     *
     * @brief Read from stored row_offsets, column_indices arrays
     *
     */
    template <bool LOAD_EDGE_VALUES>
    void FromCsr(char *f_in, bool quiet=false) {
        if (!quiet)
        {
            printf("  Reading directly from stored binary CSR arrays ...\n");
        }
        time_t mark1 = time(NULL);

        std::ifstream input(f_in);
        SizeT v, e;
        input.read(reinterpret_cast<char*>(&v), sizeof(SizeT));
        input.read(reinterpret_cast<char*>(&e), sizeof(SizeT));

        FromScratch<LOAD_EDGE_VALUES, false>(v, e);

        input.read(reinterpret_cast<char*>(row_offsets), (v + 1)*sizeof(SizeT));
        input.read(reinterpret_cast<char*>(column_indices), e*sizeof(VertexId));
        if (LOAD_EDGE_VALUES) {
            input.read(reinterpret_cast<char*>(edge_values), e*sizeof(Value));
        }

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            printf("Done reading (%ds).\n", (int) (mark2 - mark1));
        }

        // compute out_nodes
        SizeT out_node = 0;
        for (SizeT node = 0; node < nodes; node++) {
            if (row_offsets[node + 1] - row_offsets[node] > 0) {
                ++out_node;
            }
        }
        out_nodes = out_node;
    }

    /**
     * @brief Build CSR graph from COO graph, sorted or unsorted
     *
     * @param[in] output_file Output file to dump the graph topology info
     * @param[in] coo Pointer to COO-format graph
     * @param[in] coo_nodes Number of nodes in COO-format graph
     * @param[in] coo_edges Number of edges in COO-format graph
     * @param[in] ordered_rows Are the rows sorted? If not, sort them.
     * @param[in] undirected Is the graph directed or not?
     * @param[in] reversed Is the graph reversed or not?
     * Default: Assume rows are not sorted.
     */
    template <bool LOAD_EDGE_VALUES, typename Tuple>
    void FromCoo(
        char  *output_file,
        Tuple *coo,
        SizeT coo_nodes,
        SizeT coo_edges,
        bool  ordered_rows = false,
        bool  undirected = false,
        bool  reversed = false,
        bool  quiet = false) {
        if (!quiet)
        {
            printf("  Converting %d vertices, %d directed edges (%s tuples) "
                   "to CSR format...\n", coo_nodes, coo_edges,
                   ordered_rows ? "ordered" : "unordered");
        }
        time_t mark1 = time(NULL);
        fflush(stdout);

        FromScratch<LOAD_EDGE_VALUES, false>(coo_nodes, coo_edges);

        // Sort COO by row
        if (!ordered_rows) {
            std::stable_sort(coo, coo + coo_edges, RowFirstTupleCompare<Tuple>);
        }

        Tuple *new_coo = (Tuple*) malloc(sizeof(Tuple) * coo_edges);
        SizeT real_edge = 0;
        if (coo[0].col != coo[0].row) {
            new_coo[0].row = coo[0].row;
            new_coo[0].col = coo[0].col;
            new_coo[0].val = coo[0].val;
            real_edge++;
        }
        for (int i = 0; i < coo_edges - 1; ++i) {
            if (((coo[i + 1].col != coo[i].col) ||
                    (coo[i + 1].row != coo[i].row)) &&
                    (coo[i + 1].col != coo[i + 1].row)) {
                new_coo[real_edge].col = coo[i + 1].col;
                new_coo[real_edge].row = coo[i + 1].row;
                new_coo[real_edge++].val = coo[i + 1].val;
            }
        }

        VertexId prev_row = -1;
        for (SizeT edge = 0; edge < real_edge; edge++) {

            VertexId current_row = new_coo[edge].row;

            // Fill in rows up to and including the current row
            for (VertexId row = prev_row + 1; row <= current_row; row++) {
                row_offsets[row] = edge;
            }
            prev_row = current_row;

            column_indices[edge] = new_coo[edge].col;
            if (LOAD_EDGE_VALUES) {
                new_coo[edge].Val(edge_values[edge]);
            }
        }

        // Fill out any trailing edgeless nodes (and the end-of-list element)
        for (VertexId row = prev_row + 1; row <= nodes; row++) {
            row_offsets[row] = real_edge;
        }
        edges = real_edge;

        time_t mark2 = time(NULL);
        if (!quiet) {
            printf("Done converting (%ds).\n", (int)(mark2 - mark1));
        }

        // Write offsets, indices, node, edges etc. into file
        if (LOAD_EDGE_VALUES) {
            WriteToFile(output_file, nodes, edges,
                        row_offsets, column_indices, edge_values);
            //WriteToLigraFile(output_file, nodes, edges,
            //            row_offsets, column_indices, edge_values);
        } else {
            WriteToFile(output_file, nodes, edges,
                        row_offsets, column_indices);
        }

        if (new_coo) free(new_coo);

        // Compute out_nodes
        SizeT out_node = 0;
        for (SizeT node = 0; node < nodes; node++) {
            if (row_offsets[node + 1] - row_offsets[node] > 0) {
                ++out_node;
            }
        }
        out_nodes = out_node;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Print log-scale degree histogram of the graph.
     */
    void PrintHistogram() {
        fflush(stdout);

        // Initialize
        int log_counts[32];
        for (int i = 0; i < 32; i++) {
            log_counts[i] = 0;
        }

        // Scan
        int max_log_length = -1;
        for (VertexId i = 0; i < nodes; i++) {

            SizeT length = row_offsets[i + 1] - row_offsets[i];

            int log_length = -1;
            while (length > 0) {
                length >>= 1;
                log_length++;
            }
            if (log_length > max_log_length) {
                max_log_length = log_length;
            }

            log_counts[log_length + 1]++;
        }
        printf("\nDegree Histogram (%lld vertices, %lld edges):\n",
               (long long) nodes, (long long) edges);
        printf("    Degree   0: %d (%.2f%%)\n", log_counts[0],
               (float) log_counts[0] * 100.0 / nodes);
        for (int i = 0; i < max_log_length + 1; i++) {
            printf("    Degree 2^%i: %d (%.2f%%)\n", i, log_counts[i + 1],
                   (float) log_counts[i + 1] * 100.0 / nodes);
        }
        printf("\n");
        fflush(stdout);
    }


    /**
     * @brief Display CSR graph to console
     */
    void DisplayGraph(bool with_edge_value = false) {
        SizeT displayed_node_num = (nodes > 40) ? 40 : nodes;
        printf("First %d nodes's neighbor list of the input graph:\n",
               displayed_node_num);
        for (SizeT node = 0; node < displayed_node_num; node++) {
            util::PrintValue(node);
            printf(":");
            for (SizeT edge = row_offsets[node];
                    edge < row_offsets[node + 1];
                    edge++) {
                printf("[");
                util::PrintValue(column_indices[edge]);
                if (with_edge_value && edge_values != NULL) {
                    printf(",");
                    util::PrintValue(edge_values[edge]);
                }
                printf("], ");
            }
            printf("\n");
        }
    }

    bool CheckValue() {
        for (SizeT node = 0; node < nodes; ++node) {
            for (SizeT edge = row_offsets[node];
                    edge < row_offsets[node + 1];
                    ++edge) {
                int src_node = node;
                int dst_node = column_indices[edge];
                int edge_value = edge_values[edge];
                for (SizeT r_edge = row_offsets[dst_node];
                        r_edge < row_offsets[dst_node + 1];
                        ++r_edge) {
                    if (column_indices[r_edge] == src_node) {
                        if (edge_values[r_edge] != edge_value)
                            return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief Find node with largest neighbor list
     */
    int GetNodeWithHighestDegree(int& max_degree) {
        int degree = 0;
        int src = 0;
        for (SizeT node = 0; node < nodes; node++) {
            if (row_offsets[node + 1] - row_offsets[node] > degree) {
                degree = row_offsets[node + 1] - row_offsets[node];
                src = node;
            }
        }
        max_degree = degree;
        return src;
    }

    /**
     * @brief Display the neighbor list of a given node
     */
    void DisplayNeighborList(VertexId node) {
        if (node < 0 || node >= nodes) return;
        for (SizeT edge = row_offsets[node];
                edge < row_offsets[node + 1];
                edge++) {
            util::PrintValue(column_indices[edge]);
            printf(", ");
        }
        printf("\n");
    }

    /**
     * @brief Get the average degree of all the nodes in graph
     */
    SizeT GetAverageDegree() {
        if (average_degree == 0) {
            double mean = 0, count = 0;
            for (SizeT node = 0; node < nodes; ++node) {
                count += 1;
                mean += (row_offsets[node+1]-row_offsets[node]-mean)/count;
            }
            average_degree = static_cast<SizeT>(mean);
        }
        return average_degree;
    }

    /**
     * @brief Get the average node value in graph
     */
    Value GetAverageNodeValue() {
        if (abs(average_node_value - 0) < 0.001 && node_values != NULL) {
            double mean = 0, count = 0;
            for (SizeT node = 0; node < nodes; ++node) {
                if (node_values[node] < UINT_MAX) {
                    count += 1;
                    mean += (node_values[node] - mean) / count;
                }
            }
            average_node_value = static_cast<Value>(mean);
        }
        return average_node_value;
    }

    /**
     * @brief Get the average edge value in graph
     */
    Value GetAverageEdgeValue() {
        if (abs(average_edge_value - 0) < 0.001 && edge_values != NULL) {
            double mean = 0, count = 0;
            for (SizeT edge = 0; edge < edges; ++edge) {
                if (edge_values[edge] < UINT_MAX) {
                    count += 1;
                    mean += (edge_values[edge] - mean) / count;
                }
            }
            average_edge_value = static_cast<Value>(mean);
        }
        return average_edge_value;
    }

    /**@}*/

    /**
     * @brief Deallocates CSR graph
     */
    void Free() {
        if (row_offsets) {
            if (pinned) {
                gunrock::util::GRError(cudaFreeHost(row_offsets),
                                       "Csr cudaFreeHost row_offsets failed",
                                       __FILE__, __LINE__);
            } else {
                free(row_offsets);
            }
            row_offsets = NULL;
        }
        if (column_indices) {
            if (pinned) {
                gunrock::util::GRError(cudaFreeHost(column_indices),
                                       "Csr cudaFreeHost column_indices failed",
                                       __FILE__, __LINE__);
            } else {
                free(column_indices);
            }
            column_indices = NULL;
        }
        if (edge_values) { free (edge_values); edge_values = NULL; }
        if (node_values) { free (node_values); node_values = NULL; }

        nodes = 0;
        edges = 0;
    }

    /**
     * @brief CSR destructor
     */
    ~Csr() {
        Free();
    }
};

} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
