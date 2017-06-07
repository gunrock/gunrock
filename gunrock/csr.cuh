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
#include <omp.h>

#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/sort_omp.cuh>
#include <gunrock/coo.cuh>

namespace gunrock {

/**
 * @brief CSR data structure which uses Compressed Sparse Row
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 *
 * @tparam VertexId Vertex identifier.
 * @tparam Value Associated value type.
 * @tparam SizeT Graph size type.
 */
template<typename VertexId, typename SizeT, typename Value>
struct Csr
{
    SizeT nodes;            // Number of nodes in the graph
    SizeT edges;            // Number of edges in the graph
    SizeT out_nodes;        // Number of nodes which have outgoing edges
    SizeT average_degree;   // Average vertex degrees
    float stddev_degree;    // Degree standard deviation

    VertexId *column_indices; // Column indices corresponding to all the
    // non-zero values in the sparse matrix
    SizeT    *row_offsets;    // List of indices where each row of the
    // sparse matrix starts
    Value    *edge_values;    // List of values attached to edges in the graph
    Value    *node_values;    // List of values attached to nodes in the graph

    Value average_edge_value;
    Value average_node_value;

    bool  pinned;  // Whether to use pinned memory

    /**
     * @brief CSR Constructor
     *
     * @param[in] pinned Use pinned memory for CSR data structure
     * (default: do not use pinned memory)
     */
    Csr(bool pinned = false)
    {
        nodes = 0;
        edges = 0;
        average_degree = 0;
        stddev_degree = 0.0f;
        average_edge_value = 0;
        average_node_value = 0;
        out_nodes = -1;
        row_offsets = NULL;
        column_indices = NULL;
        edge_values = NULL;
        node_values = NULL;
        this->pinned = pinned;
    }

    void FromCsr(Csr<VertexId, SizeT, Value> &source)
    {
        nodes = source.nodes;
        edges = source.edges;
        average_degree = source.average_degree;
        average_edge_value = source.average_edge_value;
        average_node_value = source.average_node_value;
        out_nodes = source.out_nodes;
        if (source.row_offsets == NULL)
        {
            row_offsets = NULL;
        } else {
            row_offsets = (SizeT*) malloc(sizeof(SizeT) * (source.nodes + 1));
            memcpy(row_offsets, source.row_offsets, sizeof(SizeT) * (source.nodes + 1));
        }
        if (source.column_indices == NULL)
        {
            column_indices = NULL;
        } else {
            column_indices = (VertexId*) malloc(sizeof(VertexId) * source.edges); 
            memcpy(column_indices, source.column_indices, sizeof(VertexId) * source.edges);
        }
        if (source.edge_values == NULL)
        {
            edge_values = NULL;
        } else {
            edge_values = (Value*) malloc(sizeof(Value) * source.edges);
            memcpy(edge_values, source.edge_values, sizeof(Value) * source.edges);
        }
        if (source.node_values == NULL)
        {
            node_values = NULL;
        } else {
            node_values = (Value*) malloc(sizeof(Value) * source.nodes);
            memcpy(node_values, source.node_values, sizeof(Value) * source.nodes);
        } 
    }

    
    template <typename Tuple>
    void CsrToCsc(Csr<VertexId, SizeT, Value> &target, 
            Csr<VertexId, SizeT, Value> &source)
    {
        target.nodes = source.nodes;
        target.edges = source.edges;
        target.average_degree = source.average_degree;
        target.average_edge_value = source.average_edge_value;
        target.average_node_value = source.average_node_value;
        target.out_nodes = source.out_nodes;
        {
            Tuple *coo = (Tuple*)malloc(sizeof(Tuple) * source.edges);
            int idx = 0;
            for (int i = 0; i < source.nodes; ++i)
            {
                for (int j = source.row_offsets[i]; j < source.row_offsets[i+1]; ++j)
                {
                    coo[idx].row = source.column_indices[j];
                    coo[idx].col = i;
                    coo[idx++].val = (source.edge_values == NULL) ? 0 : source.edge_values[j];
                }
            }
            if (source.edge_values == NULL)
                target.template FromCoo<false>(NULL, coo, nodes, edges);
            else
                target.template FromCoo<true>(NULL, coo, nodes, edges);
            free(coo);
        }
    }

    /**
     * @brief Allocate memory for CSR graph.
     *
     * @tparam LOAD_EDGE_VALUES
     * @tparam LOAD_NODE_VALUES
     *
     * @param[in] nodes Number of nodes in COO-format graph
     * @param[in] edges Number of edges in COO-format graph
     */
    template <bool LOAD_EDGE_VALUES, bool LOAD_NODE_VALUES>
    void FromScratch(SizeT nodes, SizeT edges)
    {
        this->nodes = nodes;
        this->edges = edges;

        if (pinned)
        {
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

            if (LOAD_NODE_VALUES)
            {
                if (gunrock::util::GRError(
                            cudaHostAlloc((void **)&node_values,
                                          sizeof(Value) * nodes, flags),
                            "Csr cudaHostAlloc node_values failed",
                            __FILE__, __LINE__))
                    exit(1);
            }

            if (LOAD_EDGE_VALUES)
            {
                if (gunrock::util::GRError(
                            cudaHostAlloc((void **)&edge_values,
                                          sizeof(Value) * edges, flags),
                            "Csr cudaHostAlloc edge_values failed",
                            __FILE__, __LINE__))
                    exit(1);
            }

        }
        else
        {
            // Put our graph in regular memory
            row_offsets = (SizeT*) malloc(sizeof(SizeT) * (nodes + 1));
            column_indices = (VertexId*) malloc(sizeof(VertexId) * edges);
            node_values = (LOAD_NODE_VALUES) ?
                          (Value*) malloc(sizeof(Value) * nodes) : NULL;
            edge_values = (LOAD_EDGE_VALUES) ?
                          (Value*) malloc(sizeof(Value) * edges) : NULL;
        }
    }

    void WriteBinary_SM(
	char *file_name,
	SizeT v,
	Value *labels)
    {
	std::ofstream fout(file_name);
	if(fout.is_open())
	{
	    fout.write(reinterpret_cast<const char*>(labels), v*sizeof(Value));
	    fout.close();
	}
    }
    /**
     *
     * @brief Store graph information into a file.
     *
     * @param[in] file_name Original graph file path and name.
     * @param[in] v Number of vertices in input graph.
     * @param[in] e Number of edges in input graph.
     * @param[in] row Row-offsets array store row pointers.
     * @param[in] col Column-indices array store destinations.
     * @param[in] edge_values Per edge weight values associated.
     *
     */
    void WriteBinary(
        char  *file_name,
        SizeT v,
        SizeT e,
        SizeT *row,
        VertexId *col,
        Value *edge_values = NULL)
    {
        std::ofstream fout(file_name);
        if (fout.is_open())
        {
            fout.write(reinterpret_cast<const char*>(&v), sizeof(SizeT));
            fout.write(reinterpret_cast<const char*>(&e), sizeof(SizeT));
            fout.write(reinterpret_cast<const char*>(row), (v + 1)*sizeof(SizeT));
            fout.write(reinterpret_cast<const char*>(col), e * sizeof(VertexId));
            if (edge_values != NULL)
            {
                fout.write(reinterpret_cast<const char*>(edge_values),
                           e * sizeof(Value));
            }
            fout.close();
        }
    }

    /*
     * @brief Write human-readable CSR arrays into 3 files.
     * Can be easily used for python interface.
     *
     * @param[in] file_name Original graph file path and name.
     * @param[in] v Number of vertices in input graph.
     * @param[in] e Number of edges in input graph.
     * @param[in] row_offsets Row-offsets array store row pointers.
     * @param[in] col_indices Column-indices array store destinations.
     * @param[in] edge_values Per edge weight values associated.
     */
    void WriteCSR(
        char *file_name,
        SizeT v, SizeT e,
        SizeT    *row_offsets,
        VertexId *col_indices,
        Value    *edge_values = NULL)
    {
        std::cout << file_name << std::endl;
        char rows[256], cols[256], vals[256];

        sprintf(rows, "%s.rows", file_name);
        sprintf(cols, "%s.cols", file_name);
        sprintf(vals, "%s.vals", file_name);

        std::ofstream rows_output(rows);
        if (rows_output.is_open())
        {
            std::copy(row_offsets, row_offsets + v + 1,
                      std::ostream_iterator<SizeT>(rows_output, "\n"));
            rows_output.close();
        }

        std::ofstream cols_output(cols);
        if (cols_output.is_open())
        {
            std::copy(col_indices, col_indices + e,
                      std::ostream_iterator<VertexId>(cols_output, "\n"));
            cols_output.close();
        }

        if (edge_values != NULL)
        {
            std::ofstream vals_output(vals);
            if (vals_output.is_open())
            {
                std::copy(edge_values, edge_values + e,
                          std::ostream_iterator<Value>(vals_output, "\n"));
                vals_output.close();
            }
        }
    }

    /*
     * @brief Write Ligra input CSR arrays into .adj file.
     * Can be easily used for python interface.
     *
     * @param[in] file_name Original graph file path and name.
     * @param[in] v Number of vertices in input graph.
     * @param[in] e Number of edges in input graph.
     * @param[in] row Row-offsets array store row pointers.
     * @param[in] col Column-indices array store destinations.
     * @param[in] edge_values Per edge weight values associated.
     * @param[in] quiet Don't print out anything.
     */
    void WriteToLigraFile(
        const char  *file_name,
        SizeT v, SizeT e,
        SizeT *row,
        VertexId *col,
        Value *edge_values = NULL,
        bool quiet = false)
    {
        char adj_name[256];
        sprintf(adj_name, "%s.adj", file_name);
        if (!quiet)
        {
            printf("writing to ligra .adj file.\n");
        }

        std::ofstream fout3(adj_name);
        if (fout3.is_open())
        {
            fout3 << "AdjacencyGraph" << std::endl << v << std::endl << e << std::endl;
            for (int i = 0; i < v; ++i)
                fout3 << row[i] << std::endl;
            for (int i = 0; i < e; ++i)
                fout3 << col[i] << std::endl;
            if (edge_values != NULL)
            {
                for (int i = 0; i < e; ++i)
                    fout3 << edge_values[i] << std::endl;
            }
            fout3.close();
        }
    }

    void WriteToMtxFile(
        const char  *file_name,
        SizeT v, SizeT e,
        SizeT *row,
        VertexId *col,
        Value *edge_values = NULL,
        bool quiet = false)
    {
        char adj_name[256];
        sprintf(adj_name, "%s.mtx", file_name);
        if (!quiet)
        {
            printf("writing to .mtx file.\n");
        }

        std::ofstream fout3(adj_name);
        if (fout3.is_open())
        {
            fout3 << v << " " << v << " " << e << std::endl;
            for (int i = 0; i < v; ++i) {
                SizeT begin = row[i];
                SizeT end = row[i+1];
                for (int j = begin; j < end; ++j) {
                    fout3 << col[j]+1 << " " << i+1;
                    if (edge_values != NULL)
                    {
                        fout3 << " " << edge_values[j] << std::endl;
                    }
                    else
                    {
                        fout3 << " " << rand() % 64 << std::endl;
                    }
                }
            }
            fout3.close();
        }
    }


    /**
     * @brief Read from stored row_offsets, column_indices arrays.
     *
     * @tparam LOAD_EDGE_VALUES Whether or not to load edge values.
     *
     * @param[in] f_in Input file name.
     * @param[in] quiet Don't print out anything.
     */
    template <bool LOAD_EDGE_VALUES>
    void FromCsr(char *f_in, bool quiet = false)
    {
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
        input.read(reinterpret_cast<char*>(column_indices), e * sizeof(VertexId));
        if (LOAD_EDGE_VALUES)
        {
            input.read(reinterpret_cast<char*>(edge_values), e * sizeof(Value));
        }

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            printf("Done reading (%ds).\n", (int) (mark2 - mark1));
        }

        // compute out_nodes
        SizeT out_node = 0;
        for (SizeT node = 0; node < nodes; node++)
        {
            if (row_offsets[node + 1] - row_offsets[node] > 0)
            {
                ++out_node;
            }
        }
        out_nodes = out_node;
    }

    /**
     * @brief (Specific for SM) Read from stored row_offsets, column_indices arrays.
     *
     * @tparam LOAD_NODE_VALUES Whether or not to load node values.
     *
     * @param[in] f_in Input graph file name.
     * @param[in] f_label Input label file name.
     * @param[in] quiet Don't print out anything.
     */
    template <bool LOAD_NODE_VALUES>
    void FromCsr_SM(char *f_in, char *f_label, bool quiet = false)
    {
        if (!quiet)
        {
            printf("  Reading directly from stored binary CSR arrays ...\n");
	    if(LOAD_NODE_VALUES)
                printf("  Reading directly from stored binary label arrays ...\n");
        }
        time_t mark1 = time(NULL);

        std::ifstream input(f_in);
        std::ifstream input_label(f_label);

        SizeT v, e;
        input.read(reinterpret_cast<char*>(&v), sizeof(SizeT));
        input.read(reinterpret_cast<char*>(&e), sizeof(SizeT));

        FromScratch<false, LOAD_NODE_VALUES>(v, e);

        input.read(reinterpret_cast<char*>(row_offsets), (v + 1)*sizeof(SizeT));
        input.read(reinterpret_cast<char*>(column_indices), e * sizeof(VertexId));
        if (LOAD_NODE_VALUES)
        {
            input_label.read(reinterpret_cast<char*>(node_values), v * sizeof(Value));
        }
	    for(int i=0; i<v; i++) printf("%lld ", (long long)node_values[i]); printf("\n");

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            printf("Done reading (%ds).\n", (int) (mark2 - mark1));
        }

        // compute out_nodes
        SizeT out_node = 0;
        for (SizeT node = 0; node < nodes; node++)
        {
            if (row_offsets[node + 1] - row_offsets[node] > 0)
            {
                ++out_node;
            }
        }
        out_nodes = out_node;
    }

    template<bool LOAD_NODE_VALUES>
    void FromLabels(
	char *output_file,
	Value *labels,
	SizeT nodes,
	bool quiet = false)
    {
	    if(!quiet) printf("  Converting the labels of %lld vertices to binary format...\n", 
				(long long)nodes);
	    if(LOAD_NODE_VALUES) WriteBinary_SM(output_file, nodes, labels);
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
     * @param[in] quiet Don't print out anything.
     *
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
        bool  quiet = false)
    {
        if (!quiet)
        {
            printf("  Converting %lld vertices, %lld directed edges (%s tuples) "
                   "to CSR format...\n", 
                    (long long)coo_nodes, (long long)coo_edges,
                   ordered_rows ? "ordered" : "unordered");
        }

        time_t mark1 = time(NULL);
        fflush(stdout);

        FromScratch<LOAD_EDGE_VALUES, false>(coo_nodes, coo_edges);

        // Sort COO by row
        if (!ordered_rows)
        {
            util::omp_sort(coo, coo_edges, RowFirstTupleCompare<Tuple>);
        }

        SizeT *edge_offsets = NULL;
        SizeT *edge_counts  = NULL;
        #pragma omp parallel
        {
            int num_threads  = omp_get_num_threads();
            int thread_num   = omp_get_thread_num();
            if (thread_num == 0)
            {
                edge_offsets = new SizeT[num_threads+1];
                edge_counts  = new SizeT[num_threads+1];
            }
            #pragma omp barrier

            SizeT edge_start = (long long)(coo_edges) * thread_num / num_threads;
            SizeT edge_end   = (long long)(coo_edges) * (thread_num + 1) / num_threads;
            SizeT node_start = (long long)(coo_nodes) * thread_num / num_threads;
            SizeT node_end   = (long long)(coo_nodes) * (thread_num + 1) / num_threads;
            Tuple *new_coo   = (Tuple*) malloc (sizeof(Tuple) * (edge_end - edge_start));
            SizeT edge       = edge_start;
            SizeT new_edge   = 0;
            for (edge = edge_start; edge < edge_end; edge++)
            {
                VertexId col = coo[edge].col;
                VertexId row = coo[edge].row;
                if ((col != row) && (edge == 0 || col != coo[edge - 1].col || row != coo[edge - 1].row))
                {
                    new_coo[new_edge].col = col;
                    new_coo[new_edge].row = row;
                    new_coo[new_edge].val = coo[edge].val;
                    new_edge++;
                }
            }
            edge_counts[thread_num] = new_edge;
            for (VertexId node = node_start; node < node_end; node++)
                row_offsets[node] = -1;

            #pragma omp barrier
            #pragma omp single
            {
                edge_offsets[0] = 0;
                for (int i = 0; i < num_threads; i++)
                    edge_offsets[i + 1] = edge_offsets[i] + edge_counts[i];
                //util::cpu_mt::PrintCPUArray("edge_offsets", edge_offsets, num_threads+1);
                row_offsets[0] = 0;
            }

            SizeT edge_offset = edge_offsets[thread_num];
            VertexId first_row = new_edge > 0 ? new_coo[0].row : -1;
            //VertexId last_row = new_edge > 0? new_coo[new_edge-1].row : -1;
            SizeT pointer = -1;
            for (edge = 0; edge < new_edge; edge++)
            {
                SizeT edge_  = edge + edge_offset;
                VertexId row = new_coo[edge].row;
                row_offsets[row + 1] = edge_ + 1;
                if (row == first_row) pointer = edge_ + 1;
                // Fill in rows up to and including the current row
                //for (VertexId row = prev_row + 1; row <= current_row; row++) {
                //    row_offsets[row] = edge;
                //}
                //prev_row = current_row;

                column_indices[edge + edge_offset] = new_coo[edge].col;
                if (LOAD_EDGE_VALUES)
                {
                    //new_coo[edge].Val(edge_values[edge]);
                    edge_values[edge + edge_offset] = new_coo[edge].val;
                }
            }
            #pragma omp barrier
            //if (first_row != last_row)
            if (edge_start > 0 && coo[edge_start].row == coo[edge_start - 1].row) // same row as previous thread
                if (edge_end == coo_edges || coo[edge_end].row != coo[edge_start].row) // first row ends at this thread
                {
                    row_offsets[first_row + 1] = pointer;
                }
            #pragma omp barrier
            // Fill out any trailing edgeless nodes (and the end-of-list element)
            //for (VertexId row = prev_row + 1; row <= nodes; row++) {
            //    row_offsets[row] = real_edge;
            //}
            if (row_offsets[node_start] == -1)
            {
                VertexId i = node_start;
                while (row_offsets[i] == -1) i--;
                row_offsets[node_start] = row_offsets[i];
            }
            for (VertexId node = node_start + 1; node < node_end; node++)
                if (row_offsets[node] == -1)
                {
                    row_offsets[node] = row_offsets[node - 1];
                }
            if (thread_num == 0) edges = edge_offsets[num_threads];

            free(new_coo); new_coo = NULL;
        }

        row_offsets[nodes] = edges;
        delete[] edge_offsets; edge_offsets = NULL;
        delete[] edge_counts ; edge_counts  = NULL;

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            printf("Done converting (%ds).\n", (int)(mark2 - mark1));
        }

        // Write offsets, indices, node, edges etc. into file
        if (LOAD_EDGE_VALUES)
        {
            WriteBinary(output_file, nodes, edges,
                        row_offsets, column_indices, edge_values);
            //WriteCSR(output_file, nodes, edges,
            //         row_offsets, column_indices, edge_values);
            //WriteToLigraFile(output_file, nodes, edges,
            //                 row_offsets, column_indices, edge_values);
        }
        else
        {
            WriteBinary(output_file, nodes, edges,
                        row_offsets, column_indices);
        }

        // Compute out_nodes
        SizeT out_node = 0;
        for (SizeT node = 0; node < nodes; node++)
        {
            if (row_offsets[node + 1] - row_offsets[node] > 0)
            {
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
    void PrintHistogram()
    {
        fflush(stdout);

        // Initialize
        SizeT log_counts[32];
        for (int i = 0; i < 32; i++)
        {
            log_counts[i] = 0;
        }

        // Scan
        SizeT max_log_length = -1;
        for (VertexId i = 0; i < nodes; i++)
        {

            SizeT length = row_offsets[i + 1] - row_offsets[i];

            int log_length = -1;
            while (length > 0)
            {
                length >>= 1;
                log_length++;
            }
            if (log_length > max_log_length)
            {
                max_log_length = log_length;
            }

            log_counts[log_length + 1]++;
        }
        printf("\nDegree Histogram (%lld vertices, %lld edges):\n",
               (long long) nodes, (long long) edges);
        printf("    Degree   0: %lld (%.2f%%)\n", 
               (long long) log_counts[0],
               (float) log_counts[0] * 100.0 / nodes);
        for (int i = 0; i < max_log_length + 1; i++)
        {
            printf("    Degree 2^%i: %lld (%.2f%%)\n", 
                i, (long long)log_counts[i + 1],
                (float) log_counts[i + 1] * 100.0 / nodes);
        }
        printf("\n");
        fflush(stdout);
    }


    /**
     * @brief Display CSR graph to console
     *
     * @param[in] with_edge_value Whether display graph with edge values.
     */
    void DisplayGraph(bool with_edge_value = false)
    {
        SizeT displayed_node_num = (nodes > 40) ? 40 : nodes;
        printf("First %d nodes's neighbor list of the input graph:\n",
               displayed_node_num);
        for (SizeT node = 0; node < displayed_node_num; node++)
        {
            util::PrintValue(node);
            printf(":");
            for (SizeT edge = row_offsets[node];
                    edge < row_offsets[node + 1];
                    edge++)
            {
                if (edge - row_offsets[node] > 40) break;
                printf("[");
                util::PrintValue(column_indices[edge]);
                if (with_edge_value && edge_values != NULL)
                {
                    printf(",");
                    util::PrintValue(edge_values[edge]);
                }
                printf("], ");
            }
            printf("\n");
        }
    }

    /**
     * @brief Display CSR graph to console
     */
    void DisplayGraph(const char name[], SizeT limit = 40)
    {
        SizeT displayed_node_num = (nodes > limit) ? limit : nodes;
        printf("%s : #nodes = ", name); util::PrintValue(nodes);
        printf(", #edges = "); util::PrintValue(edges);
        printf("\n");

        for (SizeT i = 0; i < displayed_node_num; i++)
        {
            util::PrintValue(i);
            printf(",");
            util::PrintValue(row_offsets[i]);
            if (node_values != NULL)
            {
                printf(",");
                util::PrintValue(node_values[i]);
            }
            printf(" (");
            for (SizeT j = row_offsets[i]; j < row_offsets[i + 1]; j++)
            {
                if (j != row_offsets[i]) printf(" , ");
                util::PrintValue(column_indices[j]);
                if (edge_values != NULL)
                {
                    printf(",");
                    util::PrintValue(edge_values[j]);
                }
            }
            printf(")\n");
        }

        printf("\n");
    }

    /**
     * @brief Check values.
     */
    bool CheckValue()
    {
        for (SizeT node = 0; node < nodes; ++node)
        {
            for (SizeT edge = row_offsets[node];
                    edge < row_offsets[node + 1];
                    ++edge)
            {
                int src_node = node;
                int dst_node = column_indices[edge];
                int edge_value = edge_values[edge];
                for (SizeT r_edge = row_offsets[dst_node];
                        r_edge < row_offsets[dst_node + 1];
                        ++r_edge)
                {
                    if (column_indices[r_edge] == src_node)
                    {
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
     * @param[in] max_degree Maximum degree in the graph.
     *
     * \return int the source node with highest degree
     */
    int GetNodeWithHighestDegree(int& max_degree)
    {
        int degree = 0;
        int src = 0;
        for (SizeT node = 0; node < nodes; node++)
        {
            if (row_offsets[node + 1] - row_offsets[node] > degree)
            {
                degree = row_offsets[node + 1] - row_offsets[node];
                src = node;
            }
        }
        max_degree = degree;
        return src;
    }

    /**
     * @brief Display the neighbor list of a given node.
     *
     * @param[in] node Vertex ID to display.
     */
    void DisplayNeighborList(VertexId node)
    {
        if (node < 0 || node >= nodes) return;
        for (SizeT edge = row_offsets[node];
                edge < row_offsets[node + 1];
                edge++)
        {
            util::PrintValue(column_indices[edge]);
            printf(", ");
        }
        printf("\n");
    }

    /**
     * @brief Get the average degree of all the nodes in graph
     */
    SizeT GetAverageDegree()
    {
        if (average_degree == 0)
        {
            double mean = 0, count = 0;
            for (SizeT node = 0; node < nodes; ++node)
            {
                count += 1;
                mean += (row_offsets[node + 1] - row_offsets[node] - mean) / count;
            }
            average_degree = static_cast<SizeT>(mean);
        }
        return average_degree;
    }

    /**
     * @brief Get the average degree of all the nodes in graph
     */
    SizeT GetStddevDegree()
    {
        if (average_degree == 0)
        {
           GetAverageDegree();
        }

        float accum = 0.0f;
        for (SizeT node=0; node < nodes; ++node)
        {
            float d = (row_offsets[node+1]-row_offsets[node]);
            accum += (d - average_degree) * (d - average_degree);
        }
        stddev_degree = sqrt(accum / (nodes-1));
        return stddev_degree;
    }

    /**
     * @brief Get the degrees of all the nodes in graph
     * 
     * @param[in] node_degrees node degrees to fill in
     */
    void GetNodeDegree(SizeT *node_degrees)
    {
	for(SizeT node=0; node < nodes; ++node)
	{
		node_degrees[node] = row_offsets[node+1]-row_offsets[node];
	}
    }

    /**
     * @brief Get the average node value in graph
     */
    Value GetAverageNodeValue()
    {
        if (abs(average_node_value - 0) < 0.001 && node_values != NULL)
        {
            double mean = 0, count = 0;
            for (SizeT node = 0; node < nodes; ++node)
            {
                if (node_values[node] < UINT_MAX)
                {
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
    Value GetAverageEdgeValue()
    {
        if (abs(average_edge_value - 0) < 0.001 && edge_values != NULL)
        {
            double mean = 0, count = 0;
            for (SizeT edge = 0; edge < edges; ++edge)
            {
                if (edge_values[edge] < UINT_MAX)
                {
                    count += 1;
                    mean += (edge_values[edge] - mean) / count;
                }
            }
        }
        return average_edge_value;
    }

    /**@}*/

    /**
     * @brief Deallocates CSR graph
     */
    void Free()
    {
        if (row_offsets)
        {
            if (pinned)
            {
                gunrock::util::GRError(
                    cudaFreeHost(row_offsets),
                    "Csr cudaFreeHost row_offsets failed",
                    __FILE__, __LINE__);
            }
            else
            {
                free(row_offsets);
            }
            row_offsets = NULL;
        }
        if (column_indices)
        {
            if (pinned)
            {
                gunrock::util::GRError(
                    cudaFreeHost(column_indices),
                    "Csr cudaFreeHost column_indices failed",
                    __FILE__, __LINE__);
            }
            else
            {
                free(column_indices);
            }
            column_indices = NULL;
        }
        if (edge_values)
        {
            free (edge_values); edge_values = NULL;
        }
        if (node_values)
        {
            free (node_values); node_values = NULL;
        }

        nodes = 0;
        edges = 0;
    }

    /**
     * @brief CSR destructor
     */
    ~Csr()
    {
        Free();
    }
};

} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
