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

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/graph_base.cuh>
#include <gunrock/util/binary_search.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief CSR data structure which uses Compressed Sparse Row
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 */
template<
    typename VertexT = int,
    typename SizeT   = VertexT,
    typename ValueT  = VertexT,
    GraphFlag FLAG   = GRAPH_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct Csr :
    public GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag>
{
    typedef GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> BaseGraph;
    // Column indices corresponding to all the
    // non-zero values in the sparse matrix
    util::Array1D<SizeT, VertexT,
        util::If_Val<FLAG & GRAPH_PINNED, util::PINNED, util::ARRAY_NONE>::Value,
        cudaHostRegisterFlag> column_indices;

    // List of indices where each row of the
    // sparse matrix starts
    util::Array1D<SizeT, SizeT,
        util::If_Val<FLAG & GRAPH_PINNED, util::PINNED, util::ARRAY_NONE>::Value,
        cudaHostRegisterFlag> row_offsets;

    typedef util::Array1D<SizeT, ValueT,
        util::If_Val<FLAG & GRAPH_PINNED, util::PINNED, util::ARRAY_NONE>::Value,
        cudaHostRegisterFlag> Array_ValueT;

    // List of values attached to edges in the graph
    typename util::If<FLAG & HAS_EDGE_VALUES,
        Array_ValueT, util::NullArray<SizeT, ValueT, FLAG, cudaHostRegisterFlag> >::Type edge_values;

    // List of values attached to nodes in the graph
    typename util::If<FLAG & HAS_NODE_VALUES,
        Array_ValueT, util::NullArray<SizeT, ValueT, FLAG, cudaHostRegisterFlag> >::Type node_values;

    /**
     * @brief CSR Constructor
     *
     * @param[in] pinned Use pinned memory for CSR data structure
     * (default: do not use pinned memory)
     */
    Csr() : BaseGraph()
    {
        row_offsets   .SetName("row_offsets");
        column_indices.SetName("column_indices");
        edge_values   .SetName("edge_values");
        node_values   .SetName("node_values");
    }

    /**
     * @brief CSR destructor
     */
    ~Csr()
    {
        //Release();
    }

    /**
     * @brief Deallocates CSR graph
     */
    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (retval = row_offsets   .Release()) return retval;
        if (retval = column_indices.Release()) return retval;
        if (retval = node_values   .Release()) return retval;
        if (retval = edge_values   .Release()) return retval;
        if (retval = BaseGraph    ::Release()) return retval;
        return retval;
    }

    /**
     * @brief Allocate memory for CSR graph.
     *
     * @param[in] nodes Number of nodes in COO-format graph
     * @param[in] edges Number of edges in COO-format graph
     */
    cudaError_t Allocate(SizeT nodes, SizeT edges,
        util::Location target = GRAPH_DEFAULT_TARGET)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseGraph    ::Allocate(nodes, edges, target))
            return retval;
        if (retval = row_offsets   .Allocate(nodes + 1  , target))
            return retval;
        if (retval = column_indices.Allocate(edges      , target))
            return retval;
        if (retval = node_values   .Allocate(nodes      , target))
            return retval;
        if (retval = edge_values   .Allocate(edges      , target))
            return retval;
        return retval;
    }

    template <
        typename VertexT_in, typename SizeT_in,
        typename ValueT_in, GraphFlag FLAG_in,
        unsigned int cudaHostRegisterFlag_in>
    cudaError_t FromCsr(
        Csr<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
            cudaHostRegisterFlag_in> &source,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0)
    {
        cudaError_t retval = cudaSuccess;
        if (target == util::LOCATION_DEFAULT)
            target = source.row_offsets.setted | source.row_offsets.allocated;

        if (retval = BaseGraph::Set(source))
            return retval;

        if (retval = Allocate(source.nodes, source.edges, target))
            return retval;

        if (retval = row_offsets   .Set(source.row_offsets,
            this -> nodes + 1, target, stream))
            return retval;

        if (retval = column_indices.Set(source.column_indices,
            this -> edges, target, stream))
            return retval;

        if (retval = edge_values   .Set(source.edge_values,
            this -> edges, target, stream))
            return retval;

        if (retval = node_values   .Set(source.node_values,
            this -> nodes, target, stream))
            return retval;

        return retval;
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
    template <typename CooT>
    cudaError_t FromCoo(
        CooT &source,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0,
        //bool  ordered_rows = false,
        //bool  undirected = false,
        //bool  reversed = false,
        bool  quiet = false)
    {
        //typedef Coo<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
        //    cudaHostRegisterFlag_in> CooT;
        if (!quiet)
        {
            util::PrintMsg("  Converting " +
                std::to_string(source.CooT::nodes) +
                " vertices, " + std::to_string(source.CooT::edges) +
                (source.CooT::directed ? " directed" : " undirected") +
                " edges (" + (source.CooT::edge_order == BY_ROW_ASCENDING ? " ordered" : "unordered") +
                " tuples) to CSR format...");
        }

        time_t mark1 = time(NULL);
        cudaError_t retval = cudaSuccess;
        if (target == util::LOCATION_DEFAULT)
            target = source.CooT::edge_pairs.GetSetted() | source.CooT::edge_pairs.GetAllocated();

        /*if (retval = BaseGraph:: template Set<typename CooT::CooT>((typename CooT::CooT)source))
            return retval;
        */
        this -> nodes = source.CooT::nodes;
        this -> edges = source.CooT::edges;
        this -> directed = source.CooT::directed;

        if (retval = Allocate(source.CooT::nodes, source.CooT::edges, target))
            return retval;

        // Sort COO by row
        if (retval = source.CooT::Order(BY_ROW_ASCENDING, target, stream))
            return retval;
        source.CooT::Display();

        // assign column_indices
        if (retval = column_indices.ForEach(source.CooT::edge_pairs,
                []__host__ __device__ (VertexT &column_index,
                const typename CooT::EdgePairT &edge_pair){
                column_index = edge_pair.y;},
                this -> edges, target, stream))
            return retval;

        // assign edge_values
        if (FLAG & HAS_EDGE_VALUES)
            if (retval = edge_values.ForEach(source.CooT::edge_values,
                []__host__ __device__ (ValueT &edge_value,
                const typename CooT::ValueT &edge_value_in){
                edge_value = edge_value_in;},
                this -> edges, target, stream))
            return retval;

        // assign row_offsets
        SizeT edges = this -> edges;
        SizeT nodes = this -> nodes;
        auto row_edge_compare = [] __host__ __device__ (
            const typename CooT::EdgePairT &edge_pair,
            const VertexT &row){
            return edge_pair.x < row;
        };
        if (retval = row_offsets.ForAll(source.CooT::edge_pairs,
            [nodes, edges, row_edge_compare] __host__ __device__ (
                SizeT *row_offsets,
                const typename CooT::EdgePairT *edge_pairs,
                const VertexT &row){
                    if (row < nodes)
                        row_offsets[row] = util::BinarySearch(row,
                            edge_pairs, 0, edges,
                            row_edge_compare);
                    else row_offsets[row] = edges;
                }, this -> nodes + 1, target, stream))
            return retval;

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            util::PrintMsg("Done converting (" +
                std::to_string(mark2 - mark1) + "s).");
        }
        return retval;
        /*SizeT edge_offsets[129];
        SizeT edge_counts [129];
        #pragma omp parallel
        {
            int num_threads  = omp_get_num_threads();
            int thread_num   = omp_get_thread_num();
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
        out_nodes = out_node;*/
    }

    /**
     * @brief Display CSR graph to console
     *
     * @param[in] with_edge_value Whether display graph with edge values.
     */
    cudaError_t Display(
        std::string graph_prefix = "",
        SizeT nodes_to_show = 40,
        bool  with_edge_values = true)
    {
        cudaError_t retval = cudaSuccess;
        if (nodes_to_show > this -> nodes)
            nodes_to_show = this -> nodes;
        util::PrintMsg(graph_prefix + "Graph containing " +
            std::to_string(this -> nodes) + " vertices, " +
            std::to_string(this -> edges) + " edges, in CSR format. Neighbor list of first " + std::to_string(nodes_to_show) +
            " nodes :");
        for (SizeT node = 0; node < nodes_to_show; node++)
        {
            std::string str = "v " + std::to_string(node) +
             " " + std::to_string(row_offsets[node]) + " : ";
            for (SizeT edge = row_offsets[node];
                    edge < row_offsets[node + 1];
                    edge++)
            {
                if (edge - row_offsets[node] > 40) break;
                str = str + "[" + std::to_string(column_indices[edge]);
                if (with_edge_values && (FLAG & HAS_EDGE_VALUES))
                {
                    str = str + "," + std::to_string(edge_values[edge]);
                }
                if (edge - row_offsets[node] != 40 &&
                    edge != row_offsets[node+1] -1)
                    str = str + "], ";
                else str = str + "]";
            }
            if (row_offsets[node + 1] - row_offsets[node] > 40)
                str = str + "...";
            util::PrintMsg(str);
        }
        return retval;
    }

    /*template <typename Tuple>
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
    }*/

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
    /*void WriteBinary(
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
    }*/

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
    /*void WriteCSR(
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
    }*/

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
    /*void WriteToLigraFile(
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
    }*/


    /**
     * @brief Read from stored row_offsets, column_indices arrays.
     *
     * @tparam LOAD_EDGE_VALUES Whether or not to load edge values.
     *
     * @param[in] f_in Input file name.
     * @param[in] quiet Don't print out anything.
     */
    /*template <bool LOAD_EDGE_VALUES>
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
    }*/

    /**
     * @brief (Specific for SM) Read from stored row_offsets, column_indices arrays.
     *
     * @tparam LOAD_NODE_VALUES Whether or not to load node values.
     *
     * @param[in] f_in Input graph file name.
     * @param[in] f_label Input label file name.
     * @param[in] quiet Don't print out anything.
     */
    /*template <bool LOAD_NODE_VALUES>
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
//	    for(int i=0; i<v; i++) printf("%lld ", (long long)node_values[i]); printf("\n");

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
    }*/

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Print log-scale degree histogram of the graph.
     */
    /*void PrintHistogram()
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
    }*/

    /**
     * @brief Display CSR graph to console
     */
    /*void DisplayGraph(const char name[], SizeT limit = 40)
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
    }*/

    /**
     * @brief Check values.
     */
    /*bool CheckValue()
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
    }*/

    /**
     * @brief Find node with largest neighbor list
     * @param[in] max_degree Maximum degree in the graph.
     *
     * \return int the source node with highest degree
     */
    /*int GetNodeWithHighestDegree(int& max_degree)
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
    }*/

    /**
     * @brief Display the neighbor list of a given node.
     *
     * @param[in] node Vertex ID to display.
     */
    /*void DisplayNeighborList(VertexId node)
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
    }*/

    /**
     * @brief Get the average degree of all the nodes in graph
     */
    /*SizeT GetAverageDegree()
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
    }*/

    /**
     * @brief Get the average degree of all the nodes in graph
     */
    /*SizeT GetStddevDegree()
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
    }*/

    /**
     * @brief Get the degrees of all the nodes in graph
     *
     * @param[in] node_degrees node degrees to fill in
     */
    /*void GetNodeDegree(unsigned long long *node_degrees)
    {
	for(SizeT node=0; node < nodes; ++node)
	{
		node_degrees[node] = row_offsets[node+1]-row_offsets[node];
	}
    }*/

    /**
     * @brief Get the average node value in graph
     */
    /*Value GetAverageNodeValue()
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
    }*/

    /**
     * @brief Get the average edge value in graph
     */
    /*Value GetAverageEdgeValue()
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
    }*/

    /**@}*/
};

} // namespace graph
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
