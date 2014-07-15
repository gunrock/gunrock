// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * csr.cuh
 *
 * @brief CSR (Column Sparse Row) Graph Data Structure
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

#include <gunrock/util/error_utils.cuh>

using namespace std;

namespace gunrock {

/**
 * @brief CSR data structure which uses Compressed Sparse Row
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 */
template<typename VertexId, typename Value, typename SizeT>
struct Csr
{
    SizeT nodes;    /**< Number of nodes in the graph. */
    SizeT edges;    /**< Number of edges in the graph. */
    SizeT out_nodes; /**< Number of nodes which have outgoing edges. */
    SizeT average_degree;

    VertexId    *column_indices;/**< Column indices corresponding to all the non-zero values in the sparse matrix. */
    SizeT       *row_offsets;   /**< List of indices where each row of the sparse matrix starts. */
    Value       *edge_values;   /**< List of values attached to edges in the graph. */
    Value       *node_values;   /**< List of values attached to nodes in the graph. */

    Value       average_edge_value;
    Value       average_node_value;

    bool        pinned;        /**< Whether to use pinned memory */

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
    void FromScratch(SizeT nodes, SizeT edges)
    {
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
     * @store graph information into files
     * 
     */
  void WriteToFile(char * file_name,
		   bool undirected, 
		   bool reversed,
		   SizeT num_nodes, 
		   SizeT num_edges, 
		   SizeT *row_offsets,
		   VertexId *col_indices,
		   Value *edge_values = NULL)
  {
    printf("==> Writing into file:  %s\n", file_name);
    
    time_t mark1 = time(NULL);
    
    std::ofstream output(file_name);
    
    if (output.is_open())
    {
      output << num_nodes << " " << num_edges << " ";
      std::copy(row_offsets,   row_offsets + num_nodes + 1, ostream_iterator<int>(output, " "));
      std::copy(column_indices, column_indices + num_edges, ostream_iterator<int>(output, " "));
      if (edge_values != NULL)
      {
	std::copy(edge_values, edge_values + num_edges, ostream_iterator<int>(output, " "));
      }
      output.close();
    }
    else
    {
      std::cout << "Cannot Open The File." << std::endl;
    }

    time_t mark2 = time(NULL);
    printf("Finished writing in %ds.\n", (int)(mark2 - mark1));
  }

  // read from stored row_offsets, column_indices arrays
  template <bool LOAD_EDGE_VALUES>
  void FromCsr(char *f_in, 
	       bool undirected, 
	       bool reversed)
  {
    printf("  Reading directly from previously stored CSR arrays ...\n");
    
    ifstream _file;
    char buf[65536];
    _file.rdbuf()->pubsetbuf(buf,65536);

    _file.open(f_in);
    
    if (_file.is_open())
    {
      time_t mark1 = time(NULL);
      
      std::istream_iterator<int> start(_file), end;
      std::vector<int> v(start, end);
      
      SizeT csr_nodes = v.at(0);
      SizeT csr_edges = v.at(1);
      
      FromScratch<LOAD_EDGE_VALUES, false>(csr_nodes, csr_edges); 
      
      copy(v.begin()+2, v.begin()+3+csr_nodes, row_offsets);
      copy(v.begin()+3+csr_nodes, v.begin()+3+csr_nodes+csr_edges, column_indices);
      if(LOAD_EDGE_VALUES) 
      { 
	copy(v.begin()+3+csr_nodes+csr_edges, v.end(), edge_values); 
      }
      
      time_t mark2 = time(NULL);
      printf("Done reading (%ds).\n", (int) (mark2 - mark1));
      
      v.clear();
    }
    else 
    {
      perror("Unable to open the file."); 
    }

    // compute out_nodes
    SizeT out_node = 0;
    for (SizeT node = 0; node < nodes; node++) 
    {
      if (row_offsets[node+1] - row_offsets[node] > 0)
      {
	++out_node;
      }
    }
    out_nodes = out_node;

    fflush(stdout);
  }


    /**
     * @brief Build CSR graph from COO graph, sorted or unsorted
     *
     * @param[in] coo Pointer to COO-format graph
     * @param[in] coo_nodes Number of nodes in COO-format graph
     * @param[in] coo_edges Number of edges in COO-format graph
     * @param[in] ordered_rows Are the rows sorted? If not, sort them.
     * Default: Assume rows are not sorted.
     */
    template <bool LOAD_EDGE_VALUES, typename Tuple>
    void FromCoo(
        char *output_file,
        Tuple *coo,
        SizeT coo_nodes,
        SizeT coo_edges,
        bool ordered_rows = false,
        bool undirected = false,
        bool reversed = false)
    {
        printf("  Converting %d vertices, %d directed edges (%s tuples) "
               "to CSR format... \n",
               coo_nodes, coo_edges, ordered_rows ? "ordered" : "unordered");
        time_t mark1 = time(NULL);
        fflush(stdout);

        FromScratch<LOAD_EDGE_VALUES, false>(coo_nodes, coo_edges);

        // Sort COO by row
        if (!ordered_rows) {
            std::stable_sort(coo, coo + coo_edges, RowFirstTupleCompare<Tuple>);
        }

        Tuple *new_coo = (Tuple*) malloc(sizeof(Tuple) * coo_edges);
        SizeT real_edge = 1;
        new_coo[0].row = coo[0].row;
        new_coo[0].col = coo[0].col;
        new_coo[0].val = coo[0].val;
        for (int i = 0; i < coo_edges-1; ++i)
        {
            if (((coo[i+1].col != coo[i].col) || (coo[i+1].row != coo[i].row)) && (coo[i+1].col != coo[i+1].row))
            {
                new_coo[real_edge].col = coo[i+1].col;
                new_coo[real_edge].row = coo[i+1].row;
                new_coo[real_edge++].val = coo[i+1].val;
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
        printf("Done converting (%ds).\n", (int)(mark2 - mark1));
        
        // Write offsets, indices, node, edges etc. into file
        if (LOAD_EDGE_VALUES)
	{
	  WriteToFile(output_file, 
		      undirected, 
		      reversed, 
		      nodes, 
		      edges, 
		      row_offsets, 
		      column_indices, 
		      edge_values);
        }
        else
        {
	  WriteToFile(output_file, 
		      undirected, 
		      reversed,
		      nodes, 
		      edges, 
		      row_offsets, 
		      column_indices);
        }

        if (new_coo) free(new_coo);
        fflush(stdout);

        // Compute out_nodes
        SizeT out_node = 0;
        for (SizeT node = 0; node < nodes; node++) {
            if (row_offsets[node+1] - row_offsets[node] > 0)
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
        printf("\nDegree Histogram (%lld vertices, %lld directed edges):\n",
               (long long) nodes, (long long) edges);
        for (int i = -1; i < max_log_length + 1; i++) {
            printf("\tDegree 2^%i: %d (%.2f%%)\n", i, log_counts[i + 1],
                   (float) log_counts[i + 1] * 100.0 / nodes);
        }
        printf("\n");
        fflush(stdout);
    }


    /**
     * @brief Display CSR graph to console
     */
    void DisplayGraph(bool with_edge_value = false)
    {
        SizeT displayed_node_num = (nodes > 40) ? 40:nodes;
        printf("First %d nodes's neighbor list of the input graph:\n", displayed_node_num);
        for (SizeT node = 0; node < displayed_node_num; node++) {
            util::PrintValue(node);
            printf(":");
            for (SizeT edge = row_offsets[node];
                 edge < row_offsets[node + 1];
                 edge++) {
                 printf("[");
                util::PrintValue(column_indices[edge]);
                if (with_edge_value) {
                    printf(",");
                    util::PrintValue(edge_values[edge]);
                }
                printf("], ");
            }
            printf("\n");
        }
    }

    void DisplayGraph(const char name[], SizeT limit = 40) 
    {   
        SizeT displayed_node_num = (nodes > limit) ? limit: nodes;
        printf("%s : #nodes = ",name); util::PrintValue(nodes);
        printf(", #edges = "); util::PrintValue(edges);
        printf("\n  row_offsets = ");
        for (SizeT i=0;i<=displayed_node_num;i++) 
        {   
            if (i!=0) printf(", ");
            util::PrintValue(row_offsets[i]);
        }   

        if (node_values != NULL)
        {   
            printf("\n  node_values = ");
            for (SizeT i=0;i<displayed_node_num;i++)
            {   
                if (i!=0) printf(", ");
                util::PrintValue(node_values[i]); 
            }   
        }   
 
        printf("\n  edges = ");
        for (SizeT i=0;i<displayed_node_num;i++)
        {   
            printf("(");
            for (SizeT j=row_offsets[i];j<row_offsets[i+1];j++)
            {   
                if (j!=row_offsets[i]) printf(", ");
                util::PrintValue(column_indices[j]);
            }   
            printf(") ");
        }   
    
        if (edge_values != NULL)
        {   
            printf("\n edge_values = ");
            for (SizeT i=0;i<displayed_node_num;i++)
            {   
                printf("(");
                for (SizeT j=row_offsets[i];j<row_offsets[i+1];j++)
                {   
                    if (j!=row_offsets[i]) printf(", ");
                    util::PrintValue(edge_values[j]);
                }   
                printf(") ");
            }   
        }   

        printf("\n");
    }   

    bool CheckValue()
    {
        for (SizeT node = 0; node < nodes; ++node) {
            for (SizeT edge = row_offsets[node];
                 edge < row_offsets[node+1];
                 ++edge) {
                 int src_node = node;
                 int dst_node = column_indices[edge];
                 int edge_value = edge_values[edge];
                 for (SizeT r_edge = row_offsets[dst_node];
                 r_edge < row_offsets[dst_node+1];
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
    int GetNodeWithHighestDegree()
    {
        int degree = 0;
        int src = 0;
        for (SizeT node = 0; node < nodes; node++) {
            if (row_offsets[node+1] - row_offsets[node] > degree)
            {
                degree = row_offsets[node+1]-row_offsets[node];
                src = node;
            }
        }
        return src;
    }

    /**
     * @brief Display the neighbor list of a node
     */
    void DisplayNeighborList(VertexId node)
    {
        for (SizeT edge = row_offsets[node];
                 edge < row_offsets[node + 1];
                 edge++) {
                util::PrintValue(column_indices[edge]);
                printf(", ");
            }
            printf("\n");
    }

    SizeT GetAverageDegree() {
        if (average_degree == 0) {
            double mean = 0, count = 0;
            for (SizeT node = 0; node < nodes; ++node) {
                count += 1;
                mean += (row_offsets[node+1]- row_offsets[node] - mean) / count;
            }
            average_degree = static_cast<SizeT>(mean);
        }
        return average_degree;
    }

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
    void Free()
    {
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
