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


template<bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int ReadLabelStream(
    FILE *f_in,
    char *output_file,
    Csr<VertexId, SizeT, Value> &csr_graph,
    bool quiet = false)
{
    SizeT lines_read = -1;
    SizeT nodes = 0;

    char line[1024];
    Value *labels = NULL;

    time_t mark0 = time(NULL);
    if(!quiet) printf("Parsing node labels...\n");

    fflush(stdout);

    while (true)
    {

        if (fscanf(f_in, "%[^\n]\n", line) <= 0)
        {
            break;
        }

        if (line[0] == '%')
        {

            // Comment

        }
        else if (lines_read == -1)
        {

            // Problem description
            long long ll_nodes_x, ll_nodes_y;
            if (sscanf(line, "%lld %lld",
                       &ll_nodes_x, &ll_nodes_y) != 2)
            {
                fprintf(stderr, "Error parsing node labels:"
                        " invalid problem description.\n");
                return -1;
            }

            nodes = ll_nodes_x;

            if (!quiet)
            {
                printf(" (%lld nodes)... ",
                       (unsigned long long) ll_nodes_x);
                fflush(stdout);
            }

            // Allocate node labels
            unsigned long long allo_size = sizeof(Value);
            allo_size = allo_size * nodes;
            labels = (Value*)malloc(allo_size);
            if (labels == NULL)
            {
                fprintf(stderr, "Error parsing node labels:"
                    "labels allocation failed, sizeof(Value) = %lu,"
                    " nodes = %lld, allo_size = %lld\n", 
                    sizeof(Value), (long long)nodes, (long long)allo_size);
                return -1;
            }

            lines_read++;

        }
        else
        {

            // node label description (v -> l)
            if (!labels)
            {
                fprintf(stderr, "Error parsing node labels: invalid format\n");
                return -1;
            }
            if (lines_read >= nodes)
            {
                fprintf(stderr,
                        "Error parsing node labels:"
                        "encountered more than %lld nodes\n",
                        (long long)nodes);
                if (labels) free(labels);
                return -1;
            }

            long long ll_node, ll_label;
            if (sscanf(line, "%lld %lld", &ll_node, &ll_label) != 2)
                {
                    fprintf(stderr,
                            "Error parsing node labels: badly formed\n");
                    if (labels) free(labels);
                    return -1;
                }

	    labels[lines_read] = ll_label;

            lines_read++;

        }
    }
    
    if (labels == NULL)
    {
        fprintf(stderr, "No input labels found\n");
        return -1;
    }

    if (lines_read != nodes)
    {
        fprintf(stderr,
                "Error parsing node labels: only %lld/%lld nodes read\n",
                (long long)lines_read, (long long)nodes);
        if (labels) free(labels);
        return -1;
    }

    time_t mark1 = time(NULL);
    if (!quiet)
    {
        printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
        fflush(stdout);
    }

    // Convert labels into binary 
    csr_graph.template FromLabels<LOAD_VALUES>(output_file, labels, nodes, quiet);

    free(labels);
    fflush(stdout);

    return 0;
}
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
 * @param[in] f_in          Input MARKET graph file.
 * @param[in] output_file   Output file name for binary i/o.
 * @param[in] csr_graph     Csr graph object to store the graph data.
 * @param[in] undirected    Is the graph undirected or not?
 * @param[in] reversed      Whether or not the graph is inversed.
 *
 * \return If there is any File I/O error along the way.
 */
template<bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int ReadMarketStream(
    FILE *f_in,
    char *output_file,
    Csr<VertexId, SizeT, Value> &csr_graph,
    bool undirected,
    bool reversed,
    bool quiet = false)
{
    typedef Coo<VertexId, Value> EdgeTupleType;

    SizeT edges_read = -1;
    SizeT nodes = 0;
    SizeT edges = 0;
    EdgeTupleType *coo = NULL; // read in COO format
    bool  skew  = false; //whether edge values are the inverse for symmetric matrices
    bool  array = false; //whether the mtx file is in dense array format
    
    time_t mark0 = time(NULL);
    if (!quiet)
    {
        printf("  Parsing MARKET COO format");
    }
    fflush(stdout);

    char line[1024];

    bool ordered_rows = true;
    
    bool is_first_line_empty = false;

    while (true)
    {

        if (fscanf(f_in, "%[^\n]\n", line) <= 0)
        {
	   if (edges_read == -1)
	   {	
		is_first_line_empty = true;
	   }
	   break;
	}
	
        if (line[0] == '%')
        {

            // Comment
            if (strlen(line) >= 2 && line[1] == '%')
            {
                // Banner
                if (!undirected)
                    undirected = (strstr(line, "symmetric") != NULL);
                skew       = (strstr(line, "skew"     ) != NULL);
                array      = (strstr(line, "array"    ) != NULL);
            }

        }
        else if (edges_read == -1)
        {

            // Problem description
            long long ll_nodes_x, ll_nodes_y, ll_edges;
            int items_scanned = sscanf(line, "%lld %lld %lld",
                       &ll_nodes_x, &ll_nodes_y, &ll_edges);
 
	    if (array && items_scanned == 2)
            {
                ll_edges = ll_nodes_x * ll_nodes_y;
            } 

            else if (!array && items_scanned == 3)
            {
                if (ll_nodes_x != ll_nodes_y)
                {
                    fprintf(stderr,
                            "Error parsing MARKET graph: not square (%lld, %lld)\n",
                            ll_nodes_x, ll_nodes_y);
                    return -1;
                }
                if (undirected) ll_edges *=2;
            } 

            else {
                fprintf(stderr, "Error parsing MARKET graph:"
                        " invalid problem description.\n");
                return -1;
            }

            nodes = ll_nodes_x;
            edges = ll_edges;

            if (!quiet)
            {
                printf(" (%lld nodes, %lld directed edges)... ",
                       (unsigned long long) ll_nodes_x,
                       (unsigned long long) ll_edges);
                fflush(stdout);
            }

            // Allocate coo graph
            unsigned long long allo_size = sizeof(EdgeTupleType);
            allo_size = allo_size * edges;
            coo = (EdgeTupleType*)malloc(allo_size);
            if (coo == NULL)
            {
                fprintf(stderr, "Error parsing MARKET graph:"
                    "coo allocation failed, sizeof(EdgeTupleType) = %lu,"
                    " edges = %lld, allo_size = %lld\n", 
                    sizeof(EdgeTupleType), (long long)edges, (long long)allo_size);
                return -1;
            }

            edges_read++;

        }
        else
        {

            // Edge description (v -> w)
            if (!coo)
            {
                fprintf(stderr, "Error parsing MARKET graph: invalid format\n");
                return -1;
            }
            if (edges_read >= edges)
            {
                fprintf(stderr,
                        "Error parsing MARKET graph:"
                        "encountered more than %lld edges\n",
                        (long long)edges);
                if (coo) free(coo);
                return -1;
            }

            long long ll_row, ll_col;
            Value ll_value;  // used for parse float / double
            double lf_value; // used to sscanf value variable types
            int num_input;
            if (LOAD_VALUES)
            {
                num_input = sscanf(line, "%lld %lld %lf",
                    &ll_row, &ll_col, &lf_value);

		if (ll_row <  1 || ll_col <  1)
		{
		    fprintf(stderr, 
			    "\nInvalid graph: Bad MTX format, indices cannot start with 0.\n");
		    free(coo);
		    exit(1); 
		}		

                if (typeid(Value) == typeid(float) || typeid(Value) == typeid(double))
                    ll_value = (Value)lf_value;
                else ll_value = (Value)(lf_value + 1e-10);

                if (array && (num_input == 1))
                {
                    ll_value = ll_row;
                    ll_col   = edges_read / nodes;
                    ll_row   = edges_read - nodes * ll_col;
                    //printf("%f\n", ll_value);
                } 

                else if (array || num_input < 2)
                {
                    fprintf(stderr,                            "Error parsing MARKET graph: badly formed edge\n");
                    if (coo) free(coo);
                    return -1;
                }

                else if (num_input == 2)
                {
                    ll_value = rand() % 64;
                }
            }
            else
            {
                num_input = sscanf(line, "%lld %lld", &ll_row, &ll_col);

		if (ll_row < 1 || ll_col < 1)
                {
                    fprintf(stderr,
			    "\nInvalid graph: Bad MTX format, indices cannot start with 0.\n");
                    free(coo);
                    exit(1);
                }

                if (array && (num_input == 1))
                {
                    ll_value = ll_row;
                    ll_col   = edges_read / nodes;
                    ll_row   = edges_read - nodes * ll_col;
                } 

                else if (array || (num_input != 2))
                {
                    fprintf(stderr,
                            "Error parsing MARKET graph: badly formed edge\n");
                    if (coo) free(coo);
                    return -1;
                }
            }

            if (LOAD_VALUES)
            {
                coo[edges_read].val = ll_value;
            }
            if (reversed && !undirected)
            {
                coo[edges_read].col = ll_row - 1;   // zero-based array
                coo[edges_read].row = ll_col - 1;   // zero-based array
                ordered_rows = false;
            }
            else
            {
                coo[edges_read].row = ll_row - 1;   // zero-based array
                coo[edges_read].col = ll_col - 1;   // zero-based array
                ordered_rows = false;
            }

            edges_read++;

            if (undirected)
            {
                // Go ahead and insert reverse edge
                coo[edges_read].row = ll_col - 1;       // zero-based array
                coo[edges_read].col = ll_row - 1;       // zero-based array

                if (LOAD_VALUES)
                {
                    coo[edges_read].val = ll_value * (skew ? -1 : 1);
                }

                ordered_rows = false;
                edges_read++;
            }
        }
    }

    if (is_first_line_empty)
    {
	fprintf(stderr, "\nInvalid graph, first line cannot be empty.\n");
        exit(1);
    }

    if (coo == NULL)
    {
        fprintf(stderr, "No graph found\n");
        return -1;
    }

    if (edges_read != edges)
    {
        fprintf(stderr,
                "Error parsing MARKET graph: only %lld/%lld edges read\n",
                (long long)edges_read, (long long)edges);
        if (coo) free(coo);
        return -1;
    }

    time_t mark1 = time(NULL);
    if (!quiet)
    {
        printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
        fflush(stdout);
    }

    // Convert COO to CSR
    csr_graph.template FromCoo<LOAD_VALUES>(output_file, coo,
                                            nodes, edges, ordered_rows,
                                            undirected, reversed, quiet);

    free(coo);
    fflush(stdout);

    return 0;
}

/**
 * @brief (Special for SM) Read csr arrays directly instead of transfer from coo format
 * @param[in] f_in          Input graph file name.
 * @param[in] f_label       Input label file name.
 * @param[in] csr_graph     Csr graph object to store the graph data.
 * @param[in] undirected    Is the graph undirected or not?
 */
template <bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int ReadCsrArrays_SM(char *f_in, char *f_label, Csr<VertexId, SizeT, Value> &csr_graph,
                  bool undirected, bool quiet)
{
    csr_graph.template FromCsr_SM<LOAD_VALUES>(f_in, f_label, quiet);
    return 0;
}

/**
 * @brief Read csr arrays directly instead of transfer from coo format
 * @param[in] f_in          Input graph file name.
 * @param[in] csr_graph     Csr graph object to store the graph data.
 * @param[in] undirected    Is the graph undirected or not?
 * @param[in] reversed      Whether or not the graph is inversed.
 */
template <bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int ReadCsrArrays(char *f_in, Csr<VertexId, SizeT, Value> &csr_graph,
                  bool undirected, bool reversed, bool quiet)
{
    csr_graph.template FromCsr<LOAD_VALUES>(f_in, quiet);
    return 0;
}


/**
 * \defgroup Public Interface
 * @{
 */


/**
 * @brief Loads a MARKET-formatted CSR graph from the specified file.
 *
 * @param[in] mm_filename Graph file name, if empty, it is loaded from STDIN.
 * @param[in] output_file Output file name for binary i/o.
 * @param[in] csr_graph Reference to CSR graph object. @see Csr
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed Is the graph reversed or not?
 * @param[in] quiet If true, print no output
 *
 * \return If there is any File I/O error along the way. 0 for no error.
 */
template<bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int BuildMarketGraph(
    char *mm_filename,
    char *output_file,
    Csr<VertexId, SizeT, Value> &csr_graph,
    bool undirected,
    bool reversed,
    bool quiet = false)
{
    FILE *_file = fopen(output_file, "r");
    if (_file)
    {
        fclose(_file);
        if (ReadCsrArrays<LOAD_VALUES>(
                    output_file, csr_graph, undirected, reversed, quiet) != 0)
        {
            return -1;
        }
    }
    else
    {
        if (mm_filename == NULL)
        {
            // Read from stdin
            if (!quiet)
            {
                printf("Reading from stdin:\n");
            }
            if (ReadMarketStream<LOAD_VALUES>(
                        stdin, output_file, csr_graph, undirected, reversed) != 0)
            {
                return -1;
            }
        }
        else
        {
            // Read from file
            FILE *f_in = fopen(mm_filename, "r");
            if (f_in)
            {
                if (!quiet)
                {
                    printf("Reading from %s:\n", mm_filename);
                }
                if (ReadMarketStream<LOAD_VALUES>(
                            f_in, output_file, csr_graph,
                            undirected, reversed, quiet) != 0)
                {
                    fclose(f_in);
                    return -1;
                }
                fclose(f_in);
            }
            else
            {
                perror("Unable to open file");
                return -1;
            }
        }
    }
    return 0;
}


/**
 * @brief (Special for SM) Loads a MARKET-formatted CSR graph from the specified file.
 *
 * @param[in] mm_filename Graph file name, if empty, it is loaded from STDIN.
 * @param[in] label_filename Label file name, if empty, it is loaded from STDIN.
 * @param[in] output_file Output file name for binary i/o.
 * @param[in] output_label Output label file name for binary i/o.
 * @param[in] csr_graph Reference to CSR graph object. @see Csr
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed   Whether or not the graph is inversed.
 * @param[in] quiet If true, print no output
 *
 * \return If there is any File I/O error along the way. 0 for no error.
 */
template<bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int BuildMarketGraph_SM(
    char *mm_filename,
    char *label_filename,
    char *output_file,
    char *output_label,
    Csr<VertexId, SizeT, Value> &csr_graph,
    bool undirected,
    bool reversed,
    bool quiet = false)
{
    FILE *_file = fopen(output_file, "r");
    FILE *_label = fopen(output_label, "r");
    if (_file && _label)
    {
        fclose(_file);
        fclose(_label);
            if (ReadCsrArrays_SM<LOAD_VALUES>(
                    output_file, output_label, csr_graph, undirected, quiet) != 0)
                return -1;
    }
    else
    {
        if (mm_filename == NULL && label_filename == NULL)
        {
            // Read from stdin
            if (!quiet)
            {
                printf("Reading from stdin:\n");
            }
            if (ReadMarketStream<false>(
                        stdin, output_file, csr_graph, undirected, reversed) != 0)
            {
                return -1;
            }
        }
        else
        {
            // Read from file
            FILE *f_in = fopen(mm_filename, "r");
            if (f_in)
            {	
                if (!quiet)
                {
                    printf("Reading from %s:\n", mm_filename);
                }
                if (ReadMarketStream<false>(
                            f_in, output_file, csr_graph,
                            undirected, reversed, quiet) != 0)
                {
                    fclose(f_in);
                    return -1;
                }
                fclose(f_in);
            }
            else
            {
                perror("Unable to open graph file");
                return -1;
            }
	    
	    // Read from label
            FILE *label_in = fopen(label_filename, "r");
	    if(label_in)
	    {
		if(!quiet) printf("Reading form %s:\n", label_filename);
 		if(ReadLabelStream<LOAD_VALUES>(label_in, output_label, csr_graph, quiet) != 0)
		{
		    fclose(label_in);
		    return -1;
		}
		fclose(label_in);
	    }
	    else
	    {
		perror("Unable to open label file");
		return -1;
	    }
	    
        }
    }
    return 0;
}

/**
 * @brief read in graph function read in graph according to its type.
 *
 * @tparam LOAD_VALUES 
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] file_in    Input MARKET graph file.
 * @param[in] file_label Input label file.
 * @param[in] graph      CSR graph object to store the graph data.
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed   Whether or not the graph is inversed.
 * @param[in] quiet     Don't print out anything to stdout
 *
 * \return int Whether error occurs (0 correct, 1 error)
 */
template <bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int BuildMarketGraph_SM(
    char *file_in,
    char *file_label,
    Csr<VertexId, SizeT, Value> &graph,
    bool undirected,
    bool reversed,
    bool quiet = false)
{
    // seperate the graph path and the file name
    char *temp1 = strdup(file_in);
    char *temp2 = strdup(file_in);
    char *file_path = dirname (temp1);
    char *file_name = basename(temp2);
    char *temp3, *temp4, *label_path, *label_name;
  if(LOAD_VALUES){
    // seperate the label path and the file name
    temp3 = strdup(file_label);
    temp4 = strdup(file_label);
    label_path = dirname (temp3);
    label_name = basename(temp4);
  }
    if (undirected)
    {
        char ud[256];  // undirected graph
	char lb[256]; // label
        sprintf(ud, "%s/.%s.ud.%d.%s%s%sbin", file_path, file_name, 0,
            ((sizeof(VertexId) == 8) ? "64bVe." : ""), 
            ((sizeof(Value   ) == 8) ? "64bVa." : ""), 
            ((sizeof(SizeT   ) == 8) ? "64bSi." : ""));

      if(LOAD_VALUES){
        sprintf(lb, "%s/.%s.lb.%s%sbin", label_path, label_name, 
            ((sizeof(VertexId) == 8) ? "64bVe." : ""), 
            ((sizeof(Value   ) == 8) ? "64bVa." : "")); 
      }
       //for(int i=0; ud[i]; i++) printf("%c",ud[i]); printf("\n");
       //for(int i=0; lb[i]; i++) printf("%c",lb[i]); printf("\n");
        if (BuildMarketGraph_SM<LOAD_VALUES>(file_in, file_label, ud, lb, graph,
                    true, reversed, quiet) != 0)
            return 1;
    }
    else
    {
        fprintf(stderr, "Unspecified Graph Type.\n");
        return 1;
    }
    return 0;
}
/**
 * @brief read in graph function read in graph according to its type.
 *
 * @tparam LOAD_VALUES
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] file_in    Input MARKET graph file.
 * @param[in] graph      CSR graph object to store the graph data.
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed   Whether or not the graph is inversed.
 * @param[in] quiet     Don't print out anything to stdout
 *
 * \return int Whether error occurs (0 correct, 1 error)
 */
template <bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int BuildMarketGraph(
    char *file_in,
    Csr<VertexId, SizeT, Value> &graph,
    bool undirected,
    bool reversed,
    bool quiet = false)
{
    // seperate the graph path and the file name
    char *temp1 = strdup(file_in);
    char *temp2 = strdup(file_in);
    char *file_path = dirname (temp1);
    char *file_name = basename(temp2);

    if (undirected)
    {
        char ud[256];  // undirected graph
        sprintf(ud, "%s/.%s.ud.%d.%s%s%sbin", file_path, file_name, (LOAD_VALUES?1:0),
            ((sizeof(VertexId) == 8) ? "64bVe." : ""), 
            ((sizeof(Value   ) == 8) ? "64bVa." : ""), 
            ((sizeof(SizeT   ) == 8) ? "64bSi." : ""));
        if (BuildMarketGraph<LOAD_VALUES>(file_in, ud, graph,
                    true, false, quiet) != 0)
            return 1;
    }
    else if (!undirected && reversed)
    {
        char rv[256];  // reversed graph
        sprintf(rv, "%s/.%s.rv.%d.%s%s%sbin", file_path, file_name, (LOAD_VALUES?1:0),
            ((sizeof(VertexId) == 8) ? "64bVe." : ""), 
            ((sizeof(Value   ) == 8) ? "64bVa." : ""), 
            ((sizeof(SizeT   ) == 8) ? "64bSi." : ""));
        if (BuildMarketGraph<LOAD_VALUES>(file_in, rv, graph,
                    false, true, quiet) != 0)
            return 1;
    }
    else if (!undirected && !reversed)
    {
        char di[256];  // directed graph
        sprintf(di, "%s/.%s.di.%d.%s%s%sbin", file_path, file_name, (LOAD_VALUES?1:0),
            ((sizeof(VertexId) == 8) ? "64bVe." : ""), 
            ((sizeof(Value   ) == 8) ? "64bVa." : ""), 
            ((sizeof(SizeT   ) == 8) ? "64bSi." : ""));
        if (BuildMarketGraph<LOAD_VALUES>(file_in, di, graph,
                    false, false, quiet) != 0)
            return 1;
    }
    else
    {
        fprintf(stderr, "Unspecified Graph Type.\n");
        return 1;
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
