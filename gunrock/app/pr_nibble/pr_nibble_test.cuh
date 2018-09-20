// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_nibble_test.cu
 *
 * @brief Test related functions for pr_nibble
 */

#pragma once

#include <iostream>

namespace gunrock {
namespace app {
// <DONE> change namespace
namespace pr_nibble {
// </DONE>


/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference pr_nibble ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    // <DONE> add problem specific inputs and outputs 
    util::Parameters &parameters,
    typename GraphT::VertexT ref_node,
    typename GraphT::ValueT *values,
    // </DONE>
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::ValueT ValueT;
    typedef typename GraphT::VertexT VertexT;

    // Compute graph statistics
    SizeT nodes          = graph.nodes;
    ValueT num_edges      = (ValueT)graph.edges / 2;
    ValueT log_num_edges = log2(num_edges);

    // Load parameters
    ValueT vol   = parameters.Get<ValueT>("vol");
    ValueT phi   = parameters.Get<ValueT>("phi");
    ValueT eps   = parameters.Get<ValueT>("eps");
    int max_iter = parameters.Get<int>("max-iter");
    
    ValueT alpha = pow(phi, 2) / (225.0 * log(100.0 * sqrt(num_edges)));
    
    // rho
    ValueT rho;
    if(1.0f + log2(vol) > log_num_edges) {
        rho = log_num_edges;
    } else {
        rho = 1.0f + log2(vol);
    }
    rho = pow(2.0f, rho);
    rho = 1.0 / rho;
    rho *= 1.0 / (48.0 * log_num_edges);

    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    // Algorithm arrays 
    ValueT *grad    = new ValueT[nodes];
    ValueT *q       = new ValueT[nodes];
    ValueT *y       = new ValueT[nodes];
    ValueT *z       = new ValueT[nodes];
    
    ValueT *d       = new ValueT[nodes];
    ValueT *d_sqrt  = new ValueT[nodes];
    ValueT *dn      = new ValueT[nodes];
    ValueT *dn_sqrt = new ValueT[nodes];
    
    for(SizeT i = 0; i < graph.nodes; ++i) {
        grad[i] = (ValueT)0;
        q[i]    = (ValueT)0;
        y[i]    = (ValueT)0;
        z[i]    = (ValueT)0;
        
        d[i]       = (ValueT)(graph.row_offsets[i + 1] - graph.row_offsets[i]);
        d_sqrt[i]  = sqrt(d[i]);
        dn[i]      = 1.0 / d[i];
        dn_sqrt[i] = 1.0 / d_sqrt[i];
    }
    
    int num_ref_nodes = 1; // HARDCODED
    
    grad[ref_node] = - (alpha / num_ref_nodes) * dn_sqrt[ref_node];
    
    // -- 
    // std::cout << "num_edges: "     << num_edges        << std::endl;
    // std::cout << "log_num_edges: " << log_num_edges    << std::endl;
    // std::cout << "alpha: "         << alpha            << std::endl;
    // std::cout << "rho: "           << rho              << std::endl;
    // std::cout << "vol: "           << vol              << std::endl;
    // std::cout << "eps: "           << eps              << std::endl;
    // std::cout << "max_iter: "      << max_iter         << std::endl;
    // printf("dn_sqrt[ref_node] %.17g\n", dn_sqrt[ref_node]);
    // printf("grad[ref_node] %.17g\n", grad[ref_node]);
    // -- 
    
    ValueT scale_grad = -1.0 * grad[ref_node] * dn_sqrt[ref_node];
    int it = 0;
    
    while(true) {
        if(scale_grad <= rho * alpha * (1.0 + eps)) {
            // printf("breaking scale_grad\n");
            break;
        }
        if(it >= max_iter) {
            // printf("breaking iter\n");
            break;
        }
        
        for(int idx = 0; idx < nodes; ++idx) {
            ValueT q_old = q[idx];
            z[idx] = y[idx] - grad[idx];
            if(z[idx] == 0) {
                continue;
            }
            ValueT thresh = rho * alpha * d_sqrt[idx];
            if(z[idx] >= thresh) {
                q[idx] = z[idx] - thresh;
            } else if(-z[idx] >= thresh) {
                q[idx] = z[idx] + thresh;
            } else {
                q[idx] = (ValueT)0;
            }
            
            if(it == 0) {
                y[idx] = q[idx];
            } else {
                ValueT beta = (1.0 - sqrt(alpha)) / (1.0 + sqrt(alpha));
                y[idx] = q[idx] + beta * (q[idx] - q_old);
            }
        }
        
        for(int idx = 0; idx < nodes; ++idx) {
            grad[idx] = y[idx] * (1.0 + alpha) / 2.0;
        }
        
        for(int idx = 0; idx < nodes; ++idx) {
            SizeT num_neighbors = graph.GetNeighborListLength(idx);
            for(int offset = 0; offset < num_neighbors; ++offset) {
                VertexT dest = graph.GetEdgeDest(graph.GetNeighborListOffset(idx) + offset);
                ValueT grad_update = - dn_sqrt[idx] * y[idx] * dn_sqrt[dest] * (1.0 - alpha) / 2.0;
                grad[dest] += grad_update;
            }
        }
        
        grad[ref_node] = grad[ref_node] - (alpha / num_ref_nodes) * dn_sqrt[ref_node];
        
        scale_grad = -1;
        for(int idx = 0; idx < nodes; ++idx) {
            ValueT tmp = abs(grad[idx] * dn_sqrt[idx]);
            if(tmp > scale_grad) {
                scale_grad = tmp;
            }
        }
        
        it += 1;
    }
    
    for(SizeT i = 0; i < graph.nodes; ++i) {
        values[i] = abs(q[i] * d_sqrt[i]);
    }
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

/**
 * @brief Validation of pr_nibble results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(
             util::Parameters &parameters,
             GraphT           &graph,
             typename GraphT::ValueT *h_values,
             typename GraphT::ValueT *ref_values,
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::SizeT   SizeT;

    bool quiet = parameters.Get<bool>("quiet");

    // printf("Validate_Results: \n");
    // for(int i = 0; i < graph.nodes; ++i) {
    //     printf("%d %.17g %.17g\n", i, h_values[i], ref_values[i]);
    // }

    // Check agreement (within a small tolerance)
    SizeT num_errors = 0;
    ValueT tolerance = 0.00001;
    for (SizeT i = 0; i < graph.nodes; i++) {
        if (h_values[i] != ref_values[i]) {
            float err = abs(h_values[i] - ref_values[i]) / abs(ref_values[i]);
            if (err > tolerance) {
                num_errors++;
                util::PrintMsg("FAIL: [" + std::to_string(i)
                    + "]: " + std::to_string(h_values[i])
                    + " != " + std::to_string(ref_values[i]));
            }
        }
    }

    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);

    return num_errors;
}

} // namespace pr_nibble
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
