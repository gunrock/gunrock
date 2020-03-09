/*
 * @brief Unit tests for dynamic graph operations
 * @file test_dynamic_graph.h
 */

#include <gunrock/graphio/graphio.cuh>
#include <unordered_set>

using namespace gunrock;

TEST(dynamicGraph, buildWeighted) {
    
    using SizeT = uint32_t;
    using VertexT = uint32_t;
    using ValueT = uint32_t;

    using WeightedCSRGraphT = app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_EDGE_VALUES |
                                    graph::HAS_CSR>;
    using WeightedDYNGraphT = app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_EDGE_VALUES |
                                    graph::HAS_CSR |
                                    graph::HAS_DYN>;

    //rng
    std::mt19937 rng(0);

    //graph parameters    
    SizeT nodes = 100; 
    SizeT edges_per_node = 5;
    SizeT edges = nodes * edges_per_node;
    
    //reference CSR graph
    WeightedCSRGraphT ref_graph;
    auto ref_csr_graph = ref_graph.csr();
    ref_csr_graph.Allocate(nodes, edges, util::HOST);

    //generate a random reference CSR graph
    SizeT cur_offset = 0;
    for(auto v = 0; v < nodes; v++){
        std::unordered_set<VertexT> v_edges;
        do{
            VertexT random_edge = rng() % nodes;
            if(random_edge != v){ //no self-edges
                v_edges.insert(random_edge);
            }
        }while(v_edges.size() != edges_per_node);

        //populate the CSR with the adj. list
        ref_csr_graph.row_offsets[v] = cur_offset;
        for(const auto &e : v_edges){
            ref_csr_graph.column_indices[cur_offset] = e;
            ref_csr_graph.edge_values[cur_offset] = rng();
            ++cur_offset;
        }
    }
    ref_csr_graph.row_offsets[nodes] = cur_offset;

    //convert CSR to dynamic graph
    WeightedDYNGraphT result_graph;
    auto result_dynamic_graph = result_graph.dyn();
    result_dynamic_graph.FromCsr(ref_csr_graph);

    //convert dynamic graph back to CSR
    auto result_csr_graph = result_graph.csr();
    result_csr_graph.Allocate(nodes, edges, util::HOST | util::DEVICE);
    result_dynamic_graph.ToCsr(result_csr_graph);
    result_csr_graph.Move(util::DEVICE, util::HOST);

    //sort both CSR graphs
    result_csr_graph.Sort();
    ref_csr_graph.Sort();

    //Compare
    EXPECT_EQ(ref_csr_graph.row_offsets[nodes], result_csr_graph.row_offsets[nodes]);

    for(auto v = 0; v < nodes; v++){
        
        EXPECT_EQ(ref_csr_graph.row_offsets[v], result_csr_graph.row_offsets[v]);

        auto start_eid = ref_csr_graph.row_offsets[v];
        auto end_eid = ref_csr_graph.row_offsets[v + 1];

        for(auto eid = start_eid; eid < end_eid; eid++){
        	//std::cout <<  v << " "<< eid << ": " << ref_csr_graph.column_indices[eid] << " " << result_csr_graph.column_indices[eid] <<std::endl;
        	//std::cout <<  v << " "<< eid << ": " << ref_csr_graph.edge_values[eid] << " " << result_csr_graph.edge_values[eid] <<std::endl;
            EXPECT_EQ(ref_csr_graph.column_indices[eid], result_csr_graph.column_indices[eid]);
            EXPECT_EQ(ref_csr_graph.edge_values[eid], result_csr_graph.edge_values[eid]);
        }
    }
    

    result_dynamic_graph.Release(util::HOST | util::DEVICE);
    ref_graph.Release(util::HOST | util::DEVICE);

}
