/*
 * @brief Unit tests for dynamic graph operations
 * @file test_dynamic_graph.h
 */

#include <gunrock/graphio/graphio.cuh>
#include <unordered_set>

using namespace gunrock;


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

using DynamicHostGraphT = std::vector<std::unordered_set<VertexT>>;

void RandomWeightedGraphToCsr(DynamicHostGraphT& ref_dyn, WeightedCSRGraphT& ref_graph, int const_weights = 0){

    //rng
    std::mt19937 rng(0);

    auto ref_csr_graph = ref_graph.csr();

    //Populate the CSR with the adj. list
    SizeT cur_offset = 0;
    for(auto v = 0; v < ref_dyn.size(); v++){
        auto& v_edges = ref_dyn[v];
        ref_csr_graph.row_offsets[v] = cur_offset;
        for(const auto &e : v_edges){
            ref_csr_graph.column_indices[cur_offset] = e;
            if(const_weights == 0)
                ref_csr_graph.edge_values[cur_offset] = rng();
            else
                ref_csr_graph.edge_values[cur_offset] = const_weights;

            ++cur_offset;
        }
    }
    ref_csr_graph.row_offsets[ref_dyn.size()] = cur_offset;
    
}
void GenerateRandomWeightedGraph(DynamicHostGraphT& ref_dyn, SizeT nodes, SizeT edges, SizeT edges_per_node, bool undirected_graph){

    //rng
    std::mt19937 rng(0);

    //generate a random reference  graph
    for(VertexT v = 0; v < nodes; v++){
        auto& v_edges = ref_dyn[v];
        SizeT added_edges = 0;
        do{
            VertexT random_edge = rng() % nodes;
            if(random_edge != v){
                auto res = v_edges.insert(random_edge);
                if(res.second){
                	if(undirected_graph){
                		ref_dyn[random_edge].insert(v);
                	}
                	added_edges++;
                }
            }
        }while(added_edges != edges_per_node);
    }
}
void CompareWeightedCSRs(WeightedCSRGraphT& ref_graph, WeightedDYNGraphT& result_graph){
    //sort both CSR graphs
    auto& result_csr_graph = result_graph.csr();
    auto& ref_csr_graph = ref_graph.csr();
    
    result_csr_graph.Sort();
    ref_csr_graph.Sort();

    SizeT nodes = result_csr_graph.nodes;

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
}


TEST(dynamicGraph, buildDirectedWeighted) {

    //rng
    std::mt19937 rng(0);

    //graph parameters    
    SizeT nodes = 100; 
    SizeT edges_per_node = 5;
    SizeT edges = nodes * edges_per_node;
    bool directed = true;

    //reference CSR graph
    WeightedCSRGraphT ref_graph;
    auto& ref_csr_graph = ref_graph.csr();
    ref_csr_graph.Allocate(nodes, edges, util::HOST);
    ref_csr_graph.directed = directed;

    DynamicHostGraphT ref_dyn(nodes);
    GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
    RandomWeightedGraphToCsr(ref_dyn, ref_graph);

    //convert CSR to dynamic graph
    WeightedDYNGraphT result_graph;
    auto& result_dynamic_graph = result_graph.dyn();
    result_dynamic_graph.FromCsr(ref_csr_graph);

    //convert dynamic graph back to CSR
    auto& result_csr_graph = result_graph.csr();
    result_graph.csr().Allocate(nodes, edges, util::HOST | util::DEVICE);
    result_dynamic_graph.ToCsr(result_csr_graph);
    result_csr_graph.Move(util::DEVICE, util::HOST);

    CompareWeightedCSRs(ref_graph, result_graph);

    result_dynamic_graph.Release(util::HOST | util::DEVICE);
    ref_graph.Release(util::HOST | util::DEVICE);

}



TEST(dynamicGraph, buildUndirectedWeighted) {

    //graph parameters    
    SizeT nodes = 100; 
    SizeT edges_per_node = 5;
    SizeT edges = nodes * edges_per_node * 2;
    bool directed = false;

    //reference CSR graph
    WeightedCSRGraphT ref_graph;
    auto& ref_csr_graph = ref_graph.csr();
    ref_csr_graph.Allocate(nodes, edges, util::HOST);
    ref_csr_graph.directed = directed;

    DynamicHostGraphT ref_dyn(nodes);
    GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
    RandomWeightedGraphToCsr(ref_dyn, ref_graph);

    //convert CSR to dynamic graph
    WeightedDYNGraphT result_graph;
    auto& result_dynamic_graph = result_graph.dyn();
    result_dynamic_graph.FromCsr(ref_csr_graph);


    //convert dynamic graph back to CSR
    auto& result_csr_graph = result_graph.csr();
    result_csr_graph.Allocate(nodes, edges, util::HOST | util::DEVICE);
    result_dynamic_graph.ToCsr(result_csr_graph);
    result_csr_graph.Move(util::DEVICE, util::HOST);

    CompareWeightedCSRs(ref_graph, result_graph);

    result_dynamic_graph.Release(util::HOST | util::DEVICE);
    ref_graph.Release(util::HOST | util::DEVICE);
}




TEST(dynamicGraph, insertUndirectedWeighted) {
    
    //graph parameters    
    SizeT nodes = 100; 
    SizeT edges_per_node = 5;
    SizeT edges = nodes * edges_per_node * 2;
    bool directed = false;
    
    //reference CSR graph
    WeightedCSRGraphT ref_graph;
    auto& ref_csr_graph = ref_graph.csr();
    ref_csr_graph.Allocate(nodes, edges, util::HOST);
    ref_csr_graph.directed = directed;

    DynamicHostGraphT ref_dyn(nodes);
    GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
    RandomWeightedGraphToCsr(ref_dyn, ref_graph, 1);

    //convert CSR to dynamic graph
    WeightedDYNGraphT result_graph;
    auto& result_dynamic_graph = result_graph.dyn();
    result_dynamic_graph.FromCsr(ref_csr_graph);

    // generate a random batch of edges to insert
    using PairT = uint2;
    SizeT batch_size = 100;
    util::Array1D<SizeT, PairT> edges_batch;
    util::Array1D<SizeT, ValueT> edges_batch_values;
    edges_batch.Allocate(batch_size, util::HOST | util::DEVICE);
    edges_batch_values.Allocate(batch_size, util::HOST | util::DEVICE);
    std::mt19937 rng(0);

    for(auto e = 0; e < batch_size; e++){
        VertexT edge_src = rng() % nodes;
        VertexT edge_dst = rng() % nodes;
        edges_batch[e] = make_uint2(edge_src, edge_dst);
        edges_batch_values[e] = 1;
    }

    result_dynamic_graph.InsertEdgesBatch(edges_batch,
                                          edges_batch_values, 
                                          batch_size);
    //result_dynamic_graph.InsertEdgesBatch();

    //Apply batch to host graph & generate values as well
    SizeT new_edges_count = edges;
    for(auto e = 0; e < batch_size; e++){
        if(edges_batch[e].x != edges_batch[e].y){
            auto res_0 = ref_dyn[edges_batch[e].x].insert(edges_batch[e].y);
            auto res_1 = ref_dyn[edges_batch[e].y].insert(edges_batch[e].x);
            if(res_0.second) new_edges_count++;
            if(res_1.second) new_edges_count++;
        }
    }

    //New static graph
    WeightedCSRGraphT ref_graph_updated;
    auto& ref_csr_graph_updated = ref_graph_updated.csr();
    ref_csr_graph_updated.Allocate(nodes, new_edges_count, util::HOST);
    ref_csr_graph_updated.directed = directed;
    RandomWeightedGraphToCsr(ref_dyn, ref_graph_updated, 1);


    //convert dynamic graph back to CSR
    auto& result_csr_graph = result_graph.csr();
    result_csr_graph.Allocate(nodes, new_edges_count, util::HOST | util::DEVICE);
    result_dynamic_graph.ToCsr(result_csr_graph);
    result_csr_graph.Move(util::DEVICE, util::HOST);

    CompareWeightedCSRs(ref_graph_updated, result_graph);

    result_dynamic_graph.Release(util::HOST | util::DEVICE);
    ref_graph.Release(util::HOST | util::DEVICE);
    ref_graph_updated.Release(util::HOST | util::DEVICE);

}
