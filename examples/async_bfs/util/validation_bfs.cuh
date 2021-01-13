#ifndef ERROR_UTIL
#include "error_util.cuh"
#endif

#include <vector>
#include <queue> 

using namespace std;
void find_mother(int vertex, int *depth, int *h_depth, Csr<int, int> &csr)
{
    for(int i=0; i<csr.edges; i++)
    {
        if(csr.column_indices[i] == vertex)
        {
            for(int j=0; j<csr.nodes+1; j++)
            {
                if(i>=csr.row_offset[j] && i<csr.row_offset[j+1])
                    cout << "vertex "<< vertex << " has mother "<< j << " indices " << i<<" csr.row_ffset "<< csr.row_offset[j]<<" , "<< csr.row_offset[j+1] << ", h_depth["<< j<<"]: "<< h_depth[j] <<" depth["<<j<<"]:"<< depth[j] <<endl;
            }
        }
    }
}

namespace host {

    template <typename VertexId, typename SizeT>
    void BFSValid(Csr<VertexId, SizeT> &csr,  BFS<VertexId, SizeT> &bfs, VertexId source)
    {
        VertexId nodes = csr.nodes;
    
        VertexId *h_depth= (VertexId*)malloc(sizeof(VertexId)*nodes);
        MALLOC_CHECK(h_depth);
        for(int i=0; i<nodes; i++)
            h_depth[i] = nodes+1;
    
        queue<BFSEntry<VertexId>> wl;
        BFSEntry<VertexId> startEntry(source, VertexId(0));
        wl.push(startEntry);
        h_depth[source] = 0;
        int enqueued_nodes = 1;

        //finish init

        while(!wl.empty())
        {
            BFSEntry<VertexId> entry = wl.front();
            VertexId node_item = entry.node;
            VertexId depth = entry.dep;
            wl.pop();
       //     h_depth[node_item] = min(h_depth[node_item], depth);

            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                h_depth[dest_item] = min(depth+1, h_depth[dest_item]);
                if(h_depth[dest_item] == depth+1)
                {
                    BFSEntry<VertexId> entry(dest_item, depth+1);
                    wl.push(entry);
                    enqueued_nodes++;
                }
            }
        }//while worklist

        int error=0.0;
        for(int i=0; i<bfs.nodes; i++)
        {
            error = error+ abs(h_depth[i]-bfs.depth[i]);
            if(h_depth[i]!=bfs.depth[i]) 
            {
                cout << "h_depth["<<i<<"]: "<< h_depth[i] << "  bfs.depth["<<i<<"]: "<< bfs.depth[i]<<endl;
                find_mother(i, h_depth, bfs.depth, csr);
            }
        }

        cout << "enqueued nodes: "<< enqueued_nodes << endl;
        cout << "ERROR between CPU and GPU implimentation: "<< error <<  endl;
    
        cout << "Print the first 20 depth: \n";
        cout << "host:\n";
        for(int i=0; i<20; i++)
            cout << h_depth[i] << " ";
        cout << endl;
        cout << "device:\n";
        for(int i=0; i<20; i++)
            cout << bfs.depth[i] << " ";
        cout << endl;
        free(h_depth);
    }//BFSValid

    template <typename VertexId, typename SizeT>
    void BFSValid2(Csr<VertexId, SizeT> &csr,  BFS<VertexId, SizeT> &bfs, VertexId source)
    {
        VertexId nodes = csr.nodes;
    
        VertexId *h_depth= (VertexId*)malloc(sizeof(VertexId)*nodes);
        MALLOC_CHECK(h_depth);
        for(int i=0; i<nodes; i++)
            h_depth[i] = nodes+1;
    
        queue<VertexId> wl;
        wl.push(source);
        h_depth[source] = 0;

        //finish init

        while(!wl.empty())
        {
            VertexId node_item = wl.front();
            VertexId depth = h_depth[node_item];
            wl.pop();

            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                if(depth+1 < h_depth[dest_item]) 
                {
                    h_depth[dest_item] = depth+1;
                    wl.push(dest_item);
                }
            }
        }//while worklist

        int error=0.0;
	int printserre = 0;
        for(int i=0; i<bfs.nodes; i++)
        {
            error = error+ abs(h_depth[i]-bfs.depth[i]);
            if(h_depth[i]!=bfs.depth[i] && printserre < 10) 
            {
                cout << "h_depth["<<i<<"]: "<< h_depth[i] << "  bfs.depth["<<i<<"]: "<< bfs.depth[i]<<endl;
		printserre++;
        //        find_mother(i, h_depth, bfs.depth, csr);
            }
        }

        cout << "enqueued nodes: "<< wl.size() << endl;
        cout << "ERROR between CPU and GPU implimentation: "<< error <<  endl;
    
        cout << "Print the first 20 depth: \n";
        cout << "host:\n";
        for(int i=0; i<20; i++)
            cout << h_depth[i] << " ";
        cout << endl;
        cout << "device:\n";
        for(int i=0; i<20; i++)
            cout << bfs.depth[i] << " ";
        cout << endl;
        free(h_depth);
    }//BFSValid

} //namespace
