#ifndef ERROR_UTIL
#include "error_util.cuh"
#endif

#include <vector>
#include <queue> 

using namespace std;

namespace host {

    template <typename VertexId, typename SizeT, typename Rank>
    void PrValid(Csr<VertexId, SizeT> &csr,  PageRank<VertexId, SizeT, Rank> &pr)
    {
        float lambda = pr.lambda;
        float epsilon = pr.epsilon;
    
        VertexId nodes = csr.nodes;
    
        Rank *h_rank = (Rank *)malloc(sizeof(Rank)*nodes);
        MALLOC_CHECK(h_rank);
        Rank *h_res = (Rank *)malloc(sizeof(Rank)*nodes);
        MALLOC_CHECK(h_res);
        memset(h_res, 0, sizeof(Rank)*nodes);
    
        queue<VertexId> wl;
        for(VertexId i=0; i<nodes; i++)
        {
            h_rank[i] = 1.0-lambda;
            wl.push(i);
    
            VertexId neigh_len = csr.row_offset[i+1]-csr.row_offset[i];
            for(int j=0; j<neigh_len; j++)
            {
                VertexId neighbor = csr.column_indices[csr.row_offset[i]+j];
                h_res[neighbor] = h_res[neighbor] + (1.0-lambda)*lambda/neigh_len;
            }
        } //for vertices
        //finish res and rank init

        while(!wl.empty())
        {
            VertexId node_item = wl.front();
            wl.pop();
            h_rank[node_item] = h_rank[node_item]+h_res[node_item];
            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            Rank res_owner = h_res[node_item];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                Rank res_old = h_res[dest_item];
                h_res[dest_item] = h_res[dest_item] + res_owner*lambda/(Rank)(destEnd-destStart); 
                if(res_old < epsilon && h_res[dest_item] > epsilon)
                    wl.push(dest_item);
            }
            h_res[node_item] = 0.0;
    
        }//while worklist
    
        Rank totalRank=0.0;
        Rank totalRes = 0.0;
        for(VertexId i=0; i<nodes; i++)
        {
            totalRank = totalRank + h_rank[i];
            totalRes= totalRes+ h_res[i];
        }
        
     //   for(int i=0; i<nodes; i++)
     //       cout << h_rank[i] << " ";
     //   cout << endl;

        cout << "CPU total mass: " << totalRank + totalRes/(1.0-lambda) << " CPU total res: " << totalRes << " CPU total rank: " << totalRank << endl;
    
        float error=0.0;
        Rank sum_rank=0.0;
        Rank sum_res = 0.0;
        uint32_t large = 0;
        for(VertexId i=0; i<pr.nodes; i++)
        {
     //       error = error + abs(check_rank[i]-h_rank[i]/totalRank);
            error = error + abs(pr.rank[i]-h_rank[i]);
            sum_rank = sum_rank + pr.rank[i];
        }
        cout << "GPU rank: sum of rank "<< sum_rank << " error from CPU "<< error << "\n";
        error = 0.0;
        for(VertexId i=0; i<pr.nodes; i++)
        {
            error = error + abs(pr.res[i]-h_res[i]);
            sum_res = sum_res + pr.res[i];
            if(pr.res[i] > epsilon)
                large++;

        }
        cout << "GPU res: sum of res "<< sum_res << " error from CPU "<< error << " "<<large << " number of res larger than "<< epsilon << "\n";
        cout << endl;

        cout<<"GPU sum_rank: "<< sum_rank << " GPU sum_res: "<< sum_res << " GPU total mass: "<< sum_rank+sum_res/(1.0-lambda) << endl;
   //     if(error > 0.01) cout << "FAILE\n";

        cout << "Print the first 20 res: \n";
        cout << "host:\n";
        for(int i=0; i<20; i++)
            cout << h_rank[i] << " ";
        cout << endl;
        cout << "device:\n";
        for(int i=0; i<20; i++)
            cout << pr.rank[i] << " ";
        cout << endl;
    }//PrValid

    template <typename VertexId, typename SizeT, typename Rank>
    void PrInitValid(Csr<VertexId, SizeT> &csr,  PageRank<VertexId, SizeT, Rank> &pr)
    {
        float lambda = pr.lambda;
    
        VertexId nodes = csr.nodes;
    
        Rank *h_rank = (Rank *)malloc(sizeof(Rank)*nodes);
        MALLOC_CHECK(h_rank);
        Rank *h_res = (Rank *)malloc(sizeof(Rank)*nodes);
        MALLOC_CHECK(h_res);
        memset(h_res, 0, sizeof(Rank)*nodes);
    
        queue<VertexId> wl;
        for(VertexId i=0; i<nodes; i++)
        {
            h_rank[i] = 1.0-lambda;
            wl.push(i);
    
            VertexId neigh_len = csr.row_offset[i+1]-csr.row_offset[i];
            for(int j=0; j<neigh_len; j++)
            {
                VertexId neighbor = csr.column_indices[csr.row_offset[i]+j];
                h_res[neighbor] = h_res[neighbor] + (1.0-lambda)*lambda/neigh_len;
            }
        } //for vertices
        //finish res and rank init

        Rank *check_res = (Rank *)malloc(sizeof(Rank)*nodes);
        MALLOC_CHECK(check_res);
        CUDA_CHECK(cudaMemcpy(check_res, pr.res, sizeof(Rank)*nodes, cudaMemcpyDeviceToHost));
        float error=0.0;
        for(int i=0; i<pr.nodes; i++)
        {
            error = error + abs(check_res[i]-h_res[i]);
        }
        cout <<"\nerror :" << error << endl;
        CUDA_CHECK(cudaMemcpy(check_res, pr.rank, sizeof(Rank)*pr.nodes, cudaMemcpyDeviceToHost));
        for(int i=0; i<pr.nodes; i++)
            if(check_res[i]!=1.0-lambda)
                cout << "Rank: " << check_res[i] << " not equal to " << 1.0-lambda << endl;
        free(check_res);
    }//PrInitValid

} //namespace
