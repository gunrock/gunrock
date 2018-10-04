#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <utility>
#include <map>
#include <set>
#include <sstream>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
//#include "graph.h"
#include <time.h>
#include <queue>

using namespace std;
typedef double captype;

/*-----------------------------------------------*/
int bfs(double** rGraph, int s, int t, int parent[], const int V) {
    // Create a visited array and mark all vertices as not visited
    bool *visited = new bool[V];
    memset(visited, 0, V*sizeof(visited[0]));

    // Create a queue, enqueue source vertex and mark source vertex
    // as visited
    queue <int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    // Standard BFS Loop
    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (visited[v] == false && rGraph[u][v] > 0) {
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }

    // If we reached sink in BFS starting from source, then return
    // true, else false
    return (visited[t] == true);
}

// A DFS based function to find all reachable vertices from s.  The function
// marks visited[i] as true if i is reachable from s.  The initial values in
// visited[] must be false. We can also use BFS to find reachable vertices
void dfs(double** rGraph, int s, bool visited[], const int V) {
    visited[s] = true;
    for (int i = 0; i < V; i++)
       if (abs(rGraph[s][i]) > 1e-6 && !visited[i])
           dfs(rGraph, i, visited, V);
}

// Prints the minimum s-t cut
double** minCut(double** graph, int s, int t, bool *visited, const int V) {
    int u, v;
    double max_flow = 0;
    // Create a residual graph and fill the residual graph with
    // given capacities in the original graph as residual capacities
    // in residual graph
    double** rGraph = new double*[V];  // rGraph[i][j] indicates residual capacity of edge i-j
    for (u = 0; u < V; u++) {
        rGraph[u] = new double[V];
        for (v = 0; v < V; v++) {
             rGraph[u][v] = graph[u][v];
        }
    }


    int *parent = new int[V];  // This array is filled by BFS and to store path

    // Augment the flow while there is a path from source to sink
    int counter = 0;
    while (bfs(rGraph, s, t, parent, V)) {
        // Find minimum residual capacity of the edges along the
        // path filled by BFS. Or we can say find the maximum flow
        // through the path found.
        double path_flow = INT_MAX;
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        // update residual capacities of the edges and reverse edges
        // along the path
        //printf("residual\n");
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            //printf("%d -> %d\n", u, v);
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }
        counter++;
        max_flow += path_flow;
    }

    cout << "maxflow for this round is " << max_flow << endl;
    // Flow is maximum now, find vertices reachable from s
    memset(visited, false, V*sizeof(visited[0]));
    dfs(rGraph, s, visited, V);
    return rGraph;
}
/*----------------------------------------------*/


void graph_tv(double *Y,// value of nodes
        int n, //number of nodes
        int m, // number of edges
        int* e1,
        int* e2,
//an array of edges of size m. There is an edge edges1[i] -> edges2[i]
        double lambda,
        float erreur){

    double lambda1 = lambda;
    int V = n+2;
    double *tr_cap = new double[n];
    unsigned int label[n];

    double **graph = new double*[V];
    for (int h = 0; h < V; h++){
        graph[h] = new double[V];
        for (int w = 0; w < V; w++) graph[h][w] = 0;
    }

    for (int i = 0; i < m; i++) {
        graph[e1[i]+2][e2[i]+2] = lambda1;
        graph[e2[i]+2][e1[i]+2] = lambda1;
    }

    // normalization as did in parametric maxflow
    #define Alloc_Size 1024
    unsigned int l,*nextlabel,nlab,oldnlab,*nums;
    unsigned char flagstop, *inactivelabel, *alive;
    double *averages;
    captype *values;
    unsigned int maxlabel; // the size of the array "values

    maxlabel=Alloc_Size;
    nextlabel= (unsigned int *) malloc(sizeof(unsigned int)*maxlabel);
    inactivelabel= (unsigned char *) malloc(sizeof(unsigned char)*maxlabel);
    nums = (unsigned int *) malloc(sizeof(unsigned int)*maxlabel);
    averages = (captype *) malloc(sizeof(captype)*maxlabel);
    values = (captype *) malloc(sizeof(captype)*maxlabel);
    alive = (unsigned char *) malloc(sizeof(unsigned char)*n);

    double moy;
    int num;

    nlab=1;
    nextlabel[0]=0;
    memset(inactivelabel,0,sizeof(unsigned char)*maxlabel);

    moy=1.25; //need to change later !!!
    moy=0.; num=0;
    for (int i = 0; i < n; i++) {
        moy += Y[i];
        alive[i]=1;
        num++;
    }

    moy /= (double) num;
    values[0]=moy;

    for (int i = 0; i < n; i++) {
        double tem = Y[i] - moy;
        if(tem > 0) graph[0][i+2] = tem;
        if(tem < 0) graph[i+2][1] = -tem;
        label[i]=0;
    }
    int iter = 0;

    /* ----------------------------------------- */
    double** graph_old = new double*[V];  // rGraph[i][j] indicates residual capacity of edge i-j

    do {
        printf("iter %d\n", iter++);
        for (int u = 0; u < V; u++) {
            graph_old[u] = new double[V];
            for (int v = 0; v < V; v++) {
                graph_old[u][v] = graph[u][v];
            }
        }

        bool *visited = new bool[V];
        graph = minCut(graph, 0, 1, visited, V);

        memset(averages,0,nlab*sizeof(captype));
        memset(nums,0,nlab*sizeof(int));
        memset(nextlabel,0,nlab*sizeof(int));
        oldnlab=nlab;

        for (int i = 0; i < n; i++)
            if (alive[i]) {
                if (visited[i+2]==1) {
                    l=nextlabel[label[i]];
                    if (l==0) {
                        l=(nextlabel[label[i]]=nlab);
                        inactivelabel[l]=0;
                        nlab++;
                        averages[l]=0.; nums[l]=0;
                        nextlabel[l]=0;
                        values[l]=values[label[i]];
                        if (nlab==maxlabel) {
                            maxlabel+=Alloc_Size;
                            inactivelabel= (unsigned char *) realloc(inactivelabel,sizeof(unsigned char)*maxlabel);
                            nextlabel= (unsigned int *) realloc(nextlabel,sizeof(unsigned int)*maxlabel);
                            nums = (unsigned int *) realloc(nums,sizeof(unsigned int)*maxlabel);
                            averages = (captype *) realloc(averages,sizeof(captype)*maxlabel);
                            values = (captype *) realloc(values,sizeof(captype)*maxlabel);
                        }
                    } // end l == 0
                    label[i] = l;
                    averages[l] += graph[0][i+2]; // might be wrong!!!
                    nums[l]++;
                } else { // what_segment(i) == sink
                    l=label[i];
                    averages[l] -= graph[i+2][1];
                    nums[l]++;
                  for (int j = 0; j < n; j++) // easy to be wrong !!!
                    if (graph_old[i+2][j+2] != 0 && visited[j+2]) { graph[i+2][j+2]=0;}
                }
            }

            // tentative d'arret a precision zero
            // detection d'un label qui n'a pas ete coupe
            for (l=0;l<oldnlab;l++) if (!inactivelabel[l]) {
                if (nextlabel[l]==0) { averages[l]=0.; inactivelabel[l]=1;}
                else if (nums[l]==0) {
                    inactivelabel[l]=inactivelabel[nextlabel[l]]=1;
                    averages[nextlabel[l]]=0.;
                } else {
                    averages[l] /= (double) nums[l];
                    values[l] += averages[l];
                }
            } else averages[l]=0.;

            for (; l<nlab; l++) {
                averages[l] /= (double) nums[l];
                values[l]   += averages[l];
            }
        flagstop=0;

        for (int i = 0; i < n; i++)
            if (alive[i]) {
                l = label[i];
                if (inactivelabel[l] || (averages[l]<=erreur && averages[l]>=-erreur)) {
                      if (visited[i+2]==1) graph[0][i+2] = 0; //!!!
                      if (visited[i+2]==0) graph[i+2][1] = 0; //!!!
                      alive[i] = 0; // noeud d�connect� � l'avenir
                      inactivelabel[l]=1;
                } else {
                    flagstop=1; // on continue
                    if (visited[i+2]==1) {
                        graph[0][i+2] -= averages[l]; //!!!
                        if(graph[0][i+2] < 0){
                            double tmp = -graph[0][i+2];
                            graph[0][i+2] = graph[i+2][1];
                            graph[i+2][1] = tmp;
                        }
                    }
                    if (visited[i+2]==0) {
                        graph[i+2][1] += averages[l]; //!!!
                        if(graph[i+2][1] < 0){
                            double tmp = -graph[i+2][1];
                            graph[i+2][1] = graph[0][i+2];
                            graph[0][i+2] = tmp;
                        }
                    } // end else
                } // end for
            } //end if
        } while (flagstop);

   free(nextlabel);
   free(inactivelabel);
   free(nums);
   free(averages);
   for(int i = 0; i < n; i++) Y[i] = (double) values[label[i]];
}

void soft_thresh(double *Y, const double thresh, const int n){
    for(int i = 0; i < n; i++){
        double tmp = max(Y[i] - thresh, 0.0);
        Y[i] = tmp + min(Y[i]+thresh, 0.0);
    }
}


int main(int argc, char** argv){
    double lambda1 = 6;
    double lambda2 = 3;
    int *edges1, *edges2;
    bool big_graph = false;
    int m, n;
    double *Y;

    string e_file_name = "./_data/e.txt";
    string n_file_name = "./_data/n.txt";

    ifstream n_infile(n_file_name);
    ifstream e_infile(e_file_name);
    int tm1, tm2, j = 0;
    double tm3;
    e_infile >> tm1 >> tm2;
    m = tm2;
    n = tm1;
    Y = new double[n]; // nodes filled
    cout << m << ' ' << n << endl;
    edges1 = new int[m];
    edges2 = new int[m];
    while(e_infile >> tm1 >> tm2){
      edges1[j] = tm1;
      edges2[j++] = tm2;
    }

    j = 0;
    while(n_infile >> tm3){
      Y[j++] = tm3;
    }

    cout << m << ' ' << n << endl;
    cout << "Done! "<< "# of nodes: " << n << "; # of edges: " << m << endl;
    for(int i = 0; i < 10; i++){
      cout << "data after read file is(Y, edge1, edge2, not graph weight pair) " << Y[i] << ' ' << edges1[i] << ' ' << edges2[i] << endl;
    }

    clock_t t1, t2;
    t1 = clock();
    graph_tv(Y, n, m, edges1, edges2, lambda1, 0.0);
    soft_thresh(Y, lambda2, n);
    t2 = clock();

    cout << "time is " << ((float)t2 - (float)t1) / CLOCKS_PER_SEC << endl;
    for(int i = 0;  i < 10; i++) printf("result is %f. \n", Y[i]);
    ofstream out( "./output_mine.txt" );
    for(int i = 0; i < n; i++) out << Y[i] << endl;

  return 0;
}

