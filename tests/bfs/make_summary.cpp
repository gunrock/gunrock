#include <fstream>
#include <iostream>
#include <string>

using namespace std;

int main()
{
    string dir_name="eval/CentOS6.4_k40cx1_dsize/";
    string fil_name=".CentOS6.4_k40cx1_dsize";
    string flags[]={"",".undir",".idempotence",".undir.idempotence",
                     ".mark_pred",".mark_pred.undir",".mark_pred.idempotence",".mark_pred.undir.idempotence"};
    string data_names[]={"ak2010",  
                     "delaunay_n10", "delaunay_n11", "delaunay_n12", "delaunay_n13", "delaunay_n14", 
                     "delaunay_n15", "delaunay_n16", "delaunay_n17", "delaunay_n18", "delaunay_n19", 
                     "delaunay_n20", "delaunay_n21", "delaunay_n22", "delaunay_n23", "delaunay_n24",
                     "kron_g500-logn16", "kron_g500-logn17", "kron_g500-logn18", "kron_g500-logn19", "kron_g500-logn20", "kron_g500-logn21",  
                     "coAuthorsDBLP","coAuthorsCiteseer","coPapersDBLP","coPapersCiteseer","citationCiteseer",
                     "preferentialAttachment","soc-LiveJournal1", "roadNet-CA", "belgium_osm","netherlands_osm",
                     "italy_osm","luxembourg_osm", "great-britain_osm","germany_osm","asia_osm","europe_osm",
                     "road_usa","road_central", "webbase-1M","tweets","bitcoin","caidaRouterLevel"};
    int num_flags = 6;
    int num_datas = 40;
    string file_name="";
    ifstream Fin;
    ofstream Fout;
    string str,pre_str;
    string num_v,num_e,cpu_depth,cpu_time,gpu_depth,gpu_time,validity,v_visited,e_visited,rate;
    char ch;

    file_name=dir_name+"bfs"+fil_name+".txt";
    Fout.open(file_name.c_str());
    Fout<<"No.\t"<<"Dataset\t"<<"Flag\t"<<"#Nodes\t"<<"#Edges\t";
    Fout<<"CPU Depth\t"<<"CPU Time (ms)\t"<<"GPU Depth\t"<<"GPU Time (ms)\t";
    Fout<<"Validity\t"<<"#Nodes visited\t"<<"#Edges visited\t"<<"rate (MiEdges/s)"<<endl;
    for (int i=0;i<num_datas;i++)
    for (int j=0;j<num_flags;j++)
    {
        file_name=dir_name+data_names[i]+fil_name+flags[j]+".txt";
        Fin.open(file_name.c_str());

        Fout<<i*num_flags+j<<"\t";
        Fout<<data_names[i]<<"\t";
        Fout<<flags[j]<<"\t";

        str="";pre_str="";
        num_v="";num_e="";
        cpu_depth="";cpu_time="";
        gpu_depth="";gpu_time="";
        validity="";rate="";
        v_visited="";e_visited="";
        while (!Fin.eof())
        {
            ch=Fin.get();
            if ((ch>='0' && ch<='9')||
                (ch>='a' && ch<='z')||
                (ch>='A' && ch<='Z')||
                ch=='.' || ch=='(' || ch==')' || ch=='_' )
            {
                str=str+ch;
            } else {
                if      (pre_str=="Histogram") { num_v    =str;num_v[0]=' ';}
                else if (pre_str=="vertices" ) { num_e    =str;}
                else if (pre_str=="in"       ) { cpu_time =str;}
                else if (pre_str=="is"       ) { cpu_depth=str;}
                else if (pre_str=="Validity" ) { validity =str;}
                else if (pre_str=="elapsed"  ) { gpu_time =str;}
                else if (pre_str=="rate"     ) { rate     =str;}
                else if (pre_str=="search_depth") { gpu_depth = str;}
                else if (pre_str=="nodes_visited") { v_visited = str;}
                else if (pre_str=="visited") {e_visited=str;}
                if (str!="") {pre_str=str;str="";}
            }
        }

        Fout<<num_v<<"\t"<<num_e<<"\t";
        Fout<<cpu_depth<<"\t"<<cpu_time<<"\t";
        Fout<<gpu_depth<<"\t"<<gpu_time<<"\t";
        Fout<<validity<<"\t"<<v_visited<<"\t";
        Fout<<e_visited<<"\t"<<rate<<endl;
        Fin.close();
    }
    Fout.close();
    return 0;
}

