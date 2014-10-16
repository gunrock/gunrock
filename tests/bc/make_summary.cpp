#include <fstream>
#include <iostream>
#include <string>

using namespace std;

int main()
{
    string dir_name="eval/ubuntu12.04.k40c/";
    string fil_name=".ubuntu12.04.k40c";
    string flags[]={"",".undir",".idempotence",".undir.idempotence",
                     ".mark_pred",".mark_pred.undir",".mark_pred.idempotence",".mark_pred.undir.idempotence"};
    string data_names[]={"ak2010", "belgium_osm", "coAuthorsDBLP", "delaunay_n13", 
                         "delaunay_n21", "soc-LiveJournal1", "kron_g500-logn21", "webbase-1M"};
    int num_flags = 1;
    int num_datas = 8;
    string file_name="";
    ifstream Fin;
    ofstream Fout;
    string str,pre_str;
    string num_v,num_e,cpu_depth,cpu_time,gpu_depth,gpu_time,validity,validity2,CTA,v_visited,e_visited,rate;
    char ch;

    file_name=dir_name+"bc"+fil_name+".txt";
    Fout.open(file_name.c_str());
    Fout<<"No.\t"<<"Dataset\t"<<"Flag\t"<<"#Nodes\t"<<"#Edges\t";
    Fout<<"CPU Depth\t"<<"CPU Time (ms)\t"<<"GPU Time (ms)\t";
    Fout<<"Validity (BC Value)\t"<<"Validity (Sigma)\t"<<"avg CTA duty";//<<"#Nodes visited\t"<<"#Edges visited\t"<<"rate (MiEdges/s)";
    Fout<<endl;
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
        validity2="";CTA="";
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
                else if (pre_str=="in"       ) { if (cpu_time=="") cpu_time =str; else gpu_time=str;}
                else if (pre_str=="is"       ) { cpu_depth=str;}
                else if (pre_str=="Value" ) { validity =str;}
                else if (pre_str=="Sigma" ) { validity2=str;}
                else if (pre_str=="duty"  ) { CTA      =str;}
                //else if (pre_str=="elapsed"  ) { gpu_time =str;}
                //else if (pre_str=="rate"     ) { rate     =str;}
                //else if (pre_str=="search_depth") { gpu_depth = str;}
                //else if (pre_str=="nodes_visited") { v_visited = str;}
                //else if (pre_str=="visited") {e_visited=str;}
                if (str!="") {pre_str=str;str="";}
            }
        }

        Fout<<num_v<<"\t"<<num_e<<"\t";
        Fout<<cpu_depth<<"\t"<<cpu_time<<"\t";
        Fout<<gpu_time<<"\t";
        Fout<<validity<<"\t"<<validity2<<"\t";
        Fout<<CTA<<endl;
        Fin.close();
    }
    Fout.close();
    return 0;
}

