// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_multi_scan.cu
 *
 * @brief Simple test driver program for multi-scan.
 */

#include <stdio.h> 
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <time.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// Multi scan include
#include <gunrock/util/scan/multi_scan.cuh>
#include <gunrock/util/multithread_utils.cuh>

using namespace gunrock;
using namespace gunrock::util;

typedef int VertexId;
typedef int SizeT;

void Usage()
{
}

void Make_Test(
    const SizeT           n,  
    const SizeT           m,
          SizeT*          &Offset,
          int*            &Partition,
          VertexId*       &Convertion,
          SizeT           &Key_Length,
          VertexId*       &Keys)
{
    srand (time(NULL));
    Offset = new SizeT[n+1];
    int *marker = new int[m];
    Offset[0]=0;
    for (SizeT node=1;node<=n;node++) 
        Offset[node]=(rand()%m)+Offset[node-1];
    Partition = new int     [Offset[n]];
    Convertion= new VertexId[Offset[n]];
    for (SizeT node=0;node<n;node++)
    {   
        memset(marker,0,sizeof(int)*m);
        for (SizeT i=Offset[node];i<Offset[node+1];i++)
        {
            int x=rand() %m;
            while (marker[x]!=0) x=rand() %m;
            marker[x]=1;
            Partition [i] = x;
            Convertion[i] = rand() %n;
        }
    }
    delete[] marker;marker=NULL;
    Key_Length=(n/2) + (rand() %(n/2));
    Keys=new VertexId[Key_Length];
    for (SizeT i=0;i<Key_Length;i++)
        Keys[i]=rand() %n;   
}

void Scan_Cpu(
    const SizeT           n,  
    const SizeT           m,
    const SizeT*          Offset,
          int*            Partition,
          VertexId*       Convertion,
          SizeT           Key_Length,
          VertexId*       Keys,
          SizeT*          &Result_Length,
          VertexId*       &Results)
{
    Result_Length = new SizeT[m];
    SizeT* Result_Offset = new SizeT[m+1];
    memset(Result_Length,0,sizeof(SizeT)*m);
    for (SizeT i=0;i<Key_Length;i++)
    {
        VertexId node=Keys[i];
        for (SizeT j=Offset[node];j<Offset[node+1];j++)
            Result_Length[Partition[j]]++;
    }
    Result_Offset[0]=0;
    for (SizeT i=0;i<m;i++) Result_Offset[i+1]=Result_Offset[i]+Result_Length[i];
    Results=new VertexId[Result_Offset[m]];

    memset(Result_Length,0,sizeof(SizeT)*m);
    for (SizeT i=0;i<Key_Length;i++)
    {
        VertexId node=Keys[i];
        for (SizeT j=Offset[node];j<Offset[node+1];j++)
        {
            int x=Partition[j];
            Results[Result_Offset[x]+Result_Length[x]]=Convertion[j];
            Result_Length[x]++;
        }
    }
    delete[] Result_Offset;
}

/*long long Compare_Result(
    const SizeT           n,
    const SizeT*    const Result,
    const SizeT*    const CPU_Result)
{
    long long Def_Sum=0;
    for (SizeT i=0;i<n;i++) Def_Sum+=( Result[i]>CPU_Result[i]? Result[i]-CPU_Result[i] : CPU_Result[i]-Result[i]);
    return Def_Sum;
}*/

void Run_CPU_Test(
    const SizeT           n,  
    const SizeT           m,
    const SizeT*          Offset,
          int*            Partition,
          VertexId*       Convertion,
          SizeT           Key_Length,
          VertexId*       Keys,
          SizeT*          &Result_Length,
          VertexId*       &Results)
{
    CpuTimer Cpu_Timer;
    Cpu_Timer.Start();
    Scan_Cpu(n,m,Offset,Partition,Convertion,Key_Length,Keys,Result_Length,Results);
    Cpu_Timer.Stop();
    float Cpu_Time = Cpu_Timer.ElapsedMillis();
    printf("CPU multi-scan   finished in %lf msec.\n",Cpu_Time);
}

void Run_GPU_Test(
    const SizeT           n,  
    const SizeT           m,
    const SizeT*          Offset,
          int*            Partition,
          VertexId*       Convertion,
          SizeT           Key_Length,
          VertexId*       Keys,
          SizeT*          &Result_Length,
          VertexId*       &Results)
{
    SizeT*    d_Offset;
    int*      d_Partition;
    VertexId* d_Convertion;
    VertexId* d_Keys;
    SizeT*    d_Result_Length;
    VertexId* d_Results;
    GpuTimer  Gpu_Timer;
    SizeT     Sum_Result_Length;
    //float     Gpu_Time;
    scan::MultiScan<VertexId,SizeT,true,256,8> Scaner;

    util::GRError(cudaMalloc(&d_Offset    ,sizeof(SizeT   ) *(n+1)     ),
          "cudaMalloc d_Offset failed"       , __FILE__, __LINE__);
    util::GRError(cudaMalloc(&d_Partition ,sizeof(int     ) *Offset[n] ),
          "cudaMalloc d_Partition failed"    , __FILE__, __LINE__);
    util::GRError(cudaMalloc(&d_Convertion,sizeof(VertexId) *Offset[n] ),
          "cudaMalloc d_Convertion failed"   , __FILE__, __LINE__);
    util::GRError(cudaMalloc(&d_Keys      ,sizeof(VertexId) *Key_Length),
          "cudaMalloc d_Keys failed"         , __FILE__, __LINE__);
    util::GRError(cudaMalloc(&d_Result_Length,sizeof(SizeT) *(m+1)     ),
          "cudaMalloc d_Result_Length failed", __FILE__, __LINE__);
    util::GRError(cudaMalloc(&d_Results   ,sizeof(VertexId) *Key_Length*m),
          "cudaMalloc d_Results failed"      , __FILE__, __LINE__);

    Gpu_Timer.Start();
    util::GRError(cudaMemcpy(d_Offset    ,Offset    ,sizeof(SizeT   ) *(n+1)     ,cudaMemcpyHostToDevice),
          "cudaMemcpy d_Offset failed"    , __FILE__, __LINE__);
    util::GRError(cudaMemcpy(d_Partition ,Partition ,sizeof(int     ) *Offset[n] ,cudaMemcpyHostToDevice),
          "cudaMemcpy d_Partition failed" , __FILE__, __LINE__);
    util::GRError(cudaMemcpy(d_Convertion,Convertion,sizeof(VertexId) *Offset[n] ,cudaMemcpyHostToDevice),
          "cudaMemcpy d_Convertion failed", __FILE__, __LINE__);
    util::GRError(cudaMemcpy(d_Keys      ,Keys      ,sizeof(VertexId) *Key_Length,cudaMemcpyHostToDevice),
          "cudaMemcpy d_Keys failed"      , __FILE__, __LINE__);
    Gpu_Timer.Stop();
    //Gpu_Time = Gpu_Timer.ElapsedMillis();
    printf("GPU input  trans finished in %lf msecs.\n",Gpu_Timer.ElapsedMillis());

    Gpu_Timer.Start();
    Scaner.Scan_with_dKeys_Backward<0,0>(
        Key_Length,
        m,
        d_Keys,
        d_Offset,
        d_Results,
        d_Partition,
        d_Convertion,
        d_Result_Length,
        (VertexId**)NULL,
        (VertexId**)NULL,
        (VertexId**)NULL,
        (VertexId**)NULL);//), "GPU MultiScan failed", __FILE__, __LINE__);
    Gpu_Timer.Stop();
    printf("GPU multi_scan   finished in %lf msecs.\n",Gpu_Timer.ElapsedMillis());

    Gpu_Timer.Start();
    Result_Length=new SizeT[m+1];
    util::GRError(cudaMemcpy(Result_Length,d_Result_Length,sizeof(SizeT) *m     ,cudaMemcpyDeviceToHost),
          "cudaMemcpy d_Result_Length failed", __FILE__, __LINE__);
    Sum_Result_Length=0;
    for (int i=0;i<m;i++) Sum_Result_Length+=Result_Length[i];
    //Sum_Result_Length=1;
    Results = new VertexId[Sum_Result_Length];
    util::GRError(cudaMemcpy(Results,d_Results,sizeof(VertexId) *Sum_Result_Length,cudaMemcpyDeviceToHost),
          "cudaMemcpy d_Results failed", __FILE__, __LINE__);
    Gpu_Timer.Stop();
    printf("GPU output trans finished in %lf msecs.\n",Gpu_Timer.ElapsedMillis());fflush(stdout);
    util::GRError(cudaFree(d_Offset    ), "cudaFree d_Offset failed"    , __FILE__, __LINE__);
    util::GRError(cudaFree(d_Partition ), "cudaFree d_Partition failed" , __FILE__, __LINE__);
    util::GRError(cudaFree(d_Convertion), "cudaFree d_Convertion failed", __FILE__, __LINE__);
    util::GRError(cudaFree(d_Keys      ), "cudaFree d_Keys failed"      , __FILE__, __LINE__);
    util::GRError(cudaFree(d_Results   ), "cudaFree d_Results failed"   , __FILE__, __LINE__);
    util::GRError(cudaFree(d_Result_Length), "cudaFree d_Result_Length failed", __FILE__, __LINE__);
}

int main( int argc, char** argv)
{
    SizeT N,M;
    CommandLineArgs args(argc, argv);

    if ((argc < 2) || (args.CheckCmdLineFlag("help")))
    {
        Usage();
        return 1;
    }

    DeviceInit(args);
    args.GetCmdLineArgument<SizeT>("Num_Elements",N);
    args.GetCmdLineArgument<SizeT>("Num_Rows",M);

    //VertexId* Select     = new VertexId [N];
    //int*      Splict     = new int      [N];
    //SizeT*    Offset     = new SizeT    [M+1];
    //SizeT*    GPU_Result = new SizeT    [N];
    //SizeT*    CPU_Result = new SizeT    [N];
    //SizeT*    GPU_Length = new SizeT    [M];

    SizeT    *Offset,*CPU_Length,*GPU_Length,Key_Length;
    int      *Partition;
    VertexId *Convertion,*Keys,*CPU_Result,*GPU_Result;

    Make_Test   (N,M,Offset,Partition,Convertion,Key_Length,Keys);
    /*for (VertexId node=0;node<N;node++)
    {
        printf("%d = ",node);
        for (SizeT i=Offset[node];i<Offset[node+1];i++)
            printf("%d,%d | ",Partition[i],Convertion[i]);
        printf("\t");
    }*/
    //util::cpu_mt::PrintCPUArray<SizeT, SizeT   >("Offset"    ,Offset    ,N+1       );
    //util::cpu_mt::PrintCPUArray<SizeT, int     >("Partition" ,Partition ,Offset[N] );
    //util::cpu_mt::PrintCPUArray<SizeT, VertexId>("Convertion",Convertion,Offset[N] );
    printf("Key_Length = %d\n",Key_Length);
    //util::cpu_mt::PrintCPUArray<SizeT, VertexId>("Keys"      ,Keys      ,Key_Length);
    Run_CPU_Test(N,M,Offset,Partition,Convertion,Key_Length,Keys,CPU_Length,CPU_Result);
    util::cpu_mt::PrintCPUArray<SizeT, SizeT   >("CPU_Length",CPU_Length,M         );
    /*SizeT Current_Offset=0;
    for (int j=0;j<M;j++)
    {
        printf("%d: ",j);
        for (int i=0;i<CPU_Length[j];i++)
            printf("%d, ",CPU_Result[i+Current_Offset]);
        Current_Offset+=CPU_Length[j];
        printf("\n");
    }*/
    Run_GPU_Test(N,M,Offset,Partition,Convertion,Key_Length,Keys,GPU_Length,GPU_Result);
    /*Current_Offset=0;
    for (int j=0;j<M;j++)
    {
        printf("%d: ",j);
        for (int i=0;i<GPU_Length[j];i++)
            printf("%d, ",GPU_Result[i+Current_Offset]);
        Current_Offset+=GPU_Length[j];
        printf("\n");
    }*/
    SizeT Sum_Length=0;
    for (int i=0;i<M;i++) Sum_Length+=CPU_Length[i];
    util::CompareResults<SizeT   ,SizeT   >(GPU_Length,CPU_Length,M,true);
    util::CompareResults<VertexId,VertexId>(GPU_Result,CPU_Result,Sum_Length,true);
    printf(" %d \n", Sum_Length);
    return 0;
}

