#include <stdio.h>
#include <math.h>

#include "gunrock.h"



int main(int argc, char** argv)
{
    /*
	* CommandLineArgs args(argc, argv);
    * DeviceInit(args);
    * cudaSetDeviceFlags(cudaDeviceMapHost);
    */
	
    int num_elements = 640;
    /*args.GetCmdLineArgument("num", num_elements);*/

    typedef float Value;
    Value *h_origins = (Value*)malloc(sizeof(Value)*num_elements);

    printf("origin data:");
    for (int i = 0; i < num_elements; ++i) 
    {
       h_origins[i] = i;
       printf("%5f, ", h_origins[i]);
    }
    printf("\n");

    gunrock_mad_float(h_origins, num_elements);
   
    return 0;
}
