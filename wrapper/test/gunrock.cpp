/**
 * @file gunrock.cpp
 * @brief mad calculation procedure
 *
 */



#include "gunrock.h"
#include <wrapper/app/mad/mad_enactor.cuh>
#include <wrapper/app/mad/mad_functor.cuh>
#include <wrapper/app/mad/mad_problem.cuh>
#include <stdio.h> // used for print out

using namespace wrapper::app::mad;

GUNROCK_DLL
GunrockResult gunrockCreate(GunrockHandle* theGunrock)
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            fprintf(stderr, "No devices supporting CUDA.\n");
            exit(1);
        }
		
		//TODO: currently hard coded to device 0
        int dev = 0;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (deviceProp.major < 1) {
            fprintf(stderr, "Device does not support CUDA.\n");
            exit(1);
        }
        
        printf("Using device %d: %s\n", dev, deviceProp.name);
        
        cudaSetDevice(dev);
        theGunrock = 0x0;
        return GUNROCK_SUCCESS;
    }


GUNROCK_DLL
GunrockResult gunrockDestroy(GunrockHandle theGunrock)
    {
	    return GUNROCK_SUCCESS;
    }

GUNROCK_DLL
GunrockResult gunrock_mad_int(int *origin_elements, int num_elements)
    {
        typedef MADProblem<int> Problem;
        int    *h_results = (int*)malloc(sizeof(int) * num_elements);
        MADEnactor mad_enactor;
        Problem *simple_problem = new Problem;

        simple_problem->Init(num_elements, origin_elements);
        mad_enactor.Enact<Problem>(simple_problem);
        simple_problem->Extract(h_results);
        printf("Complete.\n");
        
        if (h_results) { free(h_results); }
        if (simple_problem) { delete simple_problem; }
        cudaDeviceSynchronize();
		
		return GUNROCK_SUCCESS;
    }

GUNROCK_DLL
GunrockResult gunrock_mad_float(float *origin_elements, int num_elements)
    {
        typedef MADProblem<float> Problem;
        float   *h_results = (float*)malloc(sizeof(float) * num_elements);
        MADEnactor mad_enactor;
        Problem *simple_problem = new Problem;
        
        simple_problem->Init(num_elements, origin_elements);
        mad_enactor.Enact<Problem>(simple_problem);
        simple_problem->Extract(h_results);
        printf("Complete.\n");

        if (h_results) { free(h_results); }
        if (simple_problem) { delete simple_problem; }
        cudaDeviceSynchronize();
		
		return GUNROCK_SUCCESS;
    }
