/**
 * fdtd2d.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "Buffer.h"
#include "Event.h"
#include "Kernel.h"
#include "Program.h"
#include "Queue.h"
#include "Device.h"
#include "Platform.h"

#include "bench_support.h"
#include "MathUtils.h"
#include "SystemConfig.h"
#include "SystemConfiguration.h"
#include "Utils.h"

//define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

/* Problem size */
#define TMAX 3//500
#define NX 256 //2048
#define NY 256 //2048

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#define MAX_SOURCE_SIZE (0x100000)

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "fdtd2d.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

DATA_TYPE alpha = 23;
DATA_TYPE beta = 15;

cl_int errcode;
Context* context;
Platform* platform;

Kernel* kernel1;
Kernel* kernel2;
Kernel* kernel3;

Buffer* fict_mem_obj;
Buffer* ex_mem_obj;
Buffer* ey_mem_obj;
Buffer* hz_mem_obj;

std::string kernel1Name = "fdtd_kernel1";
std::string kernel2Name = "fdtd_kernel2";
std::string kernel3Name = "fdtd_kernel3";

void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	unsigned int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > ERROR_THRESHOLD) 
			{
				fail++;
			}

		}
	}
	
        assert(fail == 0 && "CPU - GPU Computation does not match!");
        std::cout << "Ok!\n";  
}

void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	unsigned int i, j;

  	for (i = 0; i < TMAX; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}

void cl_mem_init(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz,Queue& queue)
{
	fict_mem_obj =  new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * TMAX, NULL);
	ex_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NX * (NY + 1), NULL);
	ey_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * (NX + 1) * NY, NULL);
	hz_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NX * NY, NULL);
	

	queue.writeBuffer(*fict_mem_obj, sizeof(DATA_TYPE) * TMAX, _fict_);
	queue.writeBuffer(*ex_mem_obj, sizeof(DATA_TYPE) * NX * (NY + 1), ex);
	queue.writeBuffer(*ey_mem_obj, sizeof(DATA_TYPE) * (NX + 1) * NY, ey);
	queue.writeBuffer(*hz_mem_obj, sizeof(DATA_TYPE) * NX * NY, hz);
        queue.finish();
}

void cl_launch_kernel(Queue& queue)
{
	int nx = NX;
	int ny = NY;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NY) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NX) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	int t;
	for(t=0;t<TMAX;t++)
	{
		// Set the arguments of the kernel
		kernel1->setArgument( 0,*fict_mem_obj);
		kernel1->setArgument( 1,*ex_mem_obj);
		kernel1->setArgument( 2,*ey_mem_obj);
		kernel1->setArgument( 3,*hz_mem_obj);
		kernel1->setArgument( 4, sizeof(int), (void *)&t);
		kernel1->setArgument( 5, sizeof(int), (void *)&nx);
		kernel1->setArgument( 6, sizeof(int), (void *)&ny);
		
		// Execute the OpenCL kernel
		queue.run(*kernel1, 2, 0, globalWorkSize, localWorkSize);
		
		// Set the arguments of the kernel
		kernel2->setArgument( 0,*ex_mem_obj);
		kernel2->setArgument( 1,*ey_mem_obj);
		kernel2->setArgument( 2,*hz_mem_obj);
		kernel2->setArgument( 3, sizeof(int), (void *)&nx);
		kernel2->setArgument( 4, sizeof(int), (void *)&ny);
		
		// Execute the OpenCL kernel
		queue.run(*kernel2, 2,0, globalWorkSize, localWorkSize);

		// Set the arguments of the kernel
		kernel3->setArgument( 0,*ex_mem_obj);
		kernel3->setArgument( 1,*ey_mem_obj);
		kernel3->setArgument( 2,*hz_mem_obj);
		kernel3->setArgument( 3, sizeof(int), (void *)&nx);
		kernel3->setArgument( 4, sizeof(int), (void *)&ny);
		
		// Execute the OpenCL kernel
		queue.run(*kernel3, 2, 0, globalWorkSize, localWorkSize);
		queue.finish();
	}
}


void cl_clean_up()
{
	// Clean up
        delete platform;	

        delete kernel1;
	delete kernel2;
	delete kernel3;
	delete fict_mem_obj;
	delete ex_mem_obj;
	delete ey_mem_obj;
	delete hz_mem_obj;
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	unsigned int t, i, j;
	
	for(t=0; t < TMAX; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       			for (j = 0; j < NY; j++)
				{
       				ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}
	}
}


int main(void) 
{
	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	DATA_TYPE* hz_outputFromGpu;

	_fict_ = (DATA_TYPE*)malloc(TMAX*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	
	init_arrays(_fict_, ex, ey, hz);
        
        platform = new Platform(PLATFORM_ID);
        context = platform->getContext();
        Device device = platform->getDevice(DEVICE_ID);
        Queue queue(*context,device,Queue::EnableProfiling); 
 
        cl_mem_init(_fict_, ex, ey, hz,queue);

        Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
        if(!program.build(device)){
           std::cout << "Error building the program: \n";
           std::cout <<program.getBuildLog(device); 
         }
         kernel1=program.createKernel(kernel1Name.c_str());
         kernel2=program.createKernel(kernel2Name.c_str());
         kernel3=program.createKernel(kernel3Name.c_str());

         cl_launch_kernel(queue);

	queue.readBuffer(*hz_mem_obj, NX * NY * sizeof(DATA_TYPE), hz_outputFromGpu);
        queue.finish();

	runFdtd(_fict_, ex, ey, hz);
	compareResults(hz, hz_outputFromGpu);
	cl_clean_up();
	
	free(_fict_);
	free(ex);
	free(ey);
	free(hz);
	free(hz_outputFromGpu);
	
    	return 0;
}

