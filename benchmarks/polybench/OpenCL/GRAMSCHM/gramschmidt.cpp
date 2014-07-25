/**
 * gramschmidt.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define M 256 //2048
#define N 64 //2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "gramschmidt.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0

typedef double DATA_TYPE;

cl_int errcode;
Platform* platform;
Context* context;

Kernel* kernel1;
Kernel* kernel2;
Kernel* kernel3;

Buffer* a_mem_obj;
Buffer* r_mem_obj;
Buffer* q_mem_obj;

std::string kernel1Name ="gramschmidt_kernel1";
std::string kernel2Name ="gramschmidt_kernel2";
std::string kernel3Name ="gramschmidt_kernel3";

void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	unsigned int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i*N + j], A_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
			}
		}
	}
	assert(fail == 0 && "CPU - GPU Computation does not match!");
        std::cout << "Ok!\n"; 

}

void init_array(DATA_TYPE* A)
{
	unsigned int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
		}
	}
}

void cl_mem_init(DATA_TYPE* A,Queue& queue)
{
	a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * M * N, NULL);
	r_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * M * N, NULL);
	q_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * M * N, NULL);
	

	queue.writeBuffer(*a_mem_obj, sizeof(DATA_TYPE) * M * N, A);
}


void cl_launch_kernel(Queue& queue)
{
	int m = M;
	int n = N;

	size_t localWorkSize[2], globalWorkSizeKernel1[2], globalWorkSizeKernel2[2], globalWorkSizeKernel3[2];

	localWorkSize[0] = DIM_THREAD_BLOCK_X;
	localWorkSize[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel1[0] = DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel1[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel2[0] = (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)) * DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel2[1] = 1;
	globalWorkSizeKernel3[0] = (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)) * DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel3[1] = 1;

	int k;
	for (k = 0; k < N; k++)
	{
	  // Set the arguments of the kernel
	  kernel1->setArgument( 0,*a_mem_obj);
	  kernel1->setArgument( 1,*r_mem_obj);
	  kernel1->setArgument( 2,*q_mem_obj);
	  kernel1->setArgument( 3, sizeof(int), (void *)&k);
	  kernel1->setArgument( 4, sizeof(int), (void *)&m);
	  kernel1->setArgument( 5, sizeof(int), (void *)&n);
	
      // Execute the OpenCL kernel
      queue.run(*kernel1, 1, 0, globalWorkSizeKernel1, localWorkSize);
          
	  kernel2->setArgument( 0, *a_mem_obj);
	  kernel2->setArgument( 1, *r_mem_obj);
	  kernel2->setArgument( 2, *q_mem_obj);
	  kernel2->setArgument( 3, sizeof(int), (void *)&k);
	  kernel2->setArgument( 4, sizeof(int), (void *)&m);
	  kernel2->setArgument( 5, sizeof(int), (void *)&n);
	
	
	  // Execute the OpenCL kernel
      queue.run(*kernel2, 1, 0,globalWorkSizeKernel2, localWorkSize);

	  kernel3->setArgument( 0,*a_mem_obj);
	  kernel3->setArgument( 1,*r_mem_obj);
	  kernel3->setArgument( 2,*q_mem_obj);
	  kernel3->setArgument( 3, sizeof(int), (void *)&k);
	  kernel3->setArgument( 4, sizeof(int), (void *)&m);
	  kernel3->setArgument( 5, sizeof(int), (void *)&n);
	
	  // Execute the OpenCL kernel
	  queue.run(*kernel3, 1, 0,globalWorkSizeKernel3, localWorkSize);

	}
    queue.finish();
}


void cl_clean_up()
{
	// Clean up
    delete platform;
    delete kernel1;
    delete kernel2;
    delete kernel3;
	delete a_mem_obj;
	delete r_mem_obj;
	delete q_mem_obj;
}


void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
	unsigned int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i*N + k] * A[i*N + k];
		}
		
		R[k*N + k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i*N + k] = A[i*N + k] / R[k*N + k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k*N + j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k*N + j] += Q[i*N + k] * A[i*N + j];
			}
			for (i = 0; i < M; i++)
			{
				A[i*N + j] = A[i*N + j] - Q[i*N + k] * R[k*N + j];
			}
		}
	}
}


int main(void) 
{
	DATA_TYPE* A;
	DATA_TYPE* A_outputFromGpu;
	DATA_TYPE* R;
	DATA_TYPE* Q;
	
	A = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	A_outputFromGpu = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	R = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
	Q = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  

	init_array(A);
        
    platform = new Platform(PLATFORM_ID);
    context = platform->getContext();
    Device device = platform->getDevice(DEVICE_ID);
    Queue queue(*context,device,Queue::EnableProfiling); 
  
	cl_mem_init(A,queue);
    Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
    if(!program.build(device)){
        std::cout << "Error building the program: \n";
        std::cout <<program.getBuildLog(device); 
    }
    kernel1=program.createKernel(kernel1Name.c_str());
    kernel2=program.createKernel(kernel2Name.c_str());
    kernel3=program.createKernel(kernel3Name.c_str());
        
    cl_launch_kernel(queue);

	queue.readBuffer(*a_mem_obj, M*N*sizeof(DATA_TYPE), A_outputFromGpu);
    queue.finish(); 

	gramschmidt(A, R, Q);	
	compareResults(A, A_outputFromGpu);
	cl_clean_up();

	free(A);
	free(A_outputFromGpu);
	free(R);
	free(Q);  

	return 0;
}

