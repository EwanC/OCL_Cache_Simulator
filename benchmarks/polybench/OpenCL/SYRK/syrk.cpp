/**
 * syrk.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define N_DEFAULT 128//1024
#define M_DEFAULT 128//1024

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "syrk.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0


/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


DATA_TYPE acc;

DATA_TYPE alpha = 123;
DATA_TYPE beta = 14512;

cl_int errcode;
Kernel* kernel;
Platform* platform;
Context* context;

Buffer* a_mem_obj;
Buffer* c_mem_obj;

unsigned int N = N_DEFAULT;
unsigned int M = M_DEFAULT;

std::string kernelName = "syrk_kernel";

void init_arrays(DATA_TYPE* A, DATA_TYPE* C)
{
	unsigned  int i, j;
	
	for (i = 0; i < N; i++)
    	{
		for (j = 0; j < M; j++)
		{
			A[i*M + j] = random<DATA_TYPE>();
		}
		
		for (j = 0; j < N; j++)
		{
			C[i*M + j] = random<DATA_TYPE>();
		}
	}
}

void cl_mem_init(DATA_TYPE* A, DATA_TYPE* C,Queue& queue)
{
	a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * N * M, NULL);
	c_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * N * M, NULL);
	
      queue.writeBuffer(*a_mem_obj, sizeof(DATA_TYPE) * N * M, A);
      queue.writeBuffer(*c_mem_obj, sizeof(DATA_TYPE) * N * M, C);
      queue.finish();
}

void cl_launch_kernel(Queue& queue)
{
	int m = M;
	int n = N;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = N;
  globalWorkSize[1] = M;

  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "syrk_kernel", 2);
  ///////////////////////////////////////////////


	// Set the arguments of the kernel
	kernel->setArgument( 0,*a_mem_obj);
	kernel->setArgument( 1,*c_mem_obj);
	kernel->setArgument( 2, sizeof(DATA_TYPE), (void *)&alpha);
	kernel->setArgument( 3, sizeof(DATA_TYPE), (void *)&beta);
	kernel->setArgument( 4, sizeof(int), (void *)&m);
	kernel->setArgument( 5, sizeof(int), (void *)&n);


	// Execute the OpenCL kernel
	queue.run(*kernel, 2,0, globalWorkSize, localWorkSize);
        queue.finish();
}


void cl_clean_up()
{
	// Clean up
	delete kernel;
	delete platform;
        delete a_mem_obj;
	delete c_mem_obj;
}

void syrk(DATA_TYPE *A, DATA_TYPE *C, DATA_TYPE *result) {
  unsigned int i, j, k;

  int intReps = 1;

  for (i = 0; i < 128; i++) {
    for (j = 0; j < 128; j++) {
      for (int rep = 0; rep < intReps; ++rep) {
        C[i * M + j] *= beta;
        for (k = 0; k < M; k++) {
          C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
        }
      }


      assert(fabs(C[i * N + j] - result[i * N + j])  <
                 0.001 &&
             "Error!");
    }
  }

  std::cout << "Ok!\n";
}

int main(void) 
{
	DATA_TYPE* A;
	DATA_TYPE* C;
	DATA_TYPE* C_outputFromGpu;

  /////////////////////////
  size_t oldSizes[2] = { N, M };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "syrk_kernel", 2);
  N = newSizes[0];
  M = newSizes[1];
  /////////////////////////

	A = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C_outputFromGpu = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));

	init_arrays(A, C);
 
        platform = new Platform(PLATFORM_ID);
        context = platform->getContext();
        Device device = platform->getDevice(DEVICE_ID);
        Queue queue(*context,device,Queue::EnableProfiling); 
  
	cl_mem_init(A, C,queue);

	Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
        if(!program.build(device)){
           std::cout << "Error building the program: \n";
           std::cout <<program.getBuildLog(device); 
        }
        kernel=program.createKernel(kernelName.c_str());
        cl_launch_kernel(queue);


	queue.readBuffer(*c_mem_obj, M * N * sizeof(DATA_TYPE), C_outputFromGpu);

	syrk(A, C, C_outputFromGpu);
	cl_clean_up();
	
	free(A);
	free(C);
	free(C_outputFromGpu);

	return 0;
}

