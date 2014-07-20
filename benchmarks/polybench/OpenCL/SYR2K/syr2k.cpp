/**
 * syr2k.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define N_DEFAULT 128//2048
#define M_DEFAULT 128//2048

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "syr2k.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

DATA_TYPE acc;

DATA_TYPE ALPHA = 1;
DATA_TYPE BETA = 1;


cl_int errcode;
Kernel* kernel;
Platform* platform;
Context* context;

Buffer* a_mem_obj;
Buffer* b_mem_obj;
Buffer* c_mem_obj;

size_t N = N_DEFAULT;
size_t M = M_DEFAULT;

std::string kernelName = "syr2k_kernel";


void init_arrays(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	unsigned int i, j;
  
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			C[i*N + j] = random<DATA_TYPE>();
		}
      	
		for (j = 0; j < M; j++)
		{
			A[i*N + j] = random<DATA_TYPE>();
			B[i*N + j] = random<DATA_TYPE>();
		}
	}
}

void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C,Queue& queue)
{
	a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, N*M*sizeof(DATA_TYPE), NULL);
	b_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, N*M*sizeof(DATA_TYPE), NULL);
	c_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, N*M*sizeof(DATA_TYPE), NULL);
	

	queue.writeBuffer(*a_mem_obj, N*M*sizeof(DATA_TYPE), A);
	queue.writeBuffer(*b_mem_obj, N*M*sizeof(DATA_TYPE), B);
	queue.writeBuffer(*c_mem_obj, N*M*sizeof(DATA_TYPE), C);
        queue.finish();
 
}

void cl_launch_kernel(Queue& queue)
{
	int m = M;
	int n = N;

	DATA_TYPE alpha = ALPHA;
	DATA_TYPE beta = BETA;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = N;
  globalWorkSize[1] = M;

  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "syr2k_kernel", 2);
  ///////////////////////////////////////////////

	// Set the arguments of the kernel
	kernel->setArgument( 0,*a_mem_obj);
	kernel->setArgument( 1,*b_mem_obj);
	kernel->setArgument( 2,*c_mem_obj);
	kernel->setArgument( 3, sizeof(DATA_TYPE), (void *)&ALPHA);
	kernel->setArgument( 4, sizeof(DATA_TYPE), (void *)&BETA);
	kernel->setArgument( 5, sizeof(int), (void *)&m);
	kernel->setArgument( 6, sizeof(int), (void *)&n);
	

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
	delete b_mem_obj;
	delete c_mem_obj;
}

void syr2k(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *result) {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] *= BETA;
    }
  }

  int intReps = 2;
  

  for (i = 0; i < 128; i++) {
    for (j = 0; j < 128; j++) {
      for (int rep = 0; rep < intReps; ++rep) {
        for (k = 0; k < M; k++) {
          C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
          C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
        }
      }

      assert(fabs(C[i * N + j] - result[i * N + j]) / result[i * N + j] < 0.001 &&
             "Error!");
    }
  }

  std::cout << "Ok!\n"; 

}

int main(void) 
{
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* C_outputFromGpu;

  /////////////////////////
  size_t oldSizes[2] = { N, M };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "syr2k_kernel", 2);
  N = newSizes[0];
  M = newSizes[1];
  /////////////////////////

	A = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C_outputFromGpu = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));

	init_arrays(A, B, C);
        
        platform = new Platform(PLATFORM_ID);
        context = platform->getContext();
        Device device = platform->getDevice(DEVICE_ID);
        Queue queue(*context,device,Queue::EnableProfiling); 
  
	cl_mem_init(A, B, C,queue);
     
        Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
        if(!program.build(device)){
           std::cout << "Error building the program: \n";
           std::cout <<program.getBuildLog(device); 
        }
        kernel=program.createKernel(kernelName.c_str());
        cl_launch_kernel(queue);


	 queue.readBuffer(*c_mem_obj, N*M*sizeof(DATA_TYPE), C_outputFromGpu);
         queue.finish();
  
	syr2k(A, B, C, C_outputFromGpu);
	cl_clean_up();

	free(A);
	free(B);
	free(C);
	free(C_outputFromGpu);

	return 0;
}

