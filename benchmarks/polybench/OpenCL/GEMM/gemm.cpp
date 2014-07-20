/**
 * gemm.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define NI_DEFAULT 128 //512
#define NJ_DEFAULT 128 //512
#define NK_DEFAULT 128 //512

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412
#define BETA 2123

#define KERNEL_FILE_NAME "gemm.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0


/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


cl_int errcode;
Platform* platform;
Context* context;
Kernel* kernel;

Buffer* a_mem_obj;
Buffer* b_mem_obj;
Buffer* c_mem_obj;

size_t NI = NI_DEFAULT; 
size_t NJ = NJ_DEFAULT; 
size_t NK = NK_DEFAULT; 

std::string kernelName = "gemm";

void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	unsigned int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
      		A[i*NK + j] = random<DATA_TYPE>();
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		B[i*NJ + j] = random<DATA_TYPE>();
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		C[i*NJ + j] = random<DATA_TYPE>();
		}
	}
}

void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C,Queue& queue)
{
	a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NI * NK, NULL);
	b_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NK * NJ, NULL);
	c_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NI * NJ, NULL);
		

	queue.writeBuffer(*a_mem_obj, sizeof(DATA_TYPE) * NI * NK, A);
	queue.writeBuffer(*b_mem_obj, sizeof(DATA_TYPE) * NK * NJ, B);
	queue.writeBuffer(*c_mem_obj, sizeof(DATA_TYPE) * NI * NJ, C);
        queue.finish();
}

void cl_launch_kernel(Queue& queue)
{
  
  int ni=NI;
  int nj=NJ;
  int nk=NK;

  DATA_TYPE alpha = ALPHA;
  DATA_TYPE beta = BETA;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = NJ;
  globalWorkSize[1] = NI;

  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "gemm", 2);
  ///////////////////////////////////////////////


	// Set the arguments of the kernel
  kernel->setArgument( 0,*a_mem_obj);
  kernel->setArgument( 1,*b_mem_obj);
  kernel->setArgument( 2,*c_mem_obj);
  kernel->setArgument( 3, sizeof(DATA_TYPE), (void *)&alpha);
  kernel->setArgument( 4, sizeof(DATA_TYPE), (void *)&beta);
  kernel->setArgument( 5, sizeof(int), (void *)&ni);
  kernel->setArgument( 6, sizeof(int), (void *)&nj);
  kernel->setArgument( 7, sizeof(int), (void *)&nk);
	
  // Execute the OpenCL kernel
  queue.run(*kernel, 2, NULL, globalWorkSize, localWorkSize);
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

void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE* result) {
  unsigned int i, j, k;

  int intReps = 2;

  for (i = 0; i < 128; i++) {
    for (j = 0; j < 128; j++) {
      for (int rep = 0; rep < intReps; ++rep) {
        C[i * NJ + j] *= BETA;

        for (k = 0; k < NK; ++k) {
          C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
        }
      }
      if(fabs(C[i * NJ+ j] - result[i * NJ + j]) > 1 )
        std::cout<<i<<" "<<j <<"diff: "<<fabs(C[i * NJ + j] - result[i * NJ + j]) << " "<<  C[i * NJ + j] << " " << result[i * NJ + j] << "\n";
      assert(fabs(C[i * NJ + j] - result[i * NJ + j]) < 1 && "Error!");
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
  size_t oldSizes[2] = { NJ, NI };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "gemm", 2);
  NJ = newSizes[0];
  NI = newSizes[1];
  NK = NJ;  
  /////////////////////////

  A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
  B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
  C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
  C_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

  init(A, B, C); 
 
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

  queue.readBuffer(*c_mem_obj, NI*NJ*sizeof(DATA_TYPE), C_outputFromGpu);
  queue.finish();
 
  gemm(A, B, C, C_outputFromGpu);
  cl_clean_up();

  free(A);
  free(B);  
  free(C);  
  free(C_outputFromGpu); 

  return 0;
}

