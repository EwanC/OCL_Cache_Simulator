/**
 * 3DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define NI 8
#define NJ_DEFAULT 256
#define NK_DEFAULT 256

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "3DConvolution.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

cl_int errcode;
Context* context;
Kernel* kernel;
Platform* platform;

Buffer* a_mem_obj;
Buffer* b_mem_obj;

size_t NJ = NJ_DEFAULT;
size_t NK = NK_DEFAULT;

std::string kernelName = "Convolution3D_kernel";

void init(DATA_TYPE *A) {
  unsigned int i, j, k;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      for (k = 0; k < NK; ++k) {
        A[i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
      }
    }
  }
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *B,Queue& queue) {
  
  a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadOnly,sizeof(DATA_TYPE) * NI * NJ * NK, NULL);
  
  b_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NI * NJ * NK, NULL);


  queue.writeBuffer(*a_mem_obj, sizeof(DATA_TYPE) * NI * NJ * NK, A);
  
  queue.writeBuffer(* b_mem_obj, sizeof(DATA_TYPE) * NI * NJ * NK, B);
  queue.finish();
}


void cl_launch_kernel(Queue& queue) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = NK;
  globalWorkSize[1] = NJ;
 
  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "Convolution3D_kernel", 2);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  kernel->setArgument( 0, *a_mem_obj);
  kernel->setArgument( 1, *b_mem_obj);
  kernel->setArgument( 2, sizeof(int), &ni);
  kernel->setArgument( 3, sizeof(int), &nj);
  kernel->setArgument( 4, sizeof(int), &nk);

  for (int i = 1; i < NI - 1; ++i) // 0
      {
    // set the current value of 'i' for the argument in the kernel
    kernel->setArgument( 5, sizeof(int), &i);

    // Execute the OpenCL kernel
    queue.run(*kernel,2,0,globalWorkSize,localWorkSize);

  }
  queue.finish();
}

void cl_clean_up() {
  // Clean up
  delete kernel;
  delete platform;

  delete a_mem_obj;
  delete b_mem_obj;
}

void compareResults(DATA_TYPE *B, DATA_TYPE *B_outputFromGpu) {
  unsigned int i, j, k, fail;
  fail = 0;

  // Compare result from cpu and gpu...
  for (i = 1; i < NI - 1; ++i) // 0
      {
    for (j = 1; j < NJ - 1; ++j) // 1
        {
      for (k = 1; k < NK - 1; ++k) // 2
          {
        if (percentDiff(B[i * (NK * NJ) + j * NK + k],
                        B_outputFromGpu[i * (NK * NJ) + j * NK + k]) >
            PERCENT_DIFF_ERROR_THRESHOLD) {
          fail++;
        }
      }
    }
  }

  assert(fail == 0 && "Error in the computation");
  std::cout << "Ok!\n";
}

void conv3D(DATA_TYPE *A, DATA_TYPE *B) {
  unsigned int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;
  c21 = +5;
  c31 = -8;
  c12 = -3;
  c22 = +6;
  c32 = -9;
  c13 = +4;
  c23 = +7;
  c33 = +10;

  for (i = 1; i < NI - 1; ++i) // 0
      {
    for (j = 1; j < NJ - 1; ++j) // 1
        {
      for (k = 1; k < NK - 1; ++k) // 2
          {
        B[i * (NK * NJ) + j * NK + k] =
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
            c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
            c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
    }
  }
}

int main(void) {
  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *B_outputFromGpu;

  /////////////////////////
  size_t oldSizes[2] = { NK, NJ };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "Convolution3D_kernel", 2);
  NK = newSizes[0];
  NJ = newSizes[1];
  /////////////////////////

  A = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  B_outputFromGpu = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));

  init(A);
   
  platform = new Platform(PLATFORM_ID);
  context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  Queue queue = Queue(*context,device,Queue::EnableProfiling);

  cl_mem_init(A, B,queue);
  
  Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
  if(!program.build(device)){
    std::cout <<"Error building the program: \n";
    std::cout <<program.getBuildLog(device)<<std::endl;
    return 1;
  }
  kernel = program.createKernel(kernelName.c_str());
  cl_launch_kernel(queue);  



  queue.readBuffer(*b_mem_obj,NI * NJ * NK * sizeof(DATA_TYPE),(void*)B_outputFromGpu);
  queue.finish();

  conv3D(A, B);
  compareResults(B, B_outputFromGpu);
  cl_clean_up();

  free(A);
  free(B);
  free(B_outputFromGpu);

  return 0;
}

