/**
 * gesummv.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define N_DEFAULT 1024 //4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "gesummv.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0


/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

DATA_TYPE ALPHA = 1;
DATA_TYPE BETA = 1;

cl_int errcode;
Platform* platform;
Context* context;

Kernel* kernel;

Buffer* a_mem_obj;
Buffer* b_mem_obj;
Buffer* x_mem_obj;
Buffer* y_mem_obj;
Buffer* tmp_mem_obj;

std::string kernelName="gesummv_kernel";

size_t N = N_DEFAULT;

void init(DATA_TYPE *A, DATA_TYPE *x) {
  unsigned int i, j;

  for (i = 0; i < N; i++) {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = random<DATA_TYPE>();
    }
  }
}


void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
             DATA_TYPE *tmp, DATA_TYPE *result) {
  unsigned int i, j;

  int intReps = 1;
  
  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (int rep = 0; rep < intReps; ++rep) {
      for (j = 0; j < N; j++) {
        tmp[i] = A[i * N + j] * x[j] + tmp[i];
        y[i] = B[i * N + j] * x[j] + y[i];
      }

      y[i] = ALPHA * tmp[i] + BETA * y[i];
    }
    assert(fabs(y[i] - result[i]) / y[i] < 0.001 && "Error!");
  }

  std::cout << "Ok!\n";
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,DATA_TYPE *tmp,Queue& queue) {

  a_mem_obj = new Buffer(*(platform->getContext()),Buffer::ReadWrite ,sizeof(DATA_TYPE) * N * N, NULL);
  b_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite ,sizeof(DATA_TYPE) * N * N, NULL);
  x_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite ,sizeof(DATA_TYPE) * N, NULL);
  y_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite ,sizeof(DATA_TYPE) * N, NULL);
  tmp_mem_obj = new Buffer(*(platform->getContext()),Buffer::ReadWrite,sizeof(DATA_TYPE) * N, NULL);


  queue.writeBuffer(*a_mem_obj, sizeof(DATA_TYPE) * N * N, A);
  queue.writeBuffer(*b_mem_obj, sizeof(DATA_TYPE) * N * N, B);
  queue.writeBuffer(*x_mem_obj, sizeof(DATA_TYPE) * N, x);
  queue.writeBuffer(*y_mem_obj, sizeof(DATA_TYPE) * N, y);
  queue.writeBuffer(*tmp_mem_obj,sizeof(DATA_TYPE) * N, tmp);
  queue.finish();
}

void cl_launch_kernel(Queue& queue) {
  int n = N;

  size_t oldLocalWorkSize[1], globalWorkSize[1];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[0] = N;

  ///////////////////////////////////////////////
  size_t localWorkSize[1];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "gesummv_kernel", 1);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  kernel->setArgument( 0,*a_mem_obj);
  kernel->setArgument( 1,*b_mem_obj);
  kernel->setArgument( 2,*x_mem_obj);
  kernel->setArgument( 3,*y_mem_obj);
  kernel->setArgument( 4,*tmp_mem_obj);
  kernel->setArgument( 5, sizeof(DATA_TYPE), (void *)&ALPHA);
  kernel->setArgument( 6, sizeof(DATA_TYPE), (void *)&BETA);
  kernel->setArgument( 7, sizeof(int), (void *)&n);


  // Execute the OpenCL kernel
  queue.run(*kernel, 1, 0, globalWorkSize,localWorkSize);
  queue.finish();
}

void cl_clean_up() {
  // Clean up
  delete kernel;
  delete platform;
  delete a_mem_obj;
  delete b_mem_obj;
  delete x_mem_obj;
}

int main(void) {
  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  /////////////////////////
  size_t oldSizes[1] = { N };
  size_t newSizes[1];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "gesummv_kernel", 1);
  N = newSizes[0];
  /////////////////////////

  A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  init(A, x);
  
  platform = new Platform(PLATFORM_ID);
  context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  Queue queue(*context,device,Queue::EnableProfiling); 
  
  cl_mem_init(A, B, x, y, tmp,queue);
  Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
  if(!program.build(device)){
    std::cout << "Error building the program: \n";
    std::cout <<program.getBuildLog(device); 
  }
  kernel=program.createKernel(kernelName.c_str());
  cl_launch_kernel(queue);

  queue.readBuffer(*y_mem_obj,N * sizeof(DATA_TYPE), y_outputFromGpu);
  queue.finish();

  gesummv(A, B, x, y, tmp, y_outputFromGpu);
  cl_clean_up();

  free(A);
  free(B);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}
