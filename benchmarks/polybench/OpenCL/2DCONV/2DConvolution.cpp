/**
 * 2DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <math.h>

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
#define NI_DEFAULT 1024 //4096
#define NJ_DEFAULT 1024 //4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "2DConvolution.cl"
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
Buffer* c_mem_obj;

unsigned int NI = NI_DEFAULT;
unsigned int NJ = NJ_DEFAULT;

std::string kernelName="Convolution2D_kernel";


void init(DATA_TYPE *A) {
  unsigned int i, j;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      A[i * NJ + j] = random<float>();
    }
  }
}

void cl_mem_init(DATA_TYPE *A,Queue& queue) {
  a_mem_obj = new Buffer (*(platform->getContext()), Buffer::ReadOnly,sizeof(DATA_TYPE) * NI * NJ,NULL);
 
  b_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NI * NJ,NULL);

  queue.writeBuffer(*a_mem_obj,sizeof(DATA_TYPE) * NI * NJ, A);
  queue.finish();

}


void cl_launch_kernel(Queue& queue) {
  int ni = NI;
  int nj = NJ;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = NI;
  globalWorkSize[1] = NJ;

  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "Convolution2D_kernel", 2);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  kernel->setArgument(0,*a_mem_obj);
  kernel->setArgument(1,*b_mem_obj);
  kernel->setArgument(2,sizeof(int),&ni);
  kernel->setArgument(3,sizeof(int),&nj);


  // Execute the OpenCL kernel
  queue.run(*kernel,2,0,globalWorkSize,localWorkSize);
  queue.finish();
}

void cl_clean_up() {
  // Clean up
  delete kernel;
  delete platform;

  delete a_mem_obj;
  delete b_mem_obj;
  delete c_mem_obj;

}

void conv2D(DATA_TYPE *A, DATA_TYPE *B_outputFromGpu) {
  unsigned int i, j;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;
  c21 = +0.5;
  c31 = -0.8;
  c12 = -0.3;
  c22 = +0.6;
  c32 = -0.9;
  c13 = +0.4;
  c23 = +0.7;
  c33 = +0.10;

  for (i = 1; i < NI - 1; ++i) {
    for (j = 1; j < NJ - 1; ++j) {
      DATA_TYPE tmp =
          c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] +
          c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] +
          c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] +
          c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] +
          c33 * A[(i + 1) * NJ + (j + 1)];
      assert(fabs(tmp - B_outputFromGpu[i * NJ + j]) < 0.01f && "Error in the computation");
    }
  }
  std::cout << "Ok!\n";
}

int main(int argc, char *argv[]) {

  DATA_TYPE *A;
  DATA_TYPE *B_outputFromGpu;

  /////////////////////////
  size_t oldSizes[2] = { NI, NJ };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "Convolution2D_kernel", 2);
  NI = newSizes[0];
  NJ = newSizes[1];
  /////////////////////////

  A = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  B_outputFromGpu = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));

  init(A);
   
  platform = new Platform(PLATFORM_ID);
  context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  Queue queue = Queue(*context,device,Queue::EnableProfiling);

  cl_mem_init(A,queue);
  SourceFile kernelFile = KERNEL_DIRECTORY KERNEL_FILE_NAME; 
 
  // Create a program from the kernel source
  Program program(context,kernelFile);
  if(!program.build(device)) {
      std::cout << "Error building the program: " << "\n";
      std::cout << program.getBuildLog(device) << "\n";
      return 1;
    }
 
  // Create the OpenCL kernel
  kernel = program.createKernel(kernelName.c_str()); 
  cl_launch_kernel(queue);


  queue.readBuffer(*b_mem_obj,NI * NJ * sizeof(DATA_TYPE),(void*) B_outputFromGpu);
  queue.finish();

  conv2D(A, B_outputFromGpu);

  free(A);
  free(B_outputFromGpu);

  cl_clean_up();
  return 0;
}
