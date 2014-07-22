/**
 * atax.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size. */
#define NX_DEFAULT 128 * 32
#define NY_DEFAULT 64

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "atax.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

size_t NX = NX_DEFAULT;

cl_int errcode;
Platform*  platform;
Context* context;
Kernel* kernel1;
Kernel* kernel2;

Buffer* a_mem_obj;
Buffer* x_mem_obj;
Buffer* tmp_mem_obj;

std::string kernel1Name="atax_kernel1";
std::string kernel2Name="atax_kernel2";


void init_array(DATA_TYPE *x, DATA_TYPE *A) {

  for (unsigned int column = 0; column < NY_DEFAULT; ++column) {
    x[column] = random<DATA_TYPE>();

    for (unsigned int row = 0; row < NX; ++row) {
      A[row * NY_DEFAULT + column] = random<DATA_TYPE>();
    }

  }
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp,Queue& queue) {
  
  a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX * NY_DEFAULT, NULL);
  
  x_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NY_DEFAULT, NULL);
  
  
  tmp_mem_obj  = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);


  queue.writeBuffer(*a_mem_obj,sizeof(DATA_TYPE) * NX * NY_DEFAULT, A);
  queue.writeBuffer(*x_mem_obj,sizeof(DATA_TYPE) * NY_DEFAULT, x);
  queue.writeBuffer(* tmp_mem_obj,sizeof(DATA_TYPE) * NX, tmp);
  queue.finish();
}


void cl_launch_kernel(Queue& queue) {

  int nx = NX;
  int ny = NY_DEFAULT;

  size_t oldLocalWorkSize[1], globalWorkSize[1];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[0] = NX;

  ///////////////////////////////////////////////
  size_t localWorkSize[1];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize, "atax_kernel1", 1);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  kernel1->setArgument( 0,*a_mem_obj);
  kernel1->setArgument( 1,*x_mem_obj);
  kernel1->setArgument( 2,*tmp_mem_obj);
  kernel1->setArgument( 3, sizeof(int), (void *)&nx);
  kernel1->setArgument( 4, sizeof(int), (void *)&ny);
  
  // Execute the OpenCL kernel
  queue.run(*kernel1, 1,0, globalWorkSize,localWorkSize);
  queue.finish();
}

void cl_clean_up() {
  // Clean up
  delete kernel1;
  delete kernel2;
  delete platform;
 
  delete a_mem_obj;
  delete x_mem_obj;
  delete tmp_mem_obj;
  
}

void atax_cpu(DATA_TYPE *A, DATA_TYPE *x,DATA_TYPE *result) {

  
  int intReps = 1;
  

  for (int row = 0; row < 32; row++) {
    DATA_TYPE tmp  = 0;
    for (int rep = 0; rep < intReps; ++rep) {
      for (int column = 0; column < NY_DEFAULT; column++) {
        tmp += A[row * NY_DEFAULT + column] * x[column];
      }
    }

    assert(fabs(tmp - result[row]) < 1 && "Error!");
  }

  std::cout << "Ok!\n";
}

int main(void) {

  DATA_TYPE *A;
  DATA_TYPE *x;
  DATA_TYPE *tmp;

  /////////////////////////
  size_t oldSizes[1] = { NX };
  size_t newSizes[1];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "atax_kernel1", 1);
  NX = newSizes[0];
  /////////////////////////

  A = (DATA_TYPE *)malloc(NX * NY_DEFAULT * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(NY_DEFAULT * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  init_array(x, A);
  
  platform = new Platform(PLATFORM_ID);
  context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  Queue queue(*context,device,Queue::EnableProfiling); 
  

  cl_mem_init(A, x, tmp ,queue);

  Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
  if(!program.build(device)){
    std::cout << "Error building the program: \n";
    std::cout <<program.getBuildLog(device); 
  }

  kernel1=program.createKernel(kernel1Name.c_str());
  cl_launch_kernel(queue);

  queue.readBuffer(*tmp_mem_obj,NX_DEFAULT * sizeof(DATA_TYPE), tmp);
  queue.finish();

  atax_cpu(A, x, tmp);
  cl_clean_up();

  free(A);
  free(x);
  free(tmp);

  return 0;
}
