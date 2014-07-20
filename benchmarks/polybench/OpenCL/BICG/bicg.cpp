/**
 * bicg.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size. */
#define NX_DEFAULT 256 //4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256

#ifndef M_PI
#define M_PI 3.14159
#endif

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "bicg.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

cl_int errcode;
Platform* platform;
Context* context;
Kernel* kernel1;
Kernel* kernel2;

Buffer* a_mem_obj;
Buffer* r_mem_obj;
Buffer* p_mem_obj;
Buffer* q_mem_obj;
Buffer* s_mem_obj;

std::string kernel1Name = "bicgKernel1";
std::string kernel2Name = "bicgKernel2";

size_t NX = NX_DEFAULT;

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r) {
  unsigned int i, j;

  for (i = 0; i < NX; i++) {
    r[i] = i * M_PI;

    for (j = 0; j < NX; j++) {
      A[i * NX + j] = random<DATA_TYPE>();
    }
  }

  for (i = 0; i < NX; i++) {
    p[i] = i * M_PI;
  }
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p, DATA_TYPE *q,Queue& queue) {
  
  a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX * NX, NULL);
  r_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);
  s_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);
  p_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);
  q_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);


  queue.writeBuffer(*a_mem_obj,sizeof(DATA_TYPE) * NX * NX, A);
  queue.writeBuffer(*r_mem_obj,sizeof(DATA_TYPE) * NX, r);
  queue.writeBuffer(*s_mem_obj,sizeof(DATA_TYPE) * NX, s);
  queue.writeBuffer(*p_mem_obj,sizeof(DATA_TYPE) * NX, p);
  queue.writeBuffer(*q_mem_obj,sizeof(DATA_TYPE) * NX, q);
  queue.finish();
}


void cl_launch_kernel(Queue& queue) {
  int nx = NX;
  int ny = NX;

  size_t oldLocalWorkSize[1], globalWorkSize[1];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[0] = NX;

  ///////////////////////////////////////////////
  size_t localWorkSize[1];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize, "bicgKernel1", 1);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
 kernel1->setArgument( 0,*a_mem_obj);
 kernel1->setArgument( 1,*p_mem_obj);
 kernel1->setArgument( 2,*q_mem_obj);
 kernel1->setArgument( 3, sizeof(int), &nx);
 kernel1->setArgument( 4, sizeof(int), &ny);

  // Execute the 1st OpenCL kernel
  queue.run(*kernel1, 1,0, globalWorkSize,localWorkSize);
  queue.finish();

  //	globalWorkSize[0] = (size_t)ceil(((float)NX) /
  //((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
  //	globalWorkSize[1] = 1;
  //
  //	// Set the arguments of the kernel
  //	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void
  //*)&a_mem_obj);
  //	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void
  //*)&r_mem_obj);
  //	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void
  //*)&s_mem_obj);
  //	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), &nx);
  //        errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), &ny);
  //	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
  //
  //	// Execute the 2nd OpenCL kernel
  //	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL,
  //globalWorkSize, localWorkSize, 0, NULL, NULL);
  //	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
  //	clFinish(clCommandQue);
}

void cl_clean_up() {
  // Clean up
  delete kernel1;
  delete kernel2;
  delete platform;  

  delete a_mem_obj;
  delete p_mem_obj;
  delete q_mem_obj;
  delete r_mem_obj;
  delete s_mem_obj;
}

void bicg_cpu(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE *result) {
  unsigned int i, j;

  int intReps = 1;
 

  for (i = 0; i < NX; i++) {
    q[i] = 0.0;

    for (int rep = 0; rep < intReps; ++rep) {
      for (j = 0; j < NX; j++) {
        q[i] += A[i * NX + j] * p[j];
      }
    }
   if(fabs(q[i] - result[i]) / result[i] >= 0.5){
     std::cout <<"gpu "<<result[i] << " cpu "<<q[i]<<std::endl;
   }
    assert(fabs(q[i] - result[i]) / result[i] < 0.5 && "Error in the computation");
  }

  std::cout << "Ok!\n";
}

int main(void) {
  DATA_TYPE *A;
  DATA_TYPE *r;
  DATA_TYPE *s;
  DATA_TYPE *p;
  DATA_TYPE *q;
  DATA_TYPE *s_outputFromGpu;
  DATA_TYPE *q_outputFromGpu;

  /////////////////////////
  size_t oldSizes[1] = { NX };
  size_t newSizes[1];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "bicgKernel1", 1);
  NX = newSizes[0];
  /////////////////////////

  A = (DATA_TYPE *)malloc(NX * NX * sizeof(DATA_TYPE));
  r = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  s = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  p = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  q = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  s_outputFromGpu = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  q_outputFromGpu = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  init_array(A, p, r);
   
  platform = new Platform(PLATFORM_ID);
  context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  Queue queue(*context,device,Queue::EnableProfiling); 
 
  cl_mem_init(A, r, s, p, q,queue);
 
  Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
  if(!program.build(device)){
    std::cout << "Error building the program: \n";
    std::cout <<program.getBuildLog(device); 
  }

  kernel1=program.createKernel(kernel1Name.c_str());
  cl_launch_kernel(queue);


//  queue.readBuffer(*s_mem_obj, NX * sizeof(DATA_TYPE), s_outputFromGpu);
  queue.readBuffer(*q_mem_obj,NX * sizeof(DATA_TYPE), q_outputFromGpu);
  queue.finish();
  bicg_cpu(A, p, q, q_outputFromGpu);
  cl_clean_up();

  free(A);
  free(r);
  free(s);
  free(p);
  free(q);
  free(s_outputFromGpu);
  free(q_outputFromGpu);

  return 0;
}
