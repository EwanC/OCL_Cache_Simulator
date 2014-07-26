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
#define NX_DEFAULT 1024 //4096
#define NY_DEFAULT 1024 // 4096

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
size_t NY = NY_DEFAULT;

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r) {
  unsigned int i, j;

  for (i = 0; i < NX; i++) {
    r[i] = i * M_PI;

    for (j = 0; j < NY; j++) {
      A[i * NY + j] = random<DATA_TYPE>();
    }
  }

  for (i = 0; i < NY; i++) {
    p[i] = i * M_PI;
  }
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p, DATA_TYPE *q,Queue& queue) {
  
  a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX * NY, NULL);
  r_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);
  s_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);
  p_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);
  q_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * NX, NULL);


  queue.writeBuffer(*a_mem_obj,sizeof(DATA_TYPE) * NX * NY, A);
  queue.writeBuffer(*r_mem_obj,sizeof(DATA_TYPE) * NX, r);
  queue.writeBuffer(*s_mem_obj,sizeof(DATA_TYPE) * NX, s);
  queue.writeBuffer(*p_mem_obj,sizeof(DATA_TYPE) * NX, p);
  queue.writeBuffer(*q_mem_obj,sizeof(DATA_TYPE) * NX, q);
  queue.finish();
}


void cl_launch_kernel(Queue& queue) {
  int nx = NX;
  int ny = NY;

  size_t localWorkSize[2], globalWorkSize[2];
  localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
 
  globalWorkSize[0] = (size_t)ceil(((float)NX) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;;
  globalWorkSize[1] = 1;


  // Set the arguments of the kernel
  kernel1->setArgument( 0,*a_mem_obj);
  kernel1->setArgument( 1,*p_mem_obj);
  kernel1->setArgument( 2,*q_mem_obj);
  kernel1->setArgument( 3, sizeof(int), &nx);
  kernel1->setArgument( 4, sizeof(int), &ny);

  // Execute the 1st OpenCL kernel
  queue.run(*kernel1, 1,0, globalWorkSize,localWorkSize);
  queue.finish();

  globalWorkSize[0] = (size_t)ceil(((float)NY) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;

  
  // Set the arguments of the kernel
  kernel2->setArgument(0, *a_mem_obj);
  kernel2->setArgument(1, *r_mem_obj);
  kernel2->setArgument(2, *s_mem_obj);
  kernel2->setArgument(3, sizeof(int), &nx);
  kernel2->setArgument(4, sizeof(int), &ny);
  
  // Execute the 2nd OpenCL kernel
  queue.run(*kernel2, 1, 0,globalWorkSize, localWorkSize);
  queue.finish();
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

void bicg_cpu(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r,DATA_TYPE *q, DATA_TYPE *s, DATA_TYPE *q_gpu,DATA_TYPE *s_gpu) {
  unsigned int i, j;

  int intReps = 1;
 

  for (i = 0; i < NY; i++)
  {
    s[i] = 0.0;
  }


  for (i = 0; i < NX; i++) {
    q[i] = 0.0;

    for (int rep = 0; rep < intReps; ++rep) {
      for (j = 0; j < NY; j++) {
          s[j] += r[i] * A[i*NY + j]; 
          q[i] += A[i * NY + j] * p[j];
      }
    }

    assert(fabs(q[i] - q_gpu[i]) / q_gpu[i] < 0.5 && "Error in the computation");
  }

   for (i = 0; i < NY; i++)
  {
    assert(fabs(s[i] - s_gpu[i]) / s_gpu[i] < 0.5 && "Error in the computation");
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

  A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
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
  kernel2=program.createKernel(kernel2Name.c_str());
  cl_launch_kernel(queue);


  queue.readBuffer(*s_mem_obj, NX * sizeof(DATA_TYPE), s_outputFromGpu);
  queue.readBuffer(*q_mem_obj, NX * sizeof(DATA_TYPE), q_outputFromGpu);
  queue.finish();
  bicg_cpu(A, p, r,q,s, q_outputFromGpu,s_outputFromGpu);
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
