/**
 * 2mm.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define NI_DEFAULT 64//2048
#define NJ_DEFAULT 64//2048
#define NK_DEFAULT 64//2048
#define NL_DEFAULT 64//2048

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "2mm.cl"
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
Buffer* b_mem_obj;
Buffer* c_mem_obj;
Buffer* d_mem_obj;
Buffer* e_mem_obj;

size_t NJ = NJ_DEFAULT;
size_t NK = NK_DEFAULT;
size_t NI = NI_DEFAULT; 
size_t NL = NL_DEFAULT;

std::string kernel1Name="mm2_kernel1";
std::string kernel2Name="mm2_kernel2";


void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D) {
  unsigned int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NI + j] = random<DATA_TYPE>();
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NK + j] = random<DATA_TYPE>();
    }
  }

  
  for(i=0;i<NI;i++){
    for(j=0;j<NL;j++){
       D[i*NL+j] = random<DATA_TYPE>();
    }
  }
}

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,DATA_TYPE *E,Queue& queue) {

  a_mem_obj = new Buffer(*(platform->getContext()),Buffer::ReadOnly,sizeof(DATA_TYPE) * NI * NK, NULL);

  b_mem_obj = new Buffer(*(platform->getContext()),Buffer::ReadOnly,sizeof(DATA_TYPE) * NK * NJ, NULL);

  c_mem_obj = new Buffer(*(platform->getContext()),Buffer::ReadWrite,sizeof(DATA_TYPE) * NI * NJ, NULL);
  
  d_mem_obj = new Buffer(*(platform->getContext()),Buffer::ReadWrite,sizeof(DATA_TYPE) * NJ * NL, NULL);
 
  e_mem_obj = new Buffer(*(platform->getContext()),Buffer::ReadWrite,sizeof(DATA_TYPE) * NI * NL, NULL);


  queue.writeBuffer(*a_mem_obj,sizeof(DATA_TYPE) * NI * NK,A);
  queue.writeBuffer(*b_mem_obj,sizeof(DATA_TYPE) * NK * NJ,B);
  queue.writeBuffer(*c_mem_obj,sizeof(DATA_TYPE) * NI * NJ,C);
  queue.writeBuffer(*d_mem_obj,sizeof(DATA_TYPE) * NJ * NL,D);
  queue.writeBuffer(*e_mem_obj,sizeof(DATA_TYPE) * NI * NL,E);
  queue.finish();
}


void cl_launch_kernel(Queue& queue) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  size_t oldLocalWorkSize[2], globalWorkSize[2];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  oldLocalWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
  globalWorkSize[0] = NI;
  globalWorkSize[1] = NL;

  ///////////////////////////////////////////////
  size_t localWorkSize[2];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize,
              "mm2_kernel1", 2);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  kernel1->setArgument(0,*a_mem_obj);
  kernel1->setArgument(1,*b_mem_obj);
  kernel1->setArgument(2,*c_mem_obj);
  kernel1->setArgument(3,sizeof(int),(void*)&ni);
  kernel1->setArgument(4,sizeof(int),(void*)&nk);
  kernel1->setArgument(5,sizeof(int),(void*)&nj);

  queue.run(*kernel1,2,0,globalWorkSize,localWorkSize);


  globalWorkSize[0] = NI;
  globalWorkSize[1] = NL;

  kernel2->setArgument( 0, *c_mem_obj);
  kernel2->setArgument( 1, *d_mem_obj);
  kernel2->setArgument( 2, *e_mem_obj);
  kernel2->setArgument( 3, sizeof(int), (void *)&ni);
  kernel2->setArgument( 4, sizeof(int), (void *)&nj);
  kernel2->setArgument( 5, sizeof(int), (void *)&nl);

  queue.run(*kernel2, 2, 0, globalWorkSize, localWorkSize);
  queue.finish();
}

void cl_clean_up() {
  // Clean up
  delete kernel1;
  delete kernel2;
  delete platform;

  delete a_mem_obj;
  delete b_mem_obj;
  delete c_mem_obj;
  delete d_mem_obj;
  delete e_mem_obj;
}

void compareResults(DATA_TYPE *E, DATA_TYPE *E_outputFromGpu) {
  unsigned int i, j, fail;
  fail = 0;

  // Compare a and b
  for (i = 1; i < NL; i++) {
    for (j = 1; j < NI; j++) {
      if (percentDiff(E[i * NI + j], E_outputFromGpu[i * NI + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  assert(fail == 0 && "CPU - GPU Computation does not match!");

  std::cout << "Ok!\n";
}

void mm2_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C,DATA_TYPE *D,DATA_TYPE *E) {
  unsigned int i, j, k;
  int intReps = 1;
  
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      DATA_TYPE tmp = 0;
      for (k = 0; k < NK; ++k) {
        tmp += A[i * NK + k] * B[k * NJ + j];
      }
      tmp *= intReps;
      C[i *NJ + j] = tmp;
    }
  }

    
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      DATA_TYPE tmp = 0;
      for (k = 0; k < NJ; ++k) {
         tmp += C[i * NJ + k] * D[k * NL + j];       
      }
      tmp *= intReps;
      E[i*NL+j] = tmp; 

    }
  }


}



int main(void) {
  DATA_TYPE *C;
  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *D;
  DATA_TYPE *E;
  DATA_TYPE *E_outputFromGpu;

  /////////////////////////
  size_t oldSizes[2] = { NI, NL };
  size_t newSizes[2];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "mm2_kernel1", 2);
//  size_t tmpSizes[2] = {newSizes[0], newSizes[1]};
//  getNewSizes(tmpSizes, NULL, newSizes, NULL, "mm2_kernel2", 2);
  NI = newSizes[0];
  NL = newSizes[1];
  NJ = NI;
  NK = NI;
  /////////////////////////

  C = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));
  E_outputFromGpu = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));

  init_array(A, B, C, D);
 
  platform = new Platform(PLATFORM_ID);
  context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  Queue queue(*context,device,Queue::EnableProfiling); 


  cl_mem_init(A, B, C, D, E,queue);
  
  Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
  if(!program.build(device)){
    std::cout << "Error building the program: \n";
    std::cout <<program.getBuildLog(device); 
  }

  kernel1=program.createKernel(kernel1Name.c_str());
  kernel2=program.createKernel(kernel2Name.c_str());
  cl_launch_kernel(queue);

 

  queue.readBuffer(*e_mem_obj,sizeof(DATA_TYPE) * NI * NL, E_outputFromGpu);
  queue.finish();
 
  mm2_cpu(A, B,C,D,E);
  compareResults(E,E_outputFromGpu);
  cl_clean_up();

  free(C);
  free(A);
  free(B);
  free(D);
  free(E);
  free(E_outputFromGpu);

  return 0;
}
