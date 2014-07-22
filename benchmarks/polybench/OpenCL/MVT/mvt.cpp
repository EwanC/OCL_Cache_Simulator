/**
 * mvt.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define N_DEFAULT 512 //4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256

#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64) // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "mvt.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0


/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;

cl_int errcode;
Platform* platform;
Context* context;

Kernel* kernel1;
Kernel* kernel2;

Buffer* a_mem_obj;
Buffer* x1_mem_obj;
Buffer* x2_mem_obj;
Buffer* y1_mem_obj;
Buffer* y2_mem_obj;

std::string kernel1Name = "mvt_kernel1";
std::string kernel2Name = "mvt_kernel2";

size_t N = N_DEFAULT;


void init_arrays(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y_1,
                 DATA_TYPE *y_2) {
  unsigned int i, j;

  for (i = 0; i < N; i++) {
    x1[i] = 0.0;
    x2[i] = 0.0;
    y_1[i] = 0.0;
    y_2[i] = 0.0;

    for (j = 0; j < N; j++) {
      a[i * N + j] = random<DATA_TYPE>();
    }
  }
}

void cl_mem_init(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y_1,DATA_TYPE *y_2,Queue& queue) {

  a_mem_obj =  new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * N * N, NULL);
  x1_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * N, NULL);
  x2_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * N, NULL);
  y1_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * N, NULL);
  y2_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite,sizeof(DATA_TYPE) * N, NULL);


  queue.writeBuffer(*a_mem_obj,  sizeof(DATA_TYPE) * N * N, a);
  queue.writeBuffer(*x1_mem_obj, sizeof(DATA_TYPE) * N, x1 );
  queue.writeBuffer(*x2_mem_obj, sizeof(DATA_TYPE) * N, x2 );
  queue.writeBuffer(*y1_mem_obj, sizeof(DATA_TYPE) * N, y_1);
  queue.writeBuffer(*y2_mem_obj, sizeof(DATA_TYPE) * N, y_2);
  queue.finish();
}

void cl_launch_kernel(Queue& queue) {
  int n = N;

  size_t oldLocalWorkSize[1], globalWorkSize[1];
  oldLocalWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
  globalWorkSize[0] = N;

  ///////////////////////////////////////////////
  size_t localWorkSize[1];
  getNewSizes(NULL, oldLocalWorkSize, NULL, localWorkSize, "mvt_kernel1", 1);
  ///////////////////////////////////////////////

  // Set the arguments of the kernel
  kernel1->setArgument( 0,*a_mem_obj);
  kernel1->setArgument( 1,*x1_mem_obj);
  kernel1->setArgument( 2,*y1_mem_obj);
  kernel1->setArgument( 3, sizeof(int), (void *)&n);

  // Execute the OpenCL kernel
  queue.run(*kernel1, 1,0, globalWorkSize,localWorkSize);

  // Set the arguments of the kernel
 kernel2->setArgument( 0,*a_mem_obj);
 kernel2->setArgument( 1,*x2_mem_obj);
 kernel2->setArgument( 2,*y2_mem_obj);
 kernel2->setArgument( 3, sizeof(int), (void *)&n);
  
 // Execute the OpenCL kernel
 queue.run(*kernel2, 1, 0,globalWorkSize,localWorkSize);

 queue.finish();
}

void cl_clean_up() {
  // Clean up
  delete platform;
  delete kernel1;
  delete kernel2;
  delete a_mem_obj;
  delete x1_mem_obj;
  delete x2_mem_obj;
  delete y1_mem_obj;
  delete y2_mem_obj;
}

void runMvt(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
            DATA_TYPE *y2, DATA_TYPE *x1_result,DATA_TYPE *x2_result) {
  unsigned int i, j, k, l;

  int intReps = 1;

  for (i = 0; i < N; i++) {
    for (int rep = 0; rep < intReps; ++rep) {
      for (j = 0; j < N; j++) {
        x1[i] = x1[i] + a[i * N + j] * y1[j];
      }
    }

    std::cout << x1[i] << " " << result[i] << "\n";
    assert(fabs(x1[i] - x1_result[i]) < 0.01 && "Error!");
  }

  for(k=0;k<N;k++){
    for(l=0;l<N;l++){
       x2[k] = x2[k] + a[k*N +l] * y2[l];
   
    }
    std::cout<<x2[k]<<std::endl;
    assert(fabs(x2[k] - x2_result[k]) < 0.01 && "Error!");
  }

  std::cout << "Ok!\n";
}

int main(void) {
  DATA_TYPE *a;
  DATA_TYPE *x1;
  DATA_TYPE *x2;
  DATA_TYPE *x1_outputFromGpu;
  DATA_TYPE *x2_outputFromGpu;
  DATA_TYPE *y_1;
  DATA_TYPE *y_2;

  /////////////////////////
  size_t oldSizes[1] = { N };
  size_t newSizes[1];
  getNewSizes(oldSizes, NULL, newSizes, NULL, "mvt_kernel1", 1);
  N = newSizes[0];
  /////////////////////////

  a = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x1_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  init_arrays(a, x1, x2, y_1, y_2);

  platform = new Platform(PLATFORM_ID);
  context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  Queue queue(*context,device,Queue::EnableProfiling); 
  
  cl_mem_init(a, x1, x2, y_1, y_2,queue);
  
  Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
  if(!program.build(device)){
           std::cout << "Error building the program: \n";
           std::cout <<program.getBuildLog(device); 
  }
  kernel1=program.createKernel(kernel1Name.c_str());
  kernel2=program.createKernel(kernel2Name.c_str());
  cl_launch_kernel(queue);


  queue.readBuffer(*x1_mem_obj,N * sizeof(DATA_TYPE), x1_outputFromGpu);
  queue.readBuffer(*x2_mem_obj,N * sizeof(DATA_TYPE), x2_outputFromGpu);
  queue.finish();

  runMvt(a, x1, x2, y_1, y_2, x1_outputFromGpu,x2_outputFromGpu);
  cl_clean_up();

  free(a);
  free(x1);
  free(x2);
  free(x1_outputFromGpu);
  free(x2_outputFromGpu);
  free(y_1);
  free(y_2);

  return 0;
}

