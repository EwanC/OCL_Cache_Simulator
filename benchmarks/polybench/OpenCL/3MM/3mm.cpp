/**
 * 3mm.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
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

/* Problem size. */
# define NI 128 //512
# define NJ 128 //512
# define NK 128 //512
# define NL 128 //512
# define NM 128 //512

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KERNEL_FILE_NAME "3mm.cl"
#define PLATFORM_ID 0
#define DEVICE_ID 0


typedef float DATA_TYPE;


cl_int errcode;
Platform* platform;
Context* context;
Kernel* kernel1;
Kernel* kernel2;
Kernel* kernel3;

Buffer*  a_mem_obj;
Buffer*  b_mem_obj;
Buffer*  c_mem_obj;
Buffer*  d_mem_obj;
Buffer*  e_mem_obj;
Buffer*  f_mem_obj;
Buffer*  g_mem_obj;

std::string kernel1Name = "mm3_kernel1";
std::string kernel2Name = "mm3_kernel2";
std::string kernel3Name = "mm3_kernel3";

void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(G[i*NL + j], G_outputFromGpu[i*NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;				
			}
		}
	}
        
    assert(fail == 0 && "CPU - GPU Computation does not match!");
    std::cout << "Ok!\n";
}


void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}



void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G,Queue& queue)
{
	a_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadOnly, sizeof(DATA_TYPE) * NI * NK, NULL);
	b_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadOnly, sizeof(DATA_TYPE) * NK * NJ, NULL);
	c_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NJ * NM, NULL);
	d_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NM * NL, NULL);
	e_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NI * NJ, NULL);
	f_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NJ * NL, NULL);
	g_mem_obj = new Buffer(*(platform->getContext()), Buffer::ReadWrite, sizeof(DATA_TYPE) * NI * NL, NULL);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

    queue.writeBuffer(*a_mem_obj, sizeof(DATA_TYPE) * NI * NK, A);
    queue.writeBuffer(*b_mem_obj, sizeof(DATA_TYPE) * NK * NJ, B);
    queue.writeBuffer(*c_mem_obj, sizeof(DATA_TYPE) * NJ * NM, C);
    queue.writeBuffer(*d_mem_obj, sizeof(DATA_TYPE) * NM * NL, D);
    queue.writeBuffer(*e_mem_obj, sizeof(DATA_TYPE) * NI * NJ, E);	
    queue.writeBuffer(*f_mem_obj, sizeof(DATA_TYPE) * NJ * NL, F);
    queue.writeBuffer(*g_mem_obj, sizeof(DATA_TYPE) * NI * NL, G);
    queue.finish();
}

void cl_launch_kernel(Queue& queue)
{
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;
	int nm = NM;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	// Set the arguments of the kernel
	kernel1->setArgument( 0,*a_mem_obj);
	kernel1->setArgument( 1,*b_mem_obj);
	kernel1->setArgument( 2,*e_mem_obj);
	kernel1->setArgument( 3, sizeof(int), (void *)&ni);
	kernel1->setArgument( 4, sizeof(int), (void *)&nj);
	kernel1->setArgument( 5, sizeof(int), (void *)&nk);

	// Execute the OpenCL kernel
	queue.run(*kernel1, 2, 0, globalWorkSize, localWorkSize);
	
    //Second kernel

	globalWorkSize[0] = (size_t)ceil(((float)NL) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	kernel2->setArgument( 0,*c_mem_obj);
	kernel2->setArgument( 1,*d_mem_obj);
	kernel2->setArgument( 2,*f_mem_obj);
	kernel2->setArgument( 3, sizeof(int), (void *)&nj);
	kernel2->setArgument( 4, sizeof(int), (void *)&nl);
	kernel2->setArgument( 5, sizeof(int), (void *)&nm);

	// Execute the OpenCL kernel
	queue.run(*kernel2, 2, 0, globalWorkSize, localWorkSize);	
     
    //Third kernel

	globalWorkSize[0] = (size_t)ceil(((float)NL) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;
	
    kernel3->setArgument( 0,*e_mem_obj);
    kernel3->setArgument( 1,*f_mem_obj);
    kernel3->setArgument( 2,*g_mem_obj);
    kernel3->setArgument( 3, sizeof(int), (void *)&ni);
    kernel3->setArgument( 4, sizeof(int), (void *)&nl);
    kernel3->setArgument( 5, sizeof(int), (void *)&nj);
	
    // Execute the OpenCL kernel
	queue.run(*kernel3, 2, 0, globalWorkSize, localWorkSize);	
	queue.finish();
}

void cl_clean_up()
{
	// Clean up
	delete kernel1;
	delete kernel2;
	delete kernel3;
    delete platform;

	delete a_mem_obj;
	delete b_mem_obj;
	delete c_mem_obj;
	delete d_mem_obj;
	delete e_mem_obj;
	delete f_mem_obj;
	delete g_mem_obj;
}


void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int i,j,k;
	int intReps = 1;
	/* E := A*B */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{       DATA_TYPE tmp = 0;
			for (k = 0; k < NK; ++k)
			{
				tmp += A[i*NK + k] * B[k*NJ + j];
			}
                        tmp *= intReps;
			E[i*NJ + j] = tmp;
		}
	}
		
	/* F := C*D */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{       
                        DATA_TYPE tmp = 0;
			for (k = 0; k < NM; ++k)
			{
				tmp += C[i*NM + k] * D[k*NL + j];
			}
                        tmp *= intReps;                       
 			F[i*NL + j] = tmp;
		}
	}

  	/* G := E*F */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
                        DATA_TYPE tmp = 0;
			for (k = 0; k < NJ; ++k)
			{
				tmp += E[i*NJ + k] * F[k*NL + j];
			}
                        tmp *= intReps;
			G[i*NL + j] = tmp;
		}
	}
}


int main(void) 
{
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;
	DATA_TYPE* G_outputFromGpu;

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(NJ*NM*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NM*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	F = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	G = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
	G_outputFromGpu = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

	init_array(A, B, C, D);
	
    platform = new Platform(PLATFORM_ID);
    context = platform->getContext();
    Device device = platform->getDevice(DEVICE_ID);
    Queue queue(*context,device,Queue::EnableProfiling); 

    cl_mem_init(A, B, C, D, E, F, G,queue);

    Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
    if(!program.build(device)){
        std::cout << "Error building the program: \n";
        std::cout <<program.getBuildLog(device); 
    }
 
    kernel1=program.createKernel(kernel1Name.c_str());
    kernel2=program.createKernel(kernel2Name.c_str());
    kernel3=program.createKernel(kernel3Name.c_str());
    cl_launch_kernel(queue);

	queue.readBuffer(*g_mem_obj, sizeof(DATA_TYPE) * NI * NL, G_outputFromGpu);
    queue.finish();

	mm3_cpu(A, B, C, D, E, F, G);
	compareResults(G, G_outputFromGpu);
	cl_clean_up();

	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);
	free(G_outputFromGpu);

	return 0;
}

