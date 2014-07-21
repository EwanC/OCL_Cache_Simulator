#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#include <boost/filesystem.hpp>

#include "Buffer.h"
#include "Device.h"
#include "Event.h"
#include "Kernel.h"
#include "Platform.h"
#include "Program.h"
#include "Queue.h"

#include "SystemConfiguration.h"

using namespace boost::filesystem;
//-----------------------------------------------------------------------------
#define REPETITIONS 1
#define DIMENSIONS 3
#define DEVICE_ID 0
#define PLATFORM_ID 0
#define KERNEL_FILE_NAME "stencil.cl"


//-----------------------------------------------------------------------------
void initialization(int argc, char** argv);
void hostMemoryAlloc();
void deviceMemoryAlloc();
void setKernelArguments();
void writeResults();
void enqueWriteCommands(Queue& queue);
void enqueReadCommands(Queue& queue);
void run(const Context* context, Queue& queue);
void freeMemory();
int read_data(float *A0, unsigned int nx,unsigned int ny,unsigned int nz); 

//-----------------------------------------------------------------------------
// Runtime components.
Platform* platform;
Kernel* kernel;

std::string kernelName = "naive_kernel";

//host data
float *h_A0;
float *h_Anext;
  
//device
Buffer* d_A0;
Buffer* d_Anext;
Buffer* d_temp;

FILE* fd;

unsigned int nx,ny,nz;
int size;
int iteration;
float c0=1.0f/6.0f;
float c1=1.0f/6.0f/6.0f;

//-----------------------------------------------------------------------------

int main(int argc, char** argv) {
  initialization(argc, argv);
  platform = new Platform(PLATFORM_ID);
  Context* context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  std::cout << "Running " << kernelName << " on " << device.getName() << "\n";
  
  hostMemoryAlloc();

  read_data(h_A0, nx,ny,nz);
  fclose(fd);
  memcpy (h_Anext,h_A0,sizeof(float)*size);

  deviceMemoryAlloc();

  
  SourceFile kernelFile = KERNEL_DIRECTORY KERNEL_FILE_NAME;

  Program program(context, kernelFile);
  Queue queue(*context, device, Queue::EnableProfiling);
  if(!program.build(device)) {
    std::cout << "Error building the program: " << "\n";
    std::cout << program.getBuildLog(device) << "\n";
    return 1;
  }
  kernel = program.createKernel(kernelName.c_str());
  setKernelArguments();

  enqueWriteCommands(queue);

  run(context, queue);
  enqueReadCommands(queue);

  writeResults();
  freeMemory();
  return 0;
}




//-----------------------------------------------------------------------------
void initialization(int argc, char** argv) {
 if (argc<6) 
 {
       printf("Usage: probe nx ny nz t file\n"
       "nx: the grid size x\n"
       "ny: the grid size y\n"
       "nz: the grid size z\n"
       "t: the iteration time\n"
       "file: the input file\n");
       exit(1);
  }
  
  nx = atoi(argv[1]);
  if (nx<1){
    printf("grid size x must be greater than zero\n");
    exit(1);    
  }
  ny = atoi(argv[2]);
  if (ny<1){
    printf("grid size y must be greater than zero\n");
    exit(1);  
  }
  nz = atoi(argv[3]);
  if (nz<1){
    printf("grid size z must be greater than zero\n");
    exit(1);  
  }
    
  iteration = atoi(argv[4]);
  if(iteration<1){
    printf("number of iterations must be greater than zero\n");
    exit(1);  
  }

  fd = fopen(argv[5], "r");
  if(fd==NULL){
    printf("error opening file\n");
    exit(1);
  }

  


}

//-----------------------------------------------------------------------------
void freeMemory() {
  
  delete[] h_A0;
  delete[] h_Anext;
  
  delete kernel;
  delete platform;
}

//-----------------------------------------------------------------------------
void hostMemoryAlloc() {
   // allocate host memory
   size=nx*ny*nz;
  
   h_A0=(float*)malloc(sizeof(float)*size);
   h_Anext=(float*)malloc(sizeof(float)*size);

}

//-----------------------------------------------------------------------------
void deviceMemoryAlloc() {
  d_A0 = new Buffer(*(platform->getContext()), Buffer::ReadWrite,
                 size*sizeof(float), NULL);
 
  d_Anext = new Buffer(*(platform->getContext()), Buffer::ReadWrite,
                 size*sizeof(float), NULL);
  

}

//-----------------------------------------------------------------------------
void enqueWriteCommands(Queue& queue) {
  queue.writeBuffer(*d_A0,size*sizeof(float), (void*) h_A0);
  queue.writeBuffer(*d_Anext,size*sizeof(float), (void*)h_Anext);
}

//-----------------------------------------------------------------------------
void enqueReadCommands(Queue& queue) {
   // copy result from device to host

  queue.readBuffer(*d_Anext,size*sizeof(float), (void*) h_Anext);

}

//-----------------------------------------------------------------------------
void setKernelArguments() {

  kernel->setArgument(0, sizeof(float),(void*)&c0);
  kernel->setArgument(1, sizeof(float),(void*)&c0);
  kernel->setArgument(2, *d_A0);
  kernel->setArgument(3, *d_Anext);
  kernel->setArgument(4,sizeof(int),(void*)&nx);
  kernel->setArgument(5,sizeof(int),(void*)&ny);
  kernel->setArgument(6,sizeof(int),(void*)&nz);
 

}

//-----------------------------------------------------------------------------
void run(const Context* context, Queue& queue) {
 
  //only use 1D thread block
  unsigned int tx =64;
  size_t block[3] = {tx,1,1};
  size_t grid[3] = {(nx-2+tx-1)/tx*tx,ny-2,nz-2};
  // size_t offset[3] = {1,1,1};

  printf("global size %d %d %d\n",grid[0],grid[1],grid[2] );

  int t;
  for(t=0;t<iteration;t++)
  {
    queue.run(*kernel, DIMENSIONS, 0, grid, block);
 
    d_temp = d_A0;
    d_A0 = d_Anext;
    d_Anext = d_temp; 
   
    kernel->setArgument(2, *d_A0);
    kernel->setArgument(3, *d_Anext);

  }

  d_temp = d_A0;
  d_A0 = d_Anext;
  d_Anext = d_temp;

  queue.finish();

}

int read_data(float *A0, unsigned int nx,unsigned int ny,unsigned int nz) 
{ 
  int s=0;
  unsigned int i,j,k;
  for(i=0;i<nz;i++)
  {
    for(j=0;j<ny;j++)
    {
      for(k=0;k<nx;k++)
      {
        fread(A0+s,sizeof(float),1,fd);
        s++;
      }
    }
  }
  return 0;
}

void writeResults(){
   FILE *fid = fopen("stencil.out","w");
   
  uint32_t tmp32;
  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }
 
  tmp32 = nx*ny*nz;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);
  fwrite(h_A0, sizeof(float), tmp32, fid);

  fclose (fid);
}
