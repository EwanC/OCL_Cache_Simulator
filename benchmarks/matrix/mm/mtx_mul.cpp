#include <iostream>
#include <sstream>
#include <string>

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
#define ELEMENT_LIMIT 5
#define DIMENSIONS 2
#define BLOCK_SIZE 32
#define W_SIZE BLOCK_SIZE * 5
#define H_SIZE BLOCK_SIZE * 5
#define DEVICE_ID 0
#define PLATFORM_ID 0
#define KERNEL_FILE_NAME "mm.cl"


//-----------------------------------------------------------------------------
void initialization(int argc, char** argv);
void hostMemoryAlloc();
void deviceMemoryAlloc();
void setKernelArguments();
void setKernelArgumentsWithLocal();
void setKernelArgumentsClassic();
void readResult();
void enqueWriteCommands(Queue& queue);
void enqueReadCommands(Queue& queue);
float random(float rand_min, float rand_max);
void run(const Context* context, Queue& queue);
void freeMemory();
void setNDRangeSizes();
void verifyResults();

//-----------------------------------------------------------------------------
// Runtime components.
Platform* platform;
Kernel* kernel;

// Host data.
float* hostA;
float* hostB;
float* hostC;

// Device data.
Buffer* A;
Buffer* B;
Buffer* C;

cl_uint* height = NULL;
cl_uint* width = NULL;

size_t* localWorkSize = NULL;
size_t* globalWorkSize = NULL;

std::string kernelName = "mm";

//-----------------------------------------------------------------------------

int main(int argc, char** argv) {
  srand((unsigned)time(0));
  platform = new Platform(PLATFORM_ID);
  Context* context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  std::cout << "Running " << kernelName << " on " << device.getName() << "\n";
  

  hostMemoryAlloc();
  deviceMemoryAlloc();
  SourceFile kernelFile = KERNEL_DIRECTORY KERNEL_FILE_NAME;

  Program program(context, KERNEL_DIRECTORY KERNEL_FILE_NAME);
  Queue queue = Queue(*context, device, Queue::EnableProfiling);
  enqueWriteCommands(queue);
  if(!program.build(device)) {
    std::cout << "Error building the program: " << "\n";
    std::cout << program.getBuildLog(device) << "\n";
    return 1;
  }
  kernel = program.createKernel(kernelName.c_str());


  setKernelArguments();
  setNDRangeSizes();
  run(context, queue);
  enqueReadCommands(queue);
  verifyResults();
  freeMemory();
  return 0;
}
//-----------------------------------------------------------------------------
void setNDRangeSizes() {
  localWorkSize = new size_t[2];
  globalWorkSize = new size_t[2];

  localWorkSize[0] = BLOCK_SIZE;
  localWorkSize[1] = BLOCK_SIZE;
  globalWorkSize[0] = W_SIZE;
  globalWorkSize[1] = H_SIZE;
}

//-----------------------------------------------------------------------------
void freeMemory() {
  delete [] hostA;
  delete [] hostB;
  delete [] hostC;
  delete A;
  delete B;
  delete C;
  delete kernel;
  delete platform;
  delete width;
  delete height;
  delete [] localWorkSize;
  delete [] globalWorkSize;
}

//-----------------------------------------------------------------------------
void hostMemoryAlloc() {
  hostA = new float [W_SIZE * H_SIZE];
  hostB = new float [W_SIZE * H_SIZE];
  hostC = new float [W_SIZE * H_SIZE];

  for(unsigned int row = 0; row < H_SIZE; row++) {
    for(unsigned int column = 0; column < W_SIZE; column++) {
      hostA[column + H_SIZE * row] = random(-ELEMENT_LIMIT, ELEMENT_LIMIT);
      hostB[column + H_SIZE * row] = random(-ELEMENT_LIMIT, ELEMENT_LIMIT);
    }
  }
}

//-----------------------------------------------------------------------------
void deviceMemoryAlloc() {
  C = new Buffer(*(platform->getContext()), Buffer::WriteOnly,
                 W_SIZE * H_SIZE * sizeof(float), NULL);
  A = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                 W_SIZE * H_SIZE * sizeof(float), NULL);
  B = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                 W_SIZE * H_SIZE * sizeof(float), NULL);
  width = new cl_uint(W_SIZE);
  height = new cl_uint(H_SIZE);
}

//-----------------------------------------------------------------------------
void enqueWriteCommands(Queue& queue) {
  queue.writeBuffer(*A, W_SIZE * H_SIZE * sizeof(float), (void*) hostA);
  queue.writeBuffer(*B, W_SIZE * H_SIZE * sizeof(float), (void*) hostB);
  queue.finish();
}

//-----------------------------------------------------------------------------
void enqueReadCommands(Queue& queue) {
  queue.readBuffer(*C, W_SIZE * H_SIZE * sizeof(float), (void*) hostC);
  queue.finish();
}

//-----------------------------------------------------------------------------
void setKernelArguments() {
  kernel->setArgument(0, *A);
  kernel->setArgument(1, *B);
  kernel->setArgument(2, *C);
  kernel->setArgument(3, sizeof(cl_int), (void*) width);
  kernel->setArgument(4, sizeof(cl_int), (void*) height);
}

//-----------------------------------------------------------------------------
void run(const Context* context, Queue& queue) {
  long executionTime = 0l;
  for (unsigned int repetition = 0; repetition < REPETITIONS; ++repetition) {
    Event runEvent(*context);
    queue.run(*kernel, DIMENSIONS, 0, globalWorkSize, localWorkSize, runEvent);
    queue.finish();
    executionTime += runEvent.computeDuration();
  }
 // std::cout <<"execution time:" << executionTime << "\n";
}

//-----------------------------------------------------------------------------
void verifyResults() {
  float* cpuHostC = new float [W_SIZE * H_SIZE];
 
  for(unsigned int row = 0; row < H_SIZE; ++row) {
    for(unsigned int column = 0; column < W_SIZE; ++column) {
      float result = 0.0f;
      for(unsigned int index = 0; index < W_SIZE; ++index) 
        result += hostA[row * W_SIZE + index] * hostB[index * W_SIZE + column];
      if(abs(hostC[row * W_SIZE + column] - result) >= 0.001f){
        std::cout << "Error in computation " <<hostC[row * W_SIZE + column] <<" "<<result <<"\n";
        exit(1);          
      }
      cpuHostC[row * W_SIZE + column] = result;
    }
  }
   std::cout <<"Ok!\n";

  delete [] cpuHostC;
}

//-----------------------------------------------------------------------------
float random(float rand_min, float rand_max) {
  float result =(float)rand()/(float)RAND_MAX;
  return ((1.0 - result) * rand_min + result *rand_max);
}
