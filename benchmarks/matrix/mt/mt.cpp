#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <iomanip>
#include <boost/filesystem.hpp>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "Buffer.h"
#include "Device.h"
#include "Event.h"
#include "Kernel.h"
#include "Platform.h"
#include "Program.h"
#include "Queue.h"

#include "bench_support.h"
#include "SystemConfiguration.h"

using namespace boost::filesystem;
namespace po = boost::program_options;

//-----------------------------------------------------------------------------
#define REPETITIONS 1
#define ELEMENT_LIMIT 5   
#define DIMENSIONS 2     
#define MT "mt"
#define MT_LOCAL "mtLocal"
#define KERNEL_FILE_NAME "mt.cl"

#define L_SIZE 16
//-----------------------------------------------------------------------------
unsigned long computeExecutionTime(cl_event& event);
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
void freeMemory();
void printVector(const float* vector, unsigned int size); 
void printMatrix(const float* matrix, size_t width, size_t height);
void setSizes();
void verifyResults();

//-----------------------------------------------------------------------------
// Runtime components.
Platform* platform;
Kernel* kernel;

// Host data.
float* hostA;
float* hostB;

// Device data.
Buffer* A;
Buffer* B;

cl_uint* height = NULL;
cl_uint* width = NULL;

size_t* localWorkSize = NULL;
size_t* globalWorkSize = NULL;

std::string kernelName = "";
unsigned int blockSizeX = 0;
unsigned int blockSizeY = 0;
unsigned int unrollFactor = 0;

int PLATFORM_ID = 0;
int DEVICE_ID = 0;

int WIDTH = L_SIZE * 5;   //2    //mtLocal *5  //4096 = 16 * 256
int HEIGHT = L_SIZE * 5;   //2   // *5   //4096

//-----------------------------------------------------------------------------
// Usage: ./mt --kernelName [mt | mtLocal]
int main(int argc, char** argv) {
  initialization(argc, argv);
  platform = new Platform(PLATFORM_ID);
  Context* context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID); 
  std::cout << "Running "<<kernelName << " on " << device.getName() << "\n";
  hostMemoryAlloc();
  deviceMemoryAlloc();
  SourceFile kernelFile = KERNEL_DIRECTORY KERNEL_FILE_NAME;

 
  Program program(context,KERNEL_DIRECTORY KERNEL_FILE_NAME);
  Queue queue(*context, device, Queue::EnableProfiling);
  enqueWriteCommands(queue);
 
  
  bool buildResult = program.build(device);

  if(!buildResult) {
    std::cout << "Error building the program\n";
    std::cout << program.getBuildLog(device) << "\n";
    return 1;
  }
  //std::cout << program.getBinary(device) << "\n";
  kernel = program.createKernel(kernelName.c_str());
  
  setSizes();
  setKernelArguments();
  //long executionTime = 0l;

  for (unsigned int repetition = 0; repetition < REPETITIONS; ++repetition) {
    if(localWorkSize[0] == -1 || localWorkSize[1] == -1)
      localWorkSize = NULL;


    queue.run(*kernel, DIMENSIONS, 0, globalWorkSize, localWorkSize);

    queue.finish();
  }

  enqueReadCommands(queue);
  verifyResults();
  freeMemory();
  return 0;
}

//-----------------------------------------------------------------------------
void initialization(int argc, char** argv) {
  srand((unsigned)time(0));
  po::options_description cmdOptions("Allowed options");
  cmdOptions.add_options() 
    ("kernelName", po::value<std::string>(), "kernelName");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, cmdOptions), vm);
  po::notify(vm);
  kernelName = vm["kernelName"].as<std::string>();
  blockSizeX = L_SIZE;
  blockSizeY = L_SIZE;
  getPlatformDevice(&PLATFORM_ID, &DEVICE_ID);
}

//-----------------------------------------------------------------------------
void setSizes() {
  if(kernelName == MT || kernelName == MT_LOCAL) {
    localWorkSize[0] = blockSizeX;
    localWorkSize[1] = blockSizeY;
    globalWorkSize[0] = WIDTH;
    globalWorkSize[1] = HEIGHT;

    size_t* newGS = (size_t *) malloc(2 * sizeof(size_t));
    size_t* newLS = (size_t *) malloc(2 * sizeof(size_t));

    getNewSizes(globalWorkSize, localWorkSize,
                newGS, newLS, kernelName.c_str(), 2);

    localWorkSize[0] = newLS[0];
    localWorkSize[1] = newLS[1];
    globalWorkSize[0] = newGS[0];
    globalWorkSize[1] = newGS[1]; 
  }
}

//-----------------------------------------------------------------------------
void freeMemory() {
  delete [] hostA;
  delete [] hostB;
  delete A;
  delete B;
  delete kernel;
  delete platform;
  delete width;
  delete height;
  delete [] localWorkSize;
  delete [] globalWorkSize;
}

//-----------------------------------------------------------------------------
void hostMemoryAlloc() {
  size_t* newGS = (size_t *) malloc(2 * sizeof(size_t));
  size_t* newLS = (size_t *) malloc(2 * sizeof(size_t));
  size_t* oldGS = (size_t *) malloc(2 * sizeof(size_t));
  size_t* oldLS = (size_t *) malloc(2 * sizeof(size_t));
  localWorkSize = new size_t[2];
  globalWorkSize = new size_t[2];

  oldGS[0] = WIDTH; 
  oldGS[1] = HEIGHT;
  oldLS[0] = blockSizeX;  
  oldLS[1] = blockSizeY;

  getNewSizes(oldGS, oldLS, newGS, newLS, kernelName.c_str(), 2);

  localWorkSize[0] = newLS[0];
  localWorkSize[1] = newLS[1];
  globalWorkSize[0] = newGS[0];
  globalWorkSize[1] = newGS[1]; 

  hostA = new float [globalWorkSize[0] * globalWorkSize[1]];
  hostB = new float [globalWorkSize[0] * globalWorkSize[1]];

  memset(hostA, 0, globalWorkSize[0] * globalWorkSize[1]);
  memset(hostB, 0, globalWorkSize[0] * globalWorkSize[1]);

  int counter = 1;
  for(unsigned int row = 0; row < globalWorkSize[1]; row++) {
    for(unsigned int column = 0; column < globalWorkSize[0]; column++) {
      hostA[column + globalWorkSize[0] * row] = random(-ELEMENT_LIMIT, ELEMENT_LIMIT);
//      hostA[column + WIDTH * row] = counter;
      counter++;
    }
  }
}

//-----------------------------------------------------------------------------
void deviceMemoryAlloc() {
  A = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                 globalWorkSize[0] * globalWorkSize[1] * sizeof(float), NULL);
  B = new Buffer(*(platform->getContext()), Buffer::WriteOnly,
                 globalWorkSize[0] * globalWorkSize[1] * sizeof(float), NULL);
  width = new cl_uint(globalWorkSize[0]);
  height = new cl_uint(globalWorkSize[1]);
}

//-----------------------------------------------------------------------------
void enqueWriteCommands(Queue& queue) {
  queue.writeBuffer(*A, globalWorkSize[0] * globalWorkSize[1] * sizeof(float), (void*) hostA);
  queue.finish();
}

//-----------------------------------------------------------------------------
void enqueReadCommands(Queue& queue) {
  queue.readBuffer(*B, globalWorkSize[0] * globalWorkSize[1] * sizeof(float), (void*) hostB);
  queue.finish();
}

//-----------------------------------------------------------------------------
void setKernelArguments() {
  if(kernelName == MT)
    setKernelArgumentsClassic();
  else if(kernelName == MT_LOCAL)
    setKernelArgumentsWithLocal();
}

//-----------------------------------------------------------------------------
void setKernelArgumentsWithLocal() {
  kernel->setArgument(0, *B);
  kernel->setArgument(1, *A);
  kernel->setArgument(2, sizeof(cl_int), (void*) width);
  kernel->setArgument(3, sizeof(cl_int), (void*) height);
  std::cout << "workgroup size: "<<localWorkSize[0] << " " << localWorkSize[1] << "\n";

  kernel->setArgument(4, (localWorkSize[0]) * localWorkSize[1] * sizeof(float), NULL);
}

//-----------------------------------------------------------------------------
void setKernelArgumentsClassic() {
  kernel->setArgument(0, *B);
  kernel->setArgument(1, *A);
  kernel->setArgument(2, sizeof(cl_int), (void*) width);
  kernel->setArgument(3, sizeof(cl_int), (void*) height);
}

//-----------------------------------------------------------------------------
void verifyResults() {
  for(unsigned int row = 0; row < globalWorkSize[1]; ++row) {
    for(unsigned int column = 0; column < globalWorkSize[0]; ++column) {
      if(abs(hostA[row * globalWorkSize[0] + column] - hostB[column * globalWorkSize[1] + row]) >= 0.01) {
        std::cout << "Error in the computation\n";
        exit(1);
      }
    }
  }

//  printMatrix(hostA, WIDTH, HEIGHT);
//  std::cout << "\n";
//  printMatrix(hostB, HEIGHT, WIDTH);
//  std::cout << "\n";
}

//-----------------------------------------------------------------------------
float random(float rand_min, float rand_max) {
  float result =(float)rand()/(float)RAND_MAX;
  return ((1.0 - result) * rand_min + result *rand_max);
}

//-----------------------------------------------------------------------------
void printVector(const float* printVector, unsigned int size) {
  for (unsigned int index = 0; index < size; ++index) {
    std::cout << printVector[index] << " ";
  }
  std::cout << std::endl;
}

//-----------------------------------------------------------------------------
void printMatrix(const float* matrix, size_t width, size_t height) {
  for (unsigned int row = 0; row < height; ++row) {
    for (unsigned int column = 0; column < width; ++column) {
      std::cout << std::setprecision(3) << matrix[row * width + column] << " ";
    }
    std::cout << std::endl;
  }
}
