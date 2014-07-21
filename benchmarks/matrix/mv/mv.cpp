#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>

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
#define ELEMENT_LIMIT 10

#define HEIGHT 1024 // 5  //128 *128
#define WIDTH 256  // 5 //128 * 8
#define BLOCK_SIZE 16

#define UNCOALESCED0 "MatVecMulUncoalesced0"
#define UNCOALESCED1 "MatVecMulUncoalesced1"
#define COALESCED0 "MatVecMulCoalesced0"
#define COALESCED1 "MatVecMulCoalesced1"

#define KERNEL_FILE_NAME "mv.cl"
//-----------------------------------------------------------------------------
unsigned long computeExecutionTime(cl_event& event);
void initialization(int argc, char** argv);
void hostMemoryAlloc();
void deviceMemoryAlloc();
void setKernelArguments();
void readResult();
void enqueWriteCommands(Queue& queue);
void enqueReadCommands(Queue& queue);
void verifyResults();
float random(float rand_min, float rand_max);
void freeMemory(); 
void setSizes();

//-----------------------------------------------------------------------------
// Runtime components.
Platform* platform;
Kernel* kernel;

// Host data.
float* hostV;
float* hostM;
float* hostW;

// Device data.
Buffer* V;
Buffer* M;
Buffer* W;

cl_uint* height = NULL;
cl_uint* width = NULL;

size_t* localWorkSize = NULL;
size_t* globalWorkSize = NULL;

std::string kernelName = "";
unsigned int blockSize = 0;

int PLATFORM_ID = 0;
int DEVICE_ID = 0;

//-----------------------------------------------------------------------------
// Usage: ./mv --kernelName [MatVecMulUncoalesced0 | MatVecMulUncoalesced1 | 
//                           MatVecMulCoalesced0 | MatVecMulCoalesced1]
int main(int argc, char** argv) {
  initialization(argc, argv);
  platform = new Platform(PLATFORM_ID);
  Context* context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  std::cout << "Running "<<kernelName<< " on "<< device.getName() << "\n";
  hostMemoryAlloc();
  deviceMemoryAlloc();

  SourceFile kernelFile = KERNEL_DIRECTORY KERNEL_FILE_NAME;

  Program program(context,kernelFile);
  Queue queue(*context, device, Queue::EnableProfiling);
  enqueWriteCommands(queue);
  if(!program.build(device)) {
    std::cout << "Error building the program\n";
    std::cout << program.getBuildLog(device) << "\n";
    return 1;
  }
  kernel = program.createKernel(kernelName.c_str());
  //std::cout << program.getBinary(device) << "\n";

  setSizes();

  setKernelArguments();

  if(localWorkSize[0] < 0)
    localWorkSize = NULL;

  queue.run(*kernel, 1, 0, globalWorkSize, localWorkSize);
  queue.finish();

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
  blockSize = BLOCK_SIZE;
  getPlatformDevice(&PLATFORM_ID, &DEVICE_ID);
}

//-----------------------------------------------------------------------------
void setSizes() {
    localWorkSize = new size_t[1];
    globalWorkSize = new size_t[1];

    localWorkSize[0] = blockSize;
    globalWorkSize[0] = HEIGHT;
}

//-----------------------------------------------------------------------------
void freeMemory() {
  delete [] hostM;
  delete [] hostV;
  delete [] hostW;
  delete W;
  delete V;
  delete M;
  delete kernel;
  delete platform;
  delete width;
  delete height;
  delete [] localWorkSize;
  delete [] globalWorkSize;
}

//-----------------------------------------------------------------------------
void hostMemoryAlloc() {
  size_t* newGS = (size_t *) malloc(1 * sizeof(size_t));
  size_t* newLS = (size_t *) malloc(1 * sizeof(size_t));
  size_t* oldGS = (size_t *) malloc(1 * sizeof(size_t));
  size_t* oldLS = (size_t *) malloc(1 * sizeof(size_t));

  localWorkSize = new size_t[1];
  globalWorkSize = new size_t[1];

  oldGS[0] = HEIGHT;
  oldLS[0] = blockSize;
  
  getNewSizes(oldGS, oldLS, newGS, newLS, kernelName.c_str(), 2);

  globalWorkSize[0] = newGS[0];
  localWorkSize[0] = newLS[0];

  hostM = new float [globalWorkSize[0] * WIDTH];
  hostV = new float [WIDTH];
  hostW = new float [globalWorkSize[0]];

  for (unsigned int row = 0; row < globalWorkSize[0]; row++) {
    for (unsigned int column = 0; column < WIDTH; column++) 
      hostM[column + WIDTH * row] = random(-ELEMENT_LIMIT, ELEMENT_LIMIT);
  }
    
  for (unsigned int column = 0; column < WIDTH; column++) {
    hostV[column] = random(-ELEMENT_LIMIT, ELEMENT_LIMIT);
  }
}

//-----------------------------------------------------------------------------
void deviceMemoryAlloc() {
  M = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                 globalWorkSize[0] * WIDTH * sizeof(float), NULL);
  V = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                 WIDTH * sizeof(float), NULL);
  W = new Buffer(*(platform->getContext()), Buffer::WriteOnly,
                 globalWorkSize[0] * sizeof(float), NULL);
  width = new cl_uint(WIDTH);
  height = new cl_uint(globalWorkSize[0]);
}

//-----------------------------------------------------------------------------
void enqueWriteCommands(Queue& queue) {
  queue.writeBuffer(*M, globalWorkSize[0] * WIDTH * sizeof(float), (void*) hostM);
  queue.writeBuffer(*V, WIDTH * sizeof(float), (void*) hostV);
  queue.finish();
}

//-----------------------------------------------------------------------------
void enqueReadCommands(Queue& queue) {
  queue.readBuffer(*W, globalWorkSize[0] * sizeof(float), (void*) hostW);
  queue.finish();
}

//-----------------------------------------------------------------------------
void setKernelArguments() {
  kernel->setArgument(0, *M);
  kernel->setArgument(1, *V);

  kernel->setArgument(2, sizeof(cl_int), (void*) width);
  kernel->setArgument(3, sizeof(cl_int), (void*) height);

  kernel->setArgument(4, *W);

  if(kernelName == COALESCED0 || kernelName == COALESCED1)
    kernel->setArgument(5, localWorkSize[0] * sizeof(float), NULL);
}

//-----------------------------------------------------------------------------
void verifyResults() {
  float* cpuHostW = new float [4];
  //#pragma omp parallel for
  for(unsigned int row = 0; row < 4; ++row) {
    for(unsigned int column = 0; column < WIDTH; ++column) {
      float result = 0.0f;
      for(unsigned int index = 0; index < WIDTH; ++index) 
        result += hostM[row * WIDTH + index] * hostV[index];
      if(abs(hostW[row] - result) >= 0.001f) {
        std::cout << "Error\n";
        exit(1);
      }
      cpuHostW[row] = result;
    }
  }
  std::cout<<"Ok!\n";

  delete [] cpuHostW;
}

//-----------------------------------------------------------------------------
float random(float rand_min, float rand_max) {
  float result =(float)rand()/(float)RAND_MAX;
  return ((1.0 - result) * rand_min + result *rand_max);
}
