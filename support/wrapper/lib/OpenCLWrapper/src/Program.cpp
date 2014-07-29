#include "OpenCLWrapper/Program.h"
#include "OpenCLWrapper/SystemConfig.h"


#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <boost/filesystem.hpp>

#include "Utils/FileUtils.h"
#include <iostream>
#include "OpenCLWrapper/Context.h"
#include "OpenCLWrapper/Device.h"
#include "OpenCLWrapper/Kernel.h"

#include "NvidiaBuildLogParser.h"
#include "AmdISAParser.h"
#include "Utils.h"

#include <algorithm>
#include <iterator>

using namespace boost::filesystem;

const char* PTX_EXTENSION = ".ptx";
const char* OCL_EXTENSION = ".cl";
const char* NVIDIA_VERBOSE = "-cl-nv-verbose";
const char* NVIDIA_CACHE_DIRECTORY = ".nv/ComputeCache";
const char* AMD_TEMP_FILES_OPTION = " -save-temps ";


//------------------------------------------------------------------------------
void checkFileExtension(const path& file, 
                        const std::string& expectedExtension) {
  if(!is_regular_file(file))
    throw std::runtime_error(file.string() + " is not a regular file");
  if(!file.has_extension())
    throw std::runtime_error("No file extension, expected: " + 
                             expectedExtension);
  path extension = file.extension();
  if(extension != expectedExtension)
    throw std::runtime_error("Wrong file extension: " + extension.string() + 
                             ", expected: " + expectedExtension);
}

// Constructors and Destructors.
//------------------------------------------------------------------------------
BinaryFile::BinaryFile(const char* binaryFileString) : path(binaryFileString) {
  checkFileExtension(*this, PTX_EXTENSION);
}

SourceFile::SourceFile(const char* sourceFileString) : path(sourceFileString) {
  checkFileExtension(*this, OCL_EXTENSION);
}

//==============================================================================

// Constructors and Destructors.
//------------------------------------------------------------------------------
Program::Program(Context* context, const SourceFile& sourceFile) : 
                 context(context) {
  createFromSource(sourceFile); 
}

//------------------------------------------------------------------------------
Program::Program(Context* context, const Device& device, 
                 const BinaryFile& binaryFile) : 
                 context(context) {
  createFromBinary(device, binaryFile); 
}

//------------------------------------------------------------------------------
Program::~Program() throw () {
  clReleaseProgram(program);
}

//------------------------------------------------------------------------------
void Program::createFromSource(const SourceFile& sourceFile) {
  cl_int errorCode;

  /*//////////////////////////////////
     Perform first LLVM transformation 
  *///////////////////////////////////

  /*
     Add trace size buffer as a kernel paramter

  */

  std::string sourceString = readFile(sourceFile);
  std::string trace_arg = "__global unsigned long long* trace,";
  
  size_t pos = sourceString.find("__kernel");
  while(pos != std::string::npos){
    pos = sourceString.find("(",pos);
    sourceString.insert(pos+1,trace_arg);
    pos = sourceString.find("__kernel",pos);
  }

  /*
    Create .cl file for the first transformation to instrument
  */
  FILE* fp=fopen("/tmp/tmp_size.cl", "w");
  if (fp == NULL) perror ("Error opening file");
  fputs(sourceString.c_str(),fp);
  fclose(fp);

  /*
     Execute first transformation by running scripts
  */  

  const char* passPath = getenv("VIS_PASSES");  
  if(passPath == NULL){
    std::cout <<"Error: VIS_PASSES enivronmental Variable not set\n";
    exit(0);
  } 



  std::string opt_script(SCRIPT_PATH);
  opt_script.append("/size_trace.sh /tmp/tmp_size.cl ");
  opt_script.append(SCRIPT_PATH);

  system(opt_script.c_str());
  system(std::string(SCRIPT_PATH).append("/vi_size_script.sh").c_str());
  

  /*
     Create cl_Program object from result of first optimization
  */
  SourceFile optFile = "/tmp/output.cl";
  sourceString = readFile(optFile);
  const char* sourceData = sourceString.data();


  size_t sourceSize = sourceString.length();
  length_program = clCreateProgramWithSource(context->getId(), 1, 
                                      (const char**) &sourceData,
                                      &sourceSize, &errorCode);

  verifyOutputCode(errorCode, "Error creating the length program");
  system("rm /tmp/tmp_size.cl");

  /*///////////////////////////////////
     Perform second LLVM transformation 
  *////////////////////////////////////

  /*
    Add parameters to kernel for added buffers needed for the second transformation
  */
  sourceString = readFile(sourceFile);
  trace_arg = "__global unsigned long long* trace,__global unsigned long long* ids, __global unsigned long long* loop_ctr,";
  
  pos = sourceString.find("__kernel");
  while(pos != std::string::npos){
    pos = sourceString.find("(",pos);
    sourceString.insert(pos+1,trace_arg);
    pos = sourceString.find("__kernel",pos);
  }


  /*
   Create .cl file from kernel to be instrumented by our second transformation
  */
  fp=fopen("/tmp/tmp_mem.cl", "w");
  if (fp == NULL) perror ("Error opening file");
  fputs(sourceString.c_str(),fp);
  fclose(fp);


  /*
    Execute second LLVM instrumentation by calling scripts
  */
  opt_script = std::string(SCRIPT_PATH).append("/trace_script.sh /tmp/tmp_mem.cl ");
  opt_script.append(SCRIPT_PATH);

  system(opt_script.c_str());
  system(std::string(SCRIPT_PATH).append("/vi_trace_script.sh").c_str());

  
  /*
    Create cl_Program object from trace instrumentation output
  */
  sourceString = readFile(optFile);
  sourceData = sourceString.data();
  sourceSize = sourceString.length();

  program = clCreateProgramWithSource(context->getId(), 1, 
                                      (const char**) &sourceData,
                                      &sourceSize, &errorCode);
  verifyOutputCode(errorCode, "Error creating the program with source");
  createdFromSource = true;

  system("rm /tmp/tmp_mem.cl");
  system("rm /tmp/output.cl");

}

//------------------------------------------------------------------------------
void Program::createFromBinary(const Device& device, 
                               const BinaryFile& binaryFile) {
  std::string sourceString = readFile(binaryFile);

  cl_device_id deviceId = device.getId();
  size_t binarySize = sourceString.length() + 1;
  const char* sourceData = sourceString.data();

  cl_int binaryStatus;
  cl_int errorCode;
  program = clCreateProgramWithBinary(context->getId(), 1,
                                      &deviceId,
                                      &binarySize,
                                      (const unsigned char**) &sourceData,
                                      &binaryStatus,
                                      &errorCode);
  verifyOutputCode(errorCode, "Error creating the program from binary");
  verifyOutputCode(binaryStatus, "Invalid binary");
  createdFromSource = false;
}

//------------------------------------------------------------------------------

/*
   Get cl_program object associated with output from trace instrumentation
*/
cl_program Program::getId() const {
  return program;
}

/*
   Get cl_program object associated with output from first instrumentation
*/
cl_program Program::getLenId() const {
  return length_program;
}


cl_context Program::getContext() const{
  return (context->getId());
}


//------------------------------------------------------------------------------
Kernel* Program::createKernel(const char* name) {
  return new Kernel(*this, name);
}

//------------------------------------------------------------------------------
bool Program::build(const Device& device) const {
  return build(device.getId(), "");
}

//------------------------------------------------------------------------------
bool Program::build(const Device& device, const std::string& options) const {
  return build(device.getId(), options.c_str());
}

//------------------------------------------------------------------------------
bool Program::build(cl_device_id deviceId, const char* options) const {
  cl_int errorCode = clBuildProgram(program, 1, &deviceId, options, NULL, NULL);
  errorCode = clBuildProgram(length_program, 1, &deviceId, options, NULL, NULL);

  return !isError(errorCode);
}

// Get build Log.
//------------------------------------------------------------------------------
std::string Program::getBuildLog(const Device& device) const {
  size_t buildLogSize = getBuildLogSize(device);
  return getBuildLogText(device, buildLogSize);
}

//------------------------------------------------------------------------------
size_t Program::getBuildLogSize(const Device& device) const {
  size_t buildLogSize;
  cl_int errorCode = clGetProgramBuildInfo(program, device.getId(), 
                                           CL_PROGRAM_BUILD_LOG, 0, NULL,
                                           &buildLogSize);
  verifyOutputCode(errorCode, "Error querying the build log size");
  return buildLogSize;
}

//------------------------------------------------------------------------------
std::string Program::getBuildLogText(const Device& device, 
                                     size_t buildLogSize) const {
  char* buildLog = new char[buildLogSize+1];
  cl_int errorCode = clGetProgramBuildInfo(program, device.getId(),
                                           CL_PROGRAM_BUILD_LOG, buildLogSize,
                                           buildLog, NULL);
  verifyOutputCode(errorCode, "Error querying the build log");
  std::string buildLogString(buildLog);
  delete [] buildLog;
  return buildLogString;
}

// AMD specific.
//------------------------------------------------------------------------------
std::pair<unsigned int, unsigned int> Program::getAMDRegistersNumber(
                                      const Device& device,
                                      const std::string& kernelName,
                                      std::string options) const {
  options += AMD_TEMP_FILES_OPTION;
  build(device.getId(), AMD_TEMP_FILES_OPTION);
  return readAMDRegistersNumber(device, kernelName); 
}

// NVIDIA specific.
//------------------------------------------------------------------------------
unsigned int Program::getNvidiaRegistersNumber(
                      const Device& device,
                      const std::string& kernelName,
                      std::string options) const {
  forceRecompilation();
  options += " ";
  options += NVIDIA_VERBOSE; 
  build(device.getId(), options);
  std::string buildLog = getBuildLog(device);
  return getRegistersNumberFromBuildLog(buildLog, kernelName);
}

// NVIDIA specific.
//------------------------------------------------------------------------------
unsigned int Program::getNvidiaInstructionNumber(
                      const Device& device,
                      const std::string& kernelName,
                      std::string options) const {
  forceRecompilation();
  options += " ";
  options += NVIDIA_VERBOSE;
  build(device.getId(), options);
  std::string binary = getBinary(device);
  return getInstructionNumberFromBinary(binary, kernelName);
}

//------------------------------------------------------------------------------
void Program::forceRecompilation() const {
  // Delete the cache. It is in /home/USERNAME/.nv/ComputeCache
  path nvidiaCacheDirectory = getHomeDirectory();
  nvidiaCacheDirectory /= NVIDIA_CACHE_DIRECTORY;
  remove_all(nvidiaCacheDirectory);
}

// Get program source.
//------------------------------------------------------------------------------
std::string Program::getSourceCode() const {
  if(!createdFromSource)
    throw std::runtime_error("Cannot query the source code when program \
                              created from binary");
  size_t sourceCodeSize = getSourceCodeSize();
  return getSourceCodeText(sourceCodeSize);
}

//------------------------------------------------------------------------------
size_t Program::getSourceCodeSize() const {
  size_t sourceCodeSize;
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, 
                                      &sourceCodeSize);
  verifyOutputCode(errorCode, "Error querying the source code size");
  return sourceCodeSize;
}

//------------------------------------------------------------------------------
std::string Program::getSourceCodeText(size_t sourceCodeSize) const {
  char* sourceCode = new char[sourceCodeSize];
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_SOURCE, 
                                      sourceCodeSize, sourceCode, NULL);
  verifyOutputCode(errorCode, "Error querying the source code");
  return std::string(sourceCode);
}

// Get program binary.
//------------------------------------------------------------------------------
std::string Program::getBinary(const Device& device) const {
  cl_uint devicesNumber = queryDevicesNumber();
  std::vector<size_t> binariesSize = getBinariesSize(devicesNumber);
  unsigned int deviceIndex = getDeviceIndex(device);
  return getBinaryText(deviceIndex, binariesSize);
}

//------------------------------------------------------------------------------
unsigned int Program::getDeviceIndex(const Device& device) const {
  std::vector<cl_device_id> devices = queryDevices();
  std::vector<cl_device_id>::iterator deviceIter = std::find(devices.begin(), 
                                                             devices.end(), 
                                                             device.getId());
  if(deviceIter == devices.end())
    std::runtime_error("Requested binary for device not associated with the \
                        current program");
  unsigned int deviceIndex = std::distance(devices.begin(), deviceIter);
  return deviceIndex;
}

//------------------------------------------------------------------------------
std::vector<cl_device_id> Program::queryDevices() const {
  cl_uint devicesNumber = queryDevicesNumber(); 
  cl_device_id* devicesData = new cl_device_id[devicesNumber]; 
  queryDevicesId(devicesData, devicesNumber);
  std::vector<cl_device_id> result(devicesData, devicesData + devicesNumber);
  delete [] devicesData;
  return result;
}

//------------------------------------------------------------------------------
unsigned int Program::queryDevicesNumber() const {
  cl_uint devicesNumber;
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, 
                                      sizeof(cl_uint), &devicesNumber, NULL);
  verifyOutputCode(errorCode, "Error querying the number of devices \
                               associated to the progam");
  return devicesNumber;
}

//------------------------------------------------------------------------------
void Program::queryDevicesId(cl_device_id* devicesData, 
                             cl_uint devicesNumber) const {
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                                      sizeof(cl_device_id) * devicesNumber,
                                      devicesData, NULL);
  verifyOutputCode(errorCode, "Error querying the devices associated \
                               to the program");
}

//------------------------------------------------------------------------------
std::vector<size_t> Program::getBinariesSize(unsigned int devicesNumber) const {
  size_t* binarySizes = new size_t[devicesNumber];
  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                      devicesNumber * sizeof(size_t),
                                      binarySizes, NULL);
  verifyOutputCode(errorCode, "Error getting the program binary size: ");
  std::vector<size_t> result(binarySizes, binarySizes + devicesNumber);
  delete [] binarySizes;
  return result;
}

//------------------------------------------------------------------------------
// FIXME. Split in multiple functions.
std::string Program::getBinaryText(
                     unsigned int deviceIndex,
                     const std::vector<size_t>& binariesSize) const {
  unsigned int binariesNumber = binariesSize.size();
  char** binaryPrograms = new char* [binariesNumber];
  for (unsigned int index = 0; index < binariesNumber; ++index) { 
    binaryPrograms[index] = NULL;
  }
  binaryPrograms[deviceIndex] = new char[binariesSize[deviceIndex] + 1];

  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                      0, binaryPrograms, NULL);
  verifyOutputCode(errorCode, "Error getting the program binary: ");

  binaryPrograms[deviceIndex][binariesSize[deviceIndex]] = '\0';
  std::string binaryProgramString(binaryPrograms[deviceIndex]);
  delete [] binaryPrograms[deviceIndex];
  delete [] binaryPrograms;
  return binaryProgramString;
}

// Get kernels list. Only in OCL 1.2
//------------------------------------------------------------------------------
//std::string Program::getKernelsList() const {
//  size_t kernelsNumber = getKernelsNumber();
//  size_t listSize = getKernelsListSize();
//  char* kernelsList = new char[listSize];
//  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES,
//                                      listSize * sizeof(char),
//                                      &kernelsList, NULL); 
//  std::string result(kernelsList);
//  delete [] kernelsList;
//  return result;
//}
//
//size_t Program::getKernelsNumber() const {
//  size_t kernelsNumber;
//  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS,
//                                      sizeof(size_t),
//                                      &kernelsNumber, NULL);
//  verifyOutputCode(errorCode, "Error getting the kernels number: ");
//  return kernelsNumber;
//}
//
//size_t Program::getKernelsListSize() const {
//  size_t listSize;
//  cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES,
//                                      sizeof(size_t),
//                                      NULL, &listSize);
//  verifyOutputCode(errorCode, "Error getting the kernels list size: ");
//  return listSize;
//}
