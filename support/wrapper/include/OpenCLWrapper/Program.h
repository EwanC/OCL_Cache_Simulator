#ifndef PROGRAM_H
#define PROGRAM_H

#include <string>

#include <CL/cl.h>

#include <boost/filesystem.hpp>

class Context;
class Device;
class Kernel;

using namespace boost::filesystem;

class BinaryFile : public path {
public:
  BinaryFile(const char* binaryFileString);
};

class SourceFile : public path {
public:
  SourceFile(const char* sourceFileString);
};

class Program {
// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Program(Context* context, const SourceFile& sourceFile);
  Program(Context* context, const Device& device, const BinaryFile& binaryFile);
  ~Program() throw();

// Public methods.
//------------------------------------------------------------------------------
public:
  cl_program getId() const;       //Returns cl_program from trace instrumentation
  cl_program getLenId() const; //Returns cl_program from first instrumentation

  cl_context getContext() const;    
  Kernel* createKernel(const char* name);

  bool build(const Device& device) const;
  bool build(const Device& device, const std::string& options) const;

  std::string getBuildLog(const Device& device) const;

  // NVIDIA specific.
  unsigned int getNvidiaRegistersNumber(const Device& device, 
                                        const std::string& kernelName,
                                        std::string options) const;
  unsigned int getNvidiaInstructionNumber(const Device& device, 
                                          const std::string& kernelName,
                                          std::string options) const;
  // ATI specific.
  std::pair<unsigned int, unsigned int> getAMDRegistersNumber(
                                        const Device& device,
                                        const std::string& kernelName,
                                        std::string options) const;

  std::string getBinary(const Device& device) const;
  std::string getSourceCode() const;
  // OCL 1.2 only.
  //std::string getKernelsList() const;

// Private Fields.
//------------------------------------------------------------------------------
private:
  cl_program program;            //Program from second LLVM instrumentation
  cl_program length_program;     //Program from first LLVM instrumentation

  Context* context;
  bool createdFromSource;

// Private Methods.
//------------------------------------------------------------------------------
private:
  void createFromSource(const SourceFile& filePath);
  void createFromBinary(const Device& device, const BinaryFile& filePath);

  bool build(cl_device_id deviceId, const char* options) const;

  void forceRecompilation() const;

  size_t getBuildLogSize(const Device& device) const;
  std::string getBuildLogText(const Device& device, size_t buildLogSize) const;

  size_t getSourceCodeSize() const;
  std::string getSourceCodeText(size_t sourceCodeSize) const;

  // OCL 1.2 only.
  //size_t getKernelsNumber() const;
  //size_t getKernelsListSize() const;

  unsigned int getDeviceIndex(const Device& device) const;
  std::vector<cl_device_id> queryDevices() const;
  cl_uint queryDevicesNumber() const;
  void queryDevicesId(cl_device_id* devicesData, cl_uint devicesNumber) const;
  size_t getBinarySize(const Device& device) const;
  std::vector<size_t> getBinariesSize(unsigned int devicesNumber) const;
  std::string getBinaryText(unsigned int deviceIndex, 
                            const std::vector<size_t>& binariesSize) const;
};

#endif
