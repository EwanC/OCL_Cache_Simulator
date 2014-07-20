#include "AmdISAParser.h"

#include <sstream>
#include <stdexcept>

#include <boost/filesystem.hpp>

#include "Utils/FileUtils.h"

namespace fs = boost::filesystem;

// Template for New.
// NumVgprs             = 3;
// NumSgprs             = 24;

// Template for Old.
// NUM_GPRS     = 2

const char* NUM_VECTOR_GPRS = "NumVgprs";
const char* NUM_SCALAR_GPRS = "NumSgprs";
const char* NUM_GPRS_OLD = "NUM_GPRS";
const char* ISA_EXTENSION = ".isa";

//------------------------------------------------------------------------------
std::pair<unsigned int, unsigned int> readAMDRegistersNumber(
                                      const Device& device,
                                      const std::string& kernelName) {
  if(device.getName().find_first_of("Cypress") != std::string::npos)
    return readOldAMDRegistersNumber(device, kernelName);
  return readNewAMDRegistersNumber(device, kernelName);
}

//------------------------------------------------------------------------------
std::pair<unsigned int, unsigned int> readOldAMDRegistersNumber(
                                      const Device& device,
                                      const std::string& kernelName) {
  fs::path currentDirectory(".");
  checkDirectory(currentDirectory);

  fs::path isaFile = getIsaFile(currentDirectory, kernelName);
  std::string isaContent = readFile(isaFile);

  // Find NUM_GPRS.
  unsigned int vectorRegistersNumber = getOldRegistersNumber(isaContent,
                                                            NUM_GPRS_OLD);

  fs::path abs = fs::absolute(currentDirectory);

  // Clean directory.
  std::string removeString = "cd " + abs.string() + " && rm *.isa *.il *.cl";
  system(removeString.c_str());

  return std::make_pair(vectorRegistersNumber, 0);
}

//------------------------------------------------------------------------------
// Return number of vector register and number of scalar registers.
std::pair<unsigned int, unsigned int> readNewAMDRegistersNumber(
                                      const Device& device, 
                                      const std::string& kernelName) {
  fs::path currentDirectory(".");
  checkDirectory(currentDirectory);

  fs::path isaFile = getIsaFile(currentDirectory, kernelName);
  std::string isaContent = readFile(isaFile);     
  
  // Find NumVgprs.
  unsigned int vectorRegistersNumber = getRegistersNumber(isaContent, 
                                                          NUM_VECTOR_GPRS);
  // Fine NumSgprs.
  unsigned int scalarRegistersNumber = getRegistersNumber(isaContent,
                                                          NUM_SCALAR_GPRS);
  
  fs::path abs = fs::absolute(currentDirectory);

  // Clean directory.
  std::string removeString = "cd " + abs.string() + " && rm *.isa *.il *.cl";
  //system(removeString.c_str());

  return std::make_pair(vectorRegistersNumber, scalarRegistersNumber);
}

//------------------------------------------------------------------------------
void checkDirectory(const fs::path& directory) {
  if (!fs::exists(directory)) 
    throw std::runtime_error(directory.string() + " does not exist");
  if (!fs::is_directory(directory)) 
    throw std::runtime_error(directory.string() + " is not a directory");
}

//------------------------------------------------------------------------------
fs::path getIsaFile(const fs::path& directory, const std::string& kernelName) {
  for( fs::directory_iterator I(directory), E; I != E ; ++I) {
    fs::path currentFile = I->path(); 
    if(currentFile.extension().string() == ISA_EXTENSION) 
      if(currentFile.string().find(kernelName) != std::string::npos) 
        return I->path();
  } 
  throw std::runtime_error("No " + std::string(ISA_EXTENSION) +
                           " file in " + directory.string());
}

//------------------------------------------------------------------------------
// If the nuber is not found return 0.
unsigned int getRegistersNumber(const std::string& isaContent, 
                                const std::string registerTypeName) {
  size_t idPosition = isaContent.find(registerTypeName);
  if(idPosition == std::string::npos)
    return 0;
  size_t equalsPosition = isaContent.find_first_of("=", idPosition);
  if(equalsPosition == std::string::npos)
    return 0;
  size_t lineDelimeter = isaContent.find_first_of(";", idPosition);  
  if(lineDelimeter == std::string::npos)
    return 0;

  std::string registerNumberString = isaContent.substr(
                                     equalsPosition + 2, 
                                     lineDelimeter - equalsPosition); 
  int registerNumber;
  std::istringstream (registerNumberString) >> registerNumber;

  return registerNumber;
}

//------------------------------------------------------------------------------
// If the nuber is not found return 0.
unsigned int getOldRegistersNumber(const std::string& isaContent,
                                   const std::string registerTypeName) {
  size_t idPosition = isaContent.find(registerTypeName);
  if(idPosition == std::string::npos)
    return 0;
  size_t equalsPosition = isaContent.find_first_of("=", idPosition);
  if(equalsPosition == std::string::npos)
    return 0;
  size_t lineDelimeter = isaContent.find_first_of("\n", idPosition);
  if(lineDelimeter == std::string::npos)
    return 0;

  std::string registerNumberString = isaContent.substr(
                                     equalsPosition + 2,
                                     lineDelimeter - equalsPosition);
  int registerNumber;
  std::istringstream (registerNumberString) >> registerNumber;

  return registerNumber;
}
