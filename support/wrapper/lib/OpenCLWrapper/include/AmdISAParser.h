#include <boost/filesystem.hpp>

#include <utility>

#include "OpenCLWrapper/Device.h"

namespace fs = boost::filesystem;

std::pair<unsigned int, unsigned int> readAMDRegistersNumber(
                                      const Device& device,
                                      const std::string& kernelName);
std::pair<unsigned int, unsigned int> readOldAMDRegistersNumber(
                                      const Device& device,
                                      const std::string& kernelName);
std::pair<unsigned int, unsigned int> readNewAMDRegistersNumber(
                                      const Device& device,
                                      const std::string& kernelName);

void checkDirectory(const fs::path& directory);
fs::path getIsaFile(const fs::path& directory, const std::string& kernelName);
unsigned int getRegistersNumber(const std::string& isaContent,
                                const std::string registerTypeName);
unsigned int getOldRegistersNumber(const std::string& isaContent,
                                   const std::string registerTypeName);
