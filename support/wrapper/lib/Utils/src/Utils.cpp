#include "Utils/FileUtils.h"

#include <fstream>
#include <stdexcept>
#include <sstream>

#include <boost/filesystem.hpp>

using namespace boost::filesystem;

#define HOME_ENV_VARIABLE "HOME"

// No copy constructor is called, I checked.
std::string readFile(const path& filePath) {
  verifyFileExists(filePath);
  std::ifstream fileStream(filePath.string().c_str());
  verifyFileStreamOpen(filePath, fileStream);
  return readWholeFileStream(fileStream);
}

inline std::string readWholeFileStream(std::ifstream& fileStream) {
  std::stringstream stringStream;
  stringStream << fileStream.rdbuf();
  fileStream.close();
  return stringStream.str();
}

inline void verifyFileExists(const path& file) {
  if(!exists(file))
    throw std::runtime_error("File: " + file.string() + " does not exist");
}

inline void verifyFileStreamOpen(const path& filePath,
                                 const std::ifstream& fileStream) {
  if(!fileStream.is_open())
    throw std::runtime_error("Error opening file: " + filePath.string());
}

path getHomeDirectory() {
  const char* homeString = getenv(HOME_ENV_VARIABLE);
  return path(homeString);
}
