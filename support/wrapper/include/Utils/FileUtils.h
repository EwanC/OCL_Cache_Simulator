#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>

// TODO use forward declaration.
#include <boost/filesystem.hpp>

using namespace boost::filesystem;

std::string readFile(const path& filePath);

inline std::string readWholeFileStream(std::ifstream& fileStream);

inline void verifyFileExists(const path& file);

inline void verifyFileStreamOpen(const path& filePath,
                                 const std::ifstream& fileStream);

path getHomeDirectory();

#endif
