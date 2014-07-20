#include "NvidiaBuildLogParser.h"

#include <stdexcept>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#define KERNEL_DELIMETER_PREFIX "ptxas info    : Function properties for "
#define REGISTERS_NUMBER_LINE_OFFSET 2
#define REGISTERS_NUMBER_POSITION 3

#define ENTRY_POINT ".entry "

// Template.
//ptxas info    : Compiling entry function 'x' for 'sm_20'
//ptxas info    : Function properties for x
// 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
//ptxas info    : Used 2 registers, 32 bytes cmem[0]
//ptxas info    : Compiling entry function 'y' for 'sm_20'
//ptxas info    : Function properties for y
// 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
//ptxas info    : Used 2 registers, 32 bytes cmem[0]

//------------------------------------------------------------------------------
unsigned int getRegistersNumberFromBuildLog(const std::string& buildLog,
                                            const std::string& kernelName) {
  std::vector<std::string> lines = splitIntoLines(buildLog); 
  unsigned int kernelStart = findKernelStart(lines, kernelName);
  unsigned int registerLineNumber = kernelStart + REGISTERS_NUMBER_LINE_OFFSET;
  return getRegistersNumber(lines.at(registerLineNumber));
}

//------------------------------------------------------------------------------
std::vector<std::string> splitIntoLines(const std::string& inputString) {
  std::vector<std::string> result;
  boost::split(result, inputString, boost::is_any_of("\n"));
  trimLines(result);
  return result;
}

//------------------------------------------------------------------------------
void trimLines(std::vector<std::string>& lines) {
  std::for_each(lines.begin(), lines.end(), 
                boost::bind(&boost::trim<std::string>, _1, std::locale()));
}

//------------------------------------------------------------------------------
unsigned int findKernelStart(const std::vector<std::string>& lines,
                             const std::string& kernelName) {
  for (unsigned int index = 0; index < lines.size(); ++index) {
    if(lines[index] == getKernelStartingLine(kernelName))
      return index;
  }
  throw std::runtime_error("No kernel named " + kernelName + 
                           " in the build log");
}

//------------------------------------------------------------------------------
std::string getKernelStartingLine(const std::string& kernelName) {
  return KERNEL_DELIMETER_PREFIX + kernelName;
}

//------------------------------------------------------------------------------
int getRegistersNumber(const std::string& inputString) {
  boost::tokenizer<> tokenizer(inputString);
  boost::tokenizer<>::iterator iterator = tokenizer.begin();
  std::advance(iterator, REGISTERS_NUMBER_POSITION);
  return boost::lexical_cast<int>(*iterator);
}

//------------------------------------------------------------------------------
std::vector<int> getNumbersInString(const std::string& inputString) { 
  std::vector<int> result;
  boost::tokenizer<> tokenizer(inputString);
  for(boost::tokenizer<>::iterator iterator = tokenizer.begin(), 
                                   end = tokenizer.end(); 
                                   iterator != end;
                                   ++iterator) {
    try { 
      result.push_back(boost::lexical_cast<int>(*iterator));
    } catch (boost::bad_lexical_cast lexicalError) {
      continue;
    }
  } 
  return result; 
}

// Get instruction count.
//------------------------------------------------------------------------------
unsigned int findKernelSignature(const std::vector<std::string>& lines,
                                const std::string& kernelName) {
  for (unsigned int index = 0; index < lines.size(); ++index) {
    if(boost::starts_with(lines[index], ENTRY_POINT + kernelName)) 
      return index;
  }
  throw std::runtime_error("No kernel named " + kernelName + " in binary");
}

//------------------------------------------------------------------------------
unsigned int findFirstBracket(const std::vector<std::string>& lines, 
                              unsigned int startPosition,
                              char bracket) {
  for (unsigned int index = startPosition; index < lines.size(); ++index) 
    if(boost::starts_with(lines[index], 
       boost::lexical_cast<std::string>(bracket)))
      return index;
  std::string lineString = boost::lexical_cast<std::string>(startPosition);
  throw std::runtime_error("No '{' after line " + lineString + " in binary");
} 

//------------------------------------------------------------------------------
unsigned int findLastBracket(const std::vector<std::string>& lines,
                             char bracket) {
  for (unsigned int index = lines.size() - 1; index > 0; --index)
    if(boost::starts_with(lines[index],
       boost::lexical_cast<std::string>(bracket)))
      return index;
  throw std::runtime_error("Last bracket not found");
}

//------------------------------------------------------------------------------
unsigned int countBlankLines(const std::vector<std::string>& lines,
                             unsigned int start, unsigned int end) {
  unsigned int number = 0;
  for (unsigned int index = start; index < end; ++index) {
    if(lines[index] == "") 
      ++number;
  }
  return number;
}

//------------------------------------------------------------------------------
unsigned int countComments(const std::vector<std::string>& lines,
                           unsigned int start, unsigned int end) {
  unsigned int number = 0;
  for (unsigned int index = start; index < end; ++index) {
    if(boost::starts_with(lines[index], "//"))
      ++number;
  }
  return number;
}

//------------------------------------------------------------------------------
unsigned int countLabels(const std::vector<std::string>& lines, 
                         unsigned int start, unsigned int end) {
  unsigned int number = 0;
  for (unsigned int index = start; index < end; ++index) {
    if(boost::starts_with(lines[index], "BB"))
      ++number;
  }
  return number;
}

//------------------------------------------------------------------------------
unsigned int countBrackets(const std::vector<std::string>& lines,
                           unsigned int start, unsigned int end) {
  unsigned int number = 0;
  for (unsigned int index = start; index < end; ++index) {
    if(boost::starts_with(lines[index], "}") ||
       boost::starts_with(lines[index], "{"))
      ++number;
  }
  return number;
}

//------------------------------------------------------------------------------
unsigned int getInstructionNumberFromBinary(const std::string& binary,
                                            const std::string& kernelName) {
  std::vector<std::string> lines = splitIntoLines(binary);
  trimLines(lines); 
  unsigned int kernelStart = findKernelSignature(lines, kernelName);
  unsigned int firstBracketPosition = findFirstBracket(lines, kernelStart, '{');
  unsigned int lastBracketPosition = findLastBracket(lines, '}');
  unsigned int blankLines = countBlankLines(lines, firstBracketPosition, 
                                             lastBracketPosition);
  unsigned int comments = countComments(lines, firstBracketPosition, 
                                        lastBracketPosition);
  unsigned int brackets = countBrackets(lines, firstBracketPosition, lastBracketPosition);
  unsigned int labels = countLabels(lines, firstBracketPosition, lastBracketPosition);

  return lastBracketPosition - firstBracketPosition - blankLines -
         comments - brackets - labels;
}
