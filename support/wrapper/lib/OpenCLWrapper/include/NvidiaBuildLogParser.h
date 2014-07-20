#include <string>
#include <vector>

unsigned int getRegistersNumberFromBuildLog(const std::string& buildLog,
                                            const std::string& kernelName);
std::vector<std::string> splitIntoLines(const std::string& inputString);
void trimLines(std::vector<std::string>& lines);
unsigned int findKernelStart(const std::vector<std::string>& lines,
                             const std::string& kernelName);
std::string getKernelStartingLine(const std::string& kernelName);
int getRegistersNumber(const std::string& inputString);
std::vector<int> getNumbersInString(const std::string& inputString);

unsigned int getInstructionNumberFromBinary(const std::string& buildLog,
                                            const std::string& kernelName);
unsigned int countBlanckLines(const std::vector<std::string>& lines,
                              unsigned int start, unsigned int end);
unsigned int findBracket(const std::vector<std::string>& lines,
                         unsigned int startPosition,
                         char bracket);
unsigned int getKernelSignature(const std::vector<std::string>& lines,
                                const std::string& kernelName);
