#include <CL/cl.h>

#define PERCENT_DIFF_ERROR_THRESHOLD 0.0001

void verifyOutputCode(cl_int valueToCheck, const char* errorMessage);
bool isError(cl_int valueToCheck);

