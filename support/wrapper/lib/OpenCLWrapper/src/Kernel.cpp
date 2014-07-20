#include "OpenCLWrapper/Kernel.h"

#include "OpenCLWrapper/Buffer.h"
#include "OpenCLWrapper/Device.h"
#include "OpenCLWrapper/Program.h"

#include <stdlib.h>
#include <CL/cl.h>
#include <stdio.h>
#include "Utils.h"
#include <string.h>
#include <iostream>


void verifySetArgumentCode(int errorCode, unsigned int index);
//------------------------------------------------------------------------------
Kernel::Kernel(const Program& program, const char* name) {
  cl_int errorCode;
  first_run = true; 

  /*
    Create host memory for added buffer in prelimary transformation.
    Contains five entries, one for the length counter, initalized to zero, and 
    the other for loop warnings
  */
  
  h_trace1 = (unsigned long long *)calloc(TRACE1_SIZE, sizeof(long long));
  h_trace1[0] = 1;

  /*
    Create kernel from first instrumentation output cl program
  */

  length_kernel = clCreateKernel(program.getLenId(), name, &errorCode);  
  verifyOutputCode(errorCode, "Error creating the kernel");

  /*
    Create kernel from program generated with the result of the second instrumentation
  */

  kernel = clCreateKernel(program.getId(), name, &errorCode);  
  verifyOutputCode(errorCode, "Error creating the kernel");

  /*
      Create output file for kernel to dump memory trace.
      File name is the instrumented kernel name with suffix ".out"
  */
  
  fp=fopen(std::string(name).append(".out").c_str(), "w+");
  if (fp == NULL) perror ("Error opening file");

  /* 
     Create extra buffer for first instrumented kernel 
  */
  trace1_buffer = clCreateBuffer(program.getContext(),CL_MEM_READ_WRITE |
          CL_MEM_USE_HOST_PTR, sizeof(long long) * TRACE1_SIZE,h_trace1,&errorCode);
  verifyOutputCode(errorCode, "Error creating the trace buffer");

  /*
     Add extra buffer to kernel
  */
  cl_uint index = 0;
  errorCode = clSetKernelArg(length_kernel,index, sizeof(cl_mem), &trace1_buffer);
  verifySetArgumentCode(errorCode, index); 

}

//------------------------------------------------------------------------------
Kernel::~Kernel() throw() {
  fclose(fp);
  clReleaseMemObject(trace1_buffer);
  clReleaseMemObject(id_buffer);
  clReleaseMemObject(loop_buffer);
  clReleaseMemObject(addr_buffer);
  clReleaseKernel(kernel);
  clReleaseKernel(length_kernel);

  free(h_addr);
  free(h_trace1);
  free(h_id);
  free(h_loop);
}

//------------------------------------------------------------------------------

/* 
  Returns cl_kernel object corresponding to the kernel from the second instrumentation
*/
cl_kernel Kernel::getId() const {
  return kernel;
}


/* 
  Returns cl_kernel object corresponding to the kernel from the first instrumentation
*/
cl_kernel Kernel::getLenId() const {
  
  return length_kernel;
}


/*
   Allocate memory tp the second optimization's kernel based on
   reults from execution of the preliminary kernel
*/
void Kernel::alloc(const size_t *localSize,unsigned int dim){
  cl_int errorCode;

  /*
     Read length of trace
  */

  unsigned long long size = h_trace1[0] + 100;
  std::cout<<"TRACE LEN: "<<size-100<<std::endl;

  /*
     Checks for and alert user of any problems with the kernel that could
     affect results
  */
  if(h_trace1[1] ==0x15){
     std::cout<<"\nWARNING: More than 16 loops in kernel, will affect warp scheduling\n\n";
  }
  if(h_trace1[2] ==0x15){
    std::cout<<"\nWARNING: Kernel function call with global memory parameter, will affect warp scheduling\n\n";
  }
  if(h_trace1[3] ==1){
     std::cout<<"\nWARNING: Loops nested more than 3 deep, will affect warp scheduling\n\n";
  }
  if(h_trace1[4] ==1){
     std::cout<<"\nWARNING: loop with more than more than 2^16 iterations, will affect warp scheduling\n\n";
  }
  
  /*
    Allocate host memory for buffers
  */
  h_addr = (unsigned long long *)calloc(size, sizeof(long long));
  if(h_addr == NULL){
     std::cout <<"Memory allocation error, not enough space\n";
  }
  h_addr[0] = 1;

  h_id = (unsigned long long *)calloc(size, sizeof(long long));
  if(h_id == NULL){
     std::cout <<"Memory allocation error, not enough space\n";
  } 
  h_id[0] = 1;

  h_loop = (unsigned long long *)calloc(size, sizeof(long long));
  if(h_loop == NULL){
     std::cout <<"Memory allocation error, not enough space\n";
  }
  h_loop[0] = 1;

  cl_program prog; 
  cl_context context;

  errorCode= clGetKernelInfo(kernel,CL_KERNEL_CONTEXT,sizeof(cl_context),&context,NULL);
  verifyOutputCode(errorCode, "Error creating the trace buffer");

  errorCode= clGetKernelInfo(kernel,CL_KERNEL_PROGRAM,sizeof(cl_program),&prog,NULL);
  verifyOutputCode(errorCode, "Error creating the trace buffer");
  

  /*
    Initialize buffers 
   */

  addr_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE |
          CL_MEM_USE_HOST_PTR, sizeof(long long) * size,h_addr,&errorCode);

  id_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE |
          CL_MEM_USE_HOST_PTR, sizeof(long long) * size,h_id,&errorCode);
  verifyOutputCode(errorCode, "Error creating the trace buffer");

 
  loop_buffer  = clCreateBuffer(context,CL_MEM_READ_WRITE |
          CL_MEM_USE_HOST_PTR, sizeof(long long) * size,h_loop,&errorCode);
  verifyOutputCode(errorCode, "Error creating the warp buffer");

  /*
    Add buffers to kernel
  */

  cl_uint index = 0;
  errorCode = clSetKernelArg(kernel,index, sizeof(cl_mem), &addr_buffer);
  index++;
  errorCode |= clSetKernelArg(kernel,index, sizeof(cl_mem), &id_buffer);
  index++;
  errorCode |= clSetKernelArg(kernel,index, sizeof(cl_mem), &loop_buffer);
  index++;
  verifySetArgumentCode(errorCode, index); 

}

/*
   Dumps data from buffers into an output file
*/
void Kernel::write_trace(const size_t* local,unsigned int dim) const {

  /*
     Print to file local size metadata in each dimensions if this is the first time the kernel
     has executed
  */
  if(first_run){
    fprintf(fp,"local size:%d",*local);
    
    if(dim > 1){
       local++;
       fprintf(fp," %d",*local);
    }
    else{
       fprintf(fp," 0");
    }

    if(dim > 2){
      local++;
      fprintf(fp," %d\n",*local);
    }
    else{
      fprintf(fp," 0\n");
    }
  }

  /*
      Write each trace entry to file in as the entry in each buffer separated by
      a '|' sysbol
  */
  for(unsigned int i=1;i<h_trace1[0]+100;i++){
    if(h_addr[i] != 0)
     fprintf(fp,"%llX|%llX|%llX\n",h_addr[i], h_id[i],h_loop[i]);
  }

  /* 
     Signals end of file and is used as a barrier between iterations of the same kernel
  */
  fprintf(fp,"---------------\n");

 
}

//------------------------------------------------------------------------------
void verifySetArgumentCode(int errorCode, unsigned int index) {
  if(isError(errorCode)) {
    std::string errorMessage = "Error setting kernel argument number " + 
                               index;
    throwException(errorCode, errorMessage.c_str());
  }
}

//------------------------------------------------------------------------------
void Kernel::setArgument(unsigned int index, size_t size, const void* pointer) {
  
  /*
     Shift each added kernel agurment up to facilitate instrumented parameters 
     at the beginnig of each kernel
  */

  index++; 
  cl_int errorCode = clSetKernelArg(length_kernel, (cl_uint) index, size, pointer);
  verifySetArgumentCode(errorCode, index); 
  
  index += 2;
  errorCode = clSetKernelArg(kernel, (cl_uint) index, size, pointer);

  verifySetArgumentCode(errorCode, index); 
}

//------------------------------------------------------------------------------
void Kernel::setArgument(unsigned int index, const Buffer& buffer) {
  cl_mem rawBuffer = buffer.getId();
  setArgument(index, sizeof(cl_mem), &rawBuffer);
}

//------------------------------------------------------------------------------
// OpenCL 1.2 only.
//std::vector<size_t> getMaxGlobalWorkSize(const Device& device) const {
//  return KernelInfoTraits<std::vector<size_t> >::getKernelInfo(
//         kernel, device, 
//         CL_KERNEL_GLOBAL_WORK_SIZE); 
//}

//------------------------------------------------------------------------------
size_t Kernel::getMaxWorkGroupSize(const Device& device) const {
  return KernelInfoTraits<size_t>::getKernelInfo(kernel, device,
                                                 CL_KERNEL_WORK_GROUP_SIZE);
}

//------------------------------------------------------------------------------
unsigned long Kernel::getLocalMemoryUsage(const Device& device) const {
  return KernelInfoTraits<cl_ulong>::getKernelInfo(kernel, device,
                                                   CL_KERNEL_LOCAL_MEM_SIZE);
}

//------------------------------------------------------------------------------
unsigned long Kernel::getPrivateMemoryUsage(const Device& device) const {
  return KernelInfoTraits<cl_ulong>::getKernelInfo(kernel, device,
                                                   CL_KERNEL_PRIVATE_MEM_SIZE);
}

//------------------------------------------------------------------------------
size_t Kernel::getPreferredWorkGroupSizeMultiple(const Device& device) const {
  return KernelInfoTraits<size_t>::getKernelInfo(
         kernel, device,
         CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
}

//------------------------------------------------------------------------------
template <typename returnType>
returnType KernelInfoTraits<returnType>::getKernelInfo(
           cl_kernel kernelId,
           const Device& device,
           cl_kernel_work_group_info kernelInfoName) {
  returnType result;
  cl_int errorCode = clGetKernelWorkGroupInfo(kernelId, device.getId(),
                                              kernelInfoName,
                                              sizeof(returnType), 
                                              &result, NULL);
  verifyOutputCode(errorCode, "Error querying kernel info: ");
  return result;
}

//------------------------------------------------------------------------------
size_t getKernelInfoSize(
       cl_kernel kernelId,
       cl_device_id deviceId,
       cl_kernel_work_group_info kernelInfoName) {
  size_t result;
  cl_int errorCode = clGetKernelWorkGroupInfo(kernelId, deviceId,
                                              kernelInfoName,
                                              sizeof(result), 
                                              NULL, &result);
  verifyOutputCode(errorCode, "Error querying device info size: ");
  return result;
}

//------------------------------------------------------------------------------
// OpenCL 1.2 only.
//std::vector<size_t> KernelInfoTraits<std::vector<size_t> >::getKernelInfo(
//                    cl_kernel kernelId,
//                    const Device& device,
//                    cl_kernel_work_group_info kernelInfoName) {
//  size_t resultSize = getKernelInfoSize(deviceId, kernelInfoName);
//  size_t* rawResult = new size_t[resultSize];
//  cl_int errorCode = clGetKernelWorkGroupInfo(kernelId, device.getId(),
//                                              resultSize, rawResult, NULL);
//  std::vector<size_t> result(rawResult, rawResult + resultSize); 
//  delete [] rawResult;
//  return result; 
//}
