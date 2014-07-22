#ifndef KERNEL_H
#define KERNEL_H
#include <stdio.h>
#include <CL/cl.h>

#define TRACE1_SIZE  5 //number of 'long long' entries in the added buffer for
                       //the first transformation

class Buffer;
class Device;
class Program;

class Kernel {

// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Kernel(const Program& program, const char* name);
  ~Kernel() throw();

  unsigned long long* h_trace1;       // Output from buffer in preliminary pass
  unsigned long long* h_addr;         // Output from first buffer in trace
  unsigned long long* h_id;           // Output from second buffer in trace
  unsigned long long* h_loop;         // Output from third buffer in trace
 
  cl_mem trace1_buffer;    //Buffer object for preliminary transformation
  cl_mem addr_buffer;      //First buffer added to second trace
  cl_mem id_buffer;        //Second buffer added to second trace
  cl_mem loop_buffer;      //third buffer added to second trace



// Public methods.
//------------------------------------------------------------------------------
public:
  cl_kernel getId() const;
  cl_kernel getLenId() const;
  bool first_run;

  void setArgument(cl_uint index, size_t size, const void* pointer);
  void setArgument(cl_uint index, const Buffer& buffer);
  
  //Dumps the memory trace into a file
  void write_trace(const size_t* local_size,unsigned int dim) const;
  
  //Uses information from the first trace to allocate data for the second trace
  void alloc(const size_t *localSize,unsigned int dim);

  // OpenCL 1.2 only.
  //std::vector<size_t> getMaxGlobalWorkSize(const Device& device) const;
  size_t getMaxWorkGroupSize(const Device& device) const;
  unsigned long getLocalMemoryUsage(const Device& device) const;
  unsigned long getPrivateMemoryUsage(const Device& device) const;
  size_t getPreferredWorkGroupSizeMultiple(const Device& device) const;

// Private Fields.
//------------------------------------------------------------------------------
private:
  FILE* fp;                              // Output file to dump trace
  cl_kernel kernel;                      // Kernel from second transformation
  cl_kernel length_kernel;               // Kernel from first transformation

};

// Traits.
//------------------------------------------------------------------------------
template <typename returnType> struct KernelInfoTraits {
  static returnType getKernelInfo(cl_kernel kernelId,
                                  const Device& device,
                                  cl_kernel_work_group_info kernelInfoName);
};

// OpenCL 1.2 only.
//template <> struct KernelInfoTraits<std::vector<size_t> > {
//  static std::vector<size_t> getKernelInfo(
//                             cl_kernel kernelId,
//                             const Device& device,
//                             cl_kernel_work_group_info kernelInfoName);
//};

#endif
