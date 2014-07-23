#include "OpenCLWrapper/Queue.h"

#include "OpenCLWrapper/Buffer.h"
#include "OpenCLWrapper/Context.h"
#include "OpenCLWrapper/Device.h"
#include "OpenCLWrapper/Event.h"
#include "OpenCLWrapper/Kernel.h"

#include "Utils.h"
#include <cstdlib>
// Constructors and Destructors.
//------------------------------------------------------------------------------
Queue::Queue(const Context& context, const Device& device, 
             CommandQueueFlags properties) {
  cl_int errorCode;
  queue = clCreateCommandQueue(context.getId(), device.getId(), 
                               properties, &errorCode);
  verifyOutputCode(errorCode, "Error creating the queue");
}

Queue::~Queue() throw() {
  clReleaseCommandQueue(queue);
}

// Read buffer.
//------------------------------------------------------------------------------
void Queue::readBuffer(const Buffer& buffer, size_t size, void* pointer) {
  readBuffer(queue, buffer.getId(), 1, 0, size, pointer, 0, NULL, NULL);
}

void Queue::readBuffer(const Buffer& buffer, size_t size, void* pointer,
                       Event& event) {
  readBuffer(queue, buffer.getId(), 1, 0, size, pointer, 0, NULL, 
             event.getId());
}

void Queue::readBuffer(const Buffer& buffer, 
                       bool blocking, 
                       size_t offset, 
                       size_t size,
                       void* pointer, 
                       const std::vector<Event*>& eventWaitList,
                       Event& event) {
  cl_event* waitListIds = new cl_event[eventWaitList.size()];
  getEventsId(eventWaitList, waitListIds);
  readBuffer(queue, buffer.getId(), blocking, offset, size,
             pointer, eventWaitList.size(), waitListIds, event.getId());
}

inline void Queue::readBuffer(cl_command_queue queue,
                              cl_mem buffer,
                              cl_bool blocking,
                              size_t offset, 
                              size_t size,  
                              void* pointer, 
                              cl_uint waitListSize,
                              const cl_event* waitList,
                              cl_event* event) {
  cl_int errorCode = clEnqueueReadBuffer(queue, buffer, blocking,
                                         offset, size, pointer, waitListSize,
                                         waitList, event);
  verifyOutputCode(errorCode, "Error reading the buffer");


}

// Write buffer.
//------------------------------------------------------------------------------
void Queue::writeBuffer(const Buffer& buffer, size_t size, void* pointer) {
  writeBuffer(queue, buffer.getId(), 1, 0, size, pointer, 0, NULL, NULL);
}
void Queue::writeBuffer(const Buffer& buffer, size_t size, void* pointer,
                        Event& event) {
  writeBuffer(queue, buffer.getId(), 1, 0, size, pointer, 0, NULL, 
              event.getId());
}

void Queue::writeBuffer(const Buffer& buffer, size_t offset,size_t size, void* pointer){

  cl_int errorCode = clEnqueueWriteBuffer(queue, buffer.getId(), true,
                       offset, size, pointer, 0, NULL, NULL);
  verifyOutputCode(errorCode, "Error writing the buffer");

}
void Queue::writeBuffer(const Buffer& buffer,
                        bool blocking,
                        size_t offset,
                        size_t size,
                        void* pointer,
                        const std::vector<Event*>& eventWaitList,
                        Event& event) {
  cl_event* waitListIds = new cl_event[eventWaitList.size()];
  getEventsId(eventWaitList, waitListIds);
  writeBuffer(queue, buffer.getId(), blocking, offset, size,
              pointer, eventWaitList.size(), waitListIds, event.getId());
}

inline void Queue::writeBuffer(cl_command_queue queue,
                               cl_mem buffer, 
                               cl_bool blocking, 
                               size_t offset, 
                               size_t size,
                               void* pointer, 
                               unsigned int waitListSize, 
                               const cl_event* waitList, 
                               cl_event* event) {
  cl_int errorCode = clEnqueueWriteBuffer(queue, buffer, blocking,
                                          offset, size, pointer, waitListSize,
                                          waitList, event);
  verifyOutputCode(errorCode, "Error writing the buffer");
}

// Execute kernel.
//------------------------------------------------------------------------------


void Queue::run( Kernel& kernel,
                unsigned int dimensionsNumber,
                const size_t *globalOffset,
                const size_t *globalSize,
                const size_t *localSize) {
   


  /*
     Run kernel from first instrumentation
  */ 

  run(queue, kernel.getLenId(), dimensionsNumber,
      globalOffset, globalSize, localSize, 0, NULL, NULL);

  cl_int err = clEnqueueReadBuffer(queue, kernel.trace1_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_trace1),kernel.h_trace1, 0, NULL, NULL);
  
   verifyOutputCode(err, "could not read buffer");

  /*
     Allocate buffers for second instrumentation
  */ 

  kernel.alloc(localSize,dimensionsNumber);
 
  /*
     Run kernel from second instrumentation
  */ 
  
  run(queue, kernel.getId(), dimensionsNumber,
      globalOffset, globalSize, localSize, 0, NULL, NULL);

  err = clEnqueueReadBuffer(queue, kernel.addr_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_addr),kernel.h_addr, 0, NULL, NULL);
  
  err = clEnqueueReadBuffer(queue, kernel.id_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_id),kernel.h_id, 0, NULL, NULL);

  err = clEnqueueReadBuffer(queue, kernel.loop_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_loop),kernel.h_loop, 0, NULL, NULL);

  verifyOutputCode(err, "could not read buffer");
 
  /*
     Dump trace to file
  */ 

  kernel.write_trace(localSize,dimensionsNumber);


}

void Queue::run(Kernel& kernel,
                unsigned int dimensionsNumber,
                const size_t *globalOffset,
                const size_t *globalSize,
                const size_t *localSize,
                Event& event) {

  /*
     Run kernel from first instrumentation
  */ 

  run(queue, kernel.getLenId(), dimensionsNumber,
      globalOffset, globalSize, localSize, 0, NULL, event.getId());

  cl_int err = clEnqueueReadBuffer(queue, kernel.trace1_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_trace1),kernel.h_trace1, 0, NULL, NULL);
  
  verifyOutputCode(err, "could not read buffer");

  /*
     Allocate buffers for second instrumentation
  */ 
  kernel.alloc(localSize,dimensionsNumber);

  /*
     Run kernel from second instrumentation
  */ 
  run(queue, kernel.getId(), dimensionsNumber,
      globalOffset, globalSize, localSize, 0, NULL, event.getId());

  /*
     Read trace data from buffers
  */ 

  err = clEnqueueReadBuffer(queue, kernel.addr_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_addr),kernel.h_addr, 0, NULL, NULL);

  err = clEnqueueReadBuffer(queue, kernel.id_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_id),kernel.h_id, 0, NULL, NULL);

  err = clEnqueueReadBuffer(queue, kernel.loop_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_loop),kernel.h_loop, 0, NULL, NULL);

  verifyOutputCode(err, "could not read buffer");
 
  /*
     Dump trace to file
  */ 
 
  kernel.write_trace(localSize,dimensionsNumber);

}

void Queue::run(Kernel& kernel,
                unsigned int dimensionsNumber,
                const size_t *globalOffset,
                const size_t *globalSize,
                const size_t *localSize,
                const std::vector<Event*>& eventWaitList,
                Event& event) {

  /*
     Run kernel from first instrumentation
  */ 

  cl_event* waitListIds = new cl_event[eventWaitList.size()];
  getEventsId(eventWaitList, waitListIds);

  run(queue, kernel.getLenId(), dimensionsNumber,
      globalOffset, globalSize, localSize,
      eventWaitList.size(), waitListIds, event.getId());  

  cl_int err = clEnqueueReadBuffer(queue, kernel.trace1_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_trace1),kernel.h_trace1, 0, NULL, NULL);
  
  verifyOutputCode(err, "could not read buffer");

  /*
     Allocate buffers for second instrumentation
  */ 

  kernel.alloc(localSize,dimensionsNumber);

  /*
     Run kernel from second instrumentation
  */ 
  run(queue, kernel.getId(), dimensionsNumber,
      globalOffset, globalSize, localSize,
      eventWaitList.size(), waitListIds, event.getId());

  /*
     Read trace data from buffers
  */ 

  err = clEnqueueReadBuffer(queue, kernel.addr_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_addr),kernel.h_addr, 0, NULL, NULL);

  err = clEnqueueReadBuffer(queue, kernel.id_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_id),kernel.h_id, 0, NULL, NULL);

  err = clEnqueueReadBuffer(queue, kernel.loop_buffer, CL_TRUE, 0, 
         sizeof(kernel.h_loop),kernel.h_loop, 0, NULL, NULL);
  
  verifyOutputCode(err, "could not read buffer");
 
  /*
     Dump trace to file
  */ 
 
  kernel.write_trace(localSize,dimensionsNumber);

}

inline void Queue::run(cl_command_queue queue,
                       cl_kernel kernel,
                       unsigned int dimensionsNumber,
                       const size_t *globalOffset,
                       const size_t *globalSize,
                       const size_t *localSize,
                       cl_uint waitListSize,
                       const cl_event* waitList,
                       cl_event* event) {

  cl_int errorCode = clEnqueueNDRangeKernel(queue, kernel,  
                                            dimensionsNumber, globalOffset, 
                                            globalSize, localSize,
                                            waitListSize, waitList, 
                                            event);
  

  // Special management for invalid block size.
  if(errorCode == CL_INVALID_WORK_GROUP_SIZE) {
     printf("invalid work group size\n");

    exit(CL_INVALID_WORK_GROUP_SIZE);
  }
  verifyOutputCode(errorCode, "Error launching the kernel");
}

void Queue::runTask(const Kernel& kernel) {
  runTask(queue, kernel.getId(), 0, NULL, NULL);
}

void Queue::runTask(const Kernel& kernel, Event& event) {
  runTask(queue, kernel.getId(), 0, NULL, event.getId());
}

void Queue::runTask(const Kernel& kernel,
                    const std::vector<Event*>& eventWaitList,
                    Event& event) {
  cl_event* waitListIds = new cl_event[eventWaitList.size()];
  getEventsId(eventWaitList, waitListIds);
  runTask(queue, kernel.getId(), eventWaitList.size(), waitListIds, 
          event.getId());
}

inline void Queue::runTask(cl_command_queue queue,
                           cl_kernel kernel,
                           cl_uint waitListSize,
                           const cl_event* waitList,
                           cl_event* event) {
  cl_int errorCode = clEnqueueTask(queue, kernel,
                                   waitListSize, waitList, event); 
  verifyOutputCode(errorCode, "Error launching the task");
}

// Manage the queue.
//------------------------------------------------------------------------------
void Queue::finish() {
  cl_int errorCode = clFinish(queue);
  verifyOutputCode(errorCode, "Error finishing the queue");

  
}

void Queue::flush() {
  cl_int errorCode = clFlush(queue);
  verifyOutputCode(errorCode, "Error flushing the queue");
}

//------------------------------------------------------------------------------
inline void Queue::getEventsId(const std::vector<Event*>& eventWaitList, 
                               cl_event* eventsId) const {
  for(unsigned int index = 0; index < eventWaitList.size(); ++index) {
    eventsId[index] = *(eventWaitList.at(index)->getId());
  }
} 
