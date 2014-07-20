#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#include <boost/filesystem.hpp>

#include "Buffer.h"
#include "Device.h"
#include "Event.h"
#include "Kernel.h"
#include "Platform.h"
#include "Program.h"
#include "Queue.h"
#include "SystemConfiguration.h"

struct Node
{
  int starting;
  int no_of_edges;
};




using namespace boost::filesystem;
//-----------------------------------------------------------------------------
#define REPETITIONS 1
#define MAX_THREADS_PER_BLOCK 512
#define ELEMENT_LIMIT 5
#define DIMENSIONS 1
#define DEVICE_ID 0
#define PLATFORM_ID 0
#define KERNEL_FILE_NAME "bfs.cl"


//-----------------------------------------------------------------------------
void initialization(int argc, char** argv);
void hostMemoryAlloc();
void deviceMemoryAlloc();
void setKernelArguments();
void writeResults();
void enqueWriteCommands(Queue& queue);
void enqueReadCommands(Queue& queue);
void run(const Context* context, Queue& queue);
void freeMemory();


void readFile();                                                                   


//-----------------------------------------------------------------------------
// Runtime components.
Platform* platform;
Kernel* kernel;


// Device data.
Buffer*  d_graph_nodes;
Buffer*  d_graph_edges;
Buffer*  d_graph_mask;
Buffer*  d_updating_graph_mask;
Buffer*  d_graph_visited;
Buffer*  d_cost;
Buffer*  d_over;

std::string kernelName = "BFS_kernel";



FILE* fd;

//the number of nodes in the graph
int numberOfNodes= 0; 
//the number of edges in the graph
int numberOfEdges = 0;

int numberOfBlocks;
int numberOfThreadsPerBlock;
int workGroupSize;


bool *h_graph_mask,*h_updating_graph_mask,*h_graph_visited;


struct Node* h_graph_nodes;
int* h_graph_edges;
int* h_cost;


int source;  



//-----------------------------------------------------------------------------

int main(int argc, char** argv) {
  initialization(argc, argv);
  platform = new Platform(PLATFORM_ID);
  Context* context = platform->getContext();
  Device device = platform->getDevice(DEVICE_ID);
  std::cout << "Running " << kernelName << " on " << device.getName() << "\n";
  
  hostMemoryAlloc();

  readFile();


  deviceMemoryAlloc();

  
  SourceFile kernelFile = KERNEL_DIRECTORY KERNEL_FILE_NAME;

  Program program(context, kernelFile);
  Queue queue(*context, device, Queue::EnableProfiling);
  if(!program.build(device)) {
    std::cout << "Error building the program: " << "\n";
    std::cout << program.getBuildLog(device) << "\n";
    return 1;
  }
  kernel = program.createKernel(kernelName.c_str());
  
  enqueWriteCommands(queue);

  run(context, queue);

  writeResults();
  freeMemory();
  return 0;
}




//-----------------------------------------------------------------------------
void initialization(int argc, char** argv) {
  if(argc > 1){
    fd = fopen(argv[1],"r");
   if(!fd)
   {
    printf("Error Reading graph file\n");
    exit(1);
   }
 

   fscanf(fd,"%d",&numberOfNodes);
 }
  else {
    std::cout << "Error passing the parameters: file name required." << 
                 std::endl; 
    exit(1);
  }

  numberOfBlocks = 1;
  numberOfThreadsPerBlock = numberOfNodes;

  if(numberOfNodes > MAX_THREADS_PER_BLOCK){
      numberOfBlocks = (int)ceil(numberOfNodes/(double)MAX_THREADS_PER_BLOCK);
      numberOfThreadsPerBlock = MAX_THREADS_PER_BLOCK; 
  }
  workGroupSize = numberOfThreadsPerBlock;
}

//-----------------------------------------------------------------------------
void freeMemory() {
  
  delete kernel;
  delete platform;

  delete [] h_graph_nodes;
  delete [] h_graph_edges;
  
  

}

//-----------------------------------------------------------------------------
void hostMemoryAlloc() {
   // allocate host memory
  // h_graph_nodes = (struct Node*) malloc(sizeof(struct Node) * numberOfNodes);
  // colour = (int*) malloc(sizeof(int) * numberOfNodes);
 

  // allocate host memory
  h_graph_nodes = (Node*) malloc(sizeof(Node)*numberOfNodes);
  h_graph_mask = (bool*) malloc(sizeof(bool)*numberOfNodes);
  h_updating_graph_mask = (bool*) malloc(sizeof(bool)*numberOfNodes);
  h_graph_visited = (bool*) malloc(sizeof(bool)*numberOfNodes);

}

//-----------------------------------------------------------------------------
void deviceMemoryAlloc() {
  d_graph_nodes = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                 numberOfNodes*sizeof(struct Node), NULL);
 
  d_graph_edges = new Buffer(*(platform->getContext()), Buffer::ReadOnly,
                 numberOfEdges*sizeof(int), NULL);

  d_graph_mask = new Buffer(*(platform->getContext()), Buffer::ReadWrite,
                 numberOfNodes*sizeof(bool), NULL);
  d_updating_graph_mask = new Buffer(*(platform->getContext()), Buffer::ReadWrite,
                  numberOfNodes*sizeof(bool), NULL);
  d_graph_visited = new Buffer(*(platform->getContext()), Buffer::ReadWrite,
                  numberOfNodes*sizeof(bool), NULL);
  d_cost = new Buffer(*(platform->getContext()), Buffer::ReadWrite,
                  numberOfNodes*sizeof(int), NULL);

  d_over = new Buffer(*(platform->getContext()), Buffer::ReadWrite,
                    sizeof(bool), NULL);

}

//-----------------------------------------------------------------------------
void enqueWriteCommands(Queue& queue) {
  queue.writeBuffer(*d_graph_nodes,numberOfNodes*sizeof(struct Node), (void*) h_graph_nodes);
  queue.writeBuffer(*d_graph_edges,numberOfEdges*sizeof(int), (void*)h_graph_edges);
  queue.writeBuffer(*d_graph_mask,numberOfNodes*sizeof(bool), (void*)h_graph_mask);
  queue.writeBuffer(*d_updating_graph_mask,numberOfNodes*sizeof(bool), (void*)h_updating_graph_mask);
  queue.writeBuffer(*d_graph_visited,sizeof(bool),(void*) h_graph_visited);
  queue.writeBuffer(*d_cost,sizeof(int), (void*)h_cost);


  //queue.finish();
}

//-----------------------------------------------------------------------------
void enqueReadCommands(Queue& queue) {
   // copy result from device to host

  // queue.readBuffer(*d_cost,numberOfNodes*sizeof(int), (void*) h_cost);
  // queue.readBuffer(*d_color,numberOfNodes*sizeof(int), (void*) colour);

}

//-----------------------------------------------------------------------------
void setKernelArguments() {

  kernel->setArgument(0, *d_graph_nodes);
  kernel->setArgument(1, *d_graph_edges);
  kernel->setArgument(2, *d_graph_mask);
  kernel->setArgument(3, *d_updating_graph_mask);
  kernel->setArgument(4, *d_graph_visited);
  kernel->setArgument(5, *d_cost);
  kernel->setArgument(6, *d_over);
  kernel->setArgument(7,sizeof(int),(void*)&numberOfNodes);


}

//-----------------------------------------------------------------------------
void run(const Context* context, Queue& queue) {
  const size_t nodes = numberOfNodes;
  const size_t worksize = workGroupSize;
  bool h_over;
  do
  {
   
     h_over = false;
     queue.writeBuffer(*d_over, sizeof(bool),(void*) &h_over);


    setKernelArguments();
  
    queue.run(*kernel, DIMENSIONS, 0, &nodes,&worksize);
    queue.readBuffer(*d_over,sizeof(bool), (void*) &h_over);
    queue.finish();

  } while(h_over);

  printf("GPU kernel done\n");
  
}

void readFile()                                                                   
{                                                                                                    
   
  
  // initalize the memory
  int i,id,cost,edgeno,start;
  for( i = 0; i < numberOfNodes; i++) 
  {
    fscanf(fd,"%d %d",&start,&edgeno);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_graph_mask[i]=false;
    h_updating_graph_mask[i] = false;
    h_graph_visited[i]=false; 
  }
  
  //read the source node from the file
  fscanf(fd,"%d",&source);
  source=0;
  h_graph_mask[source]=true;
  h_graph_visited[source]=true;
  fscanf(fd,"%d",&numberOfEdges);
  
  h_graph_edges = (int*) malloc(sizeof(int) * numberOfEdges);

  for(i=0; i < numberOfEdges ; i++)
  {
    fscanf(fd,"%d",&id);
    fscanf(fd,"%d",&cost);
    h_graph_edges[i] = id;
  }
  if(fd)
    fclose(fd);        

  h_cost = (int*) malloc( sizeof(int)*numberOfNodes);
  for(i = 0; i < numberOfNodes; i++){
     h_cost[i] = -1;
  }
  h_cost[source] = 0;


}     

void writeResults(){
   FILE *fp = fopen("graph.out","w");
   
   fprintf(fp, "%d\n", numberOfNodes);
   
   int j = 0;
   for(j=0;j<numberOfNodes;j++)
       fprintf(fp,"%d %d\n",j,h_cost[j]);
   
   fclose(fp);
}
