/*
   Ewan Crawford, ewan.cr@gmail.com
   July 2014
*/
#include <fstream>
#include "trace.h"
#include "parse.h"
#include "schedule.h"
#include "warp.h"

/*

 Uses a command line argument to set the scheduling algorithm. 
 Recorded in the Trace object.

 If no warp size is specified for coalesced scheduling, default
 to 32 threads.

*/

unsigned int argWarpSize;


void assignAlgorithm ( char* argv[], int argc) {

  if(!strcmp(argv[2],"rr")){             // Round Robin scheduling
     Trace::algorithm = Trace::RR;
  }
  else if(!strcmp(argv[2],"seq")){       // Sequential Scheduling
    Trace::algorithm = Trace::SEQUENTIAL;
  }
  else if(!strcmp(argv[2],"rand")){      // Random scheduling
     Trace::algorithm = Trace::RANDOM;
  }
  else if(!strcmp(argv[2],"coalesced")){ // Coalesced scheduling

    if(argc < 4){
      argWarpSize = 32;
    } else{
      argWarpSize = atoi(argv[3]);
    }
    Trace::algorithm = Trace::COALESCED;
  }
  else if(!strcmp(argv[2],"none")){       // No scheduling
    Trace::algorithm = Trace::NONE;    
  }
}


/* 
   Checks a valid scheduling algorithm is given as a command line argument

*/
void validateArguments(int argc, char *argv[]){

  // Check number of arguments.
  if(argc < 3){
    std::cout << "Usage: " << argv[0] << " 'filename' 'algorithm'\n";
    std::cout << "algorithm options: 'none','rr','rand','seq','coalesced' 'warp size'\n";
    std::exit(0);
  }

  // Check valid algorithm option given
  int valid_alg = strcmp(argv[2],"none");
  valid_alg = valid_alg && strcmp(argv[2],"rr") && strcmp(argv[2],"seq");
  valid_alg = valid_alg && strcmp(argv[2],"coalesced") && strcmp(argv[2],"rand"); 

  if(valid_alg){
    std::cout <<"Algorithm not supported\n";
    std::exit(0);
  }

}

/* 

   Prints reordered trace to files for graphing and cache simulation.

*/
void writeOutput(const Trace* trace){

  // File used for plotting memory accesses using R.
  std::ofstream graph("graph.out",std::ofstream::out | std::ofstream::app);

  // File used for simulating cache performance. 
  std::ofstream cache("cache.out",std::ofstream::out | std::ofstream::app);

  
  if(!graph.is_open() || !cache.is_open()  ){
    std::cout <<"Error, could not open output file\n";
    return;
  }

  /*
    Cache simulation needs to know the number of threads in a warp and
    total number of workgroups. This is provided as the first line
    of the input file.
  */
  cache <<trace->getWarpSize() << " "<<trace->getTotalWorkgroups()<<std::endl;

  for( std::list<Trace_entry>::const_iterator iter = trace->entries.begin(), \
       end = trace->entries.end();iter!=end;++iter)
  {
 

       if(!iter->getBarrier()){   //Don't print barriers

           graph << std::hex <<iter->getMemAddr()<<std::dec << " "\
                 << iter->getRead() << " "\
                 << iter->getThreadId(0) << " " \
                 << iter->getThreadId(1) << " " \
                 << iter->getThreadId(2) << std::endl;     

           cache << std::hex <<iter->getMemAddr()<<std::dec << " "\
                 << iter->getRead() << " "\
                 << getWorkgroupId(*iter)  << " " \
                 << getWarpId(*iter) << " " \
                 << iter->getName() << std::endl;            }
  }

  cache << "------------------------"<<std::endl;

  graph.close();
  cache.close();
}

/*
  Every trace entry is given an index specifying to number of 
  entries the thread has made before.
*/
void setIndices(Trace& trace){
 std::cout << trace<<std::endl;
 // Number of accesses made by each thread
 std::vector<int>num_accesses(trace.getTotalThreads(),0);
 
 for( std::list<Trace_entry>::iterator iter = trace.entries.begin(), \
        end = trace.entries.end();iter!=end;++iter)
    {
        unsigned int tVal = iter->getThreadVal(trace);
        //std::cout <<*iter<<tVal<<" "<<trace.getTotalThreads() <<std::endl;

        iter->setIndex(num_accesses[tVal]++);

    }

}


/*
 First line of the input trace file provides metadata on the trace 
 regarding workgroup size in the form 'local size: x y z'. Where
 x, y, & z are the number of threads in the workgroup in three
 dimensions.

 If a the dimension does not exist in the thread space it's value is zero.
 This is used to record the number of dimensions in the trace.
*/
void setThreadDim(Trace* trace, std::ifstream& input){

  char localSize[maxLineSize] ;
  input.getline(localSize,maxLineSize);
  unsigned int local_dim[3];
  sscanf (localSize,"local size:%d %d %d",&local_dim[0],&local_dim[1],&local_dim[2]);
 
  // Count number of dimensions
  unsigned short dim_count = 0;
  for(unsigned int i = 0; i< maxDim; i++){
    if(local_dim[i] > 0){
       ++dim_count;
    }
    else {local_dim[i] = 1;}
  }

  trace->setDim(dim_count);
  trace->setLocalSize(local_dim[0],local_dim[1],local_dim[2]);

  /* 
    Checks that the number of threads in a warp is less than the number of 
    threads in a workgroup, since warps cannot span workgroups.
  */
  unsigned int workgroupSize = local_dim[0] * local_dim[1]* local_dim[2];
  if(workgroupSize < argWarpSize){
      trace->setWarpSize(workgroupSize);
  }
  else {
      trace->setWarpSize(argWarpSize);
  }

}

void parse(std::ifstream& input, std::vector<Trace*>& executions){
  /*
    Reads and parses the input file line by line, 
    where a line is a trace entry.
  */
  std::string line;

  Trace* curr = new Trace(); 
  
  // Sets the workgroup size based on first line of input file
  setThreadDim(curr,input);

  while(getline(input,line)){

    bool end =  parseInput(line,*curr);
    if(end && !input.eof()){

      if(curr->getGlobal(1) == 0) 
        curr->setGlobalSize(1,1); 
  
      if(curr->getGlobal(2) == 0) 
        curr->setGlobalSize(2,1);


      // Give each access an index in the order of it's thread's accesses
      setIndices(*curr);

      executions.push_back(curr);


      curr = new Trace();

      // Sets the workgroup size based on first line of input file
      setThreadDim(curr,input);


    }
  } 

}

int main(int argc, char *argv[]){
  
  // Checks valid command line arguments are given
  validateArguments(argc,argv);

  std::ifstream input_file(argv[1]);
  if(!input_file.is_open()){
  	std::cout << "unable to open file "<< argv[1] <<std::endl;
  	exit(0);
  }

  // Reads the scheduling algorithm to use.
  assignAlgorithm(argv,argc);

  std::vector<Trace*> executions;

  parse(input_file,executions);

  // Schedule trace according to specified algorithm.
  if(Trace::algorithm != Trace::NONE){
    for_each(executions.begin(),executions.end(),schedule);
  }
  
  // Prints reordered trace to output file.
  for_each(executions.begin(),executions.end(),writeOutput);

  return 0;
}

