#include <vector>
#include <fstream>
#include <list>
#include <algorithm>  

#include "parse.h"
#include "stats.h"
#include "cache.h"
#include "common.h"

 
//calculate workgroups to process based on total number of workgroups
static std::vector<unsigned int>get_workgroups(unsigned int warp_size,unsigned int total_wk){

  std::vector<unsigned int> workgroups;

  //Seed rand() function
  srand(time(NULL));

  //calculate how many workgroups will be processed on each core
  unsigned int sim_num = ceiling(total_wk,CORES);

  // if only one workgroup per core, process first workgroup
  if(sim_num == 1){
   std::cout << "wk group zero \n";
   workgroups.push_back(0);
   return workgroups;
  }

  //Picks workgroups at random to simulate, provided they are not duplicates 
  for(unsigned int i=0;i<sim_num;i++){
      unsigned int added = rand() % total_wk;
      while(std::find(workgroups.begin(),workgroups.end(),added) != workgroups.end() ){
          added = rand() % total_wk;
      }
      std::cout << "wk group " << added << std::endl;
      workgroups.push_back(added);
  }

  return workgroups;

}

/*
 *  Runs trace through simulator
*/
void exec_trace(std::list<Entry> trace,Cache& cache,std::vector<unsigned int> wk){

 

  //for each workgroup to process
  for(unsigned int j=0;j<wk.size();j++){
    //for every entry in workgroup
    for( std::list<Entry>::iterator iter = trace.begin(), \
        end = trace.end();iter!=end;++iter){          

       //check if entry is in current workgroup
       if(iter->wk_id == wk.at(j)){
         //Process with simulator
         if(iter->op==1)
           cache_read(cache,iter->address,iter->warp_id,iter->inst);        //Cache read
         else
           cache_write(cache,iter->address,iter->warp_id,iter->inst);       //Cache write
       }

     }

  }
}


int main(int argc, char *argv[]){
  
  if(argc != 7){                      //Print help if wrong number of cli arguments
    printf("usage: %s \n",argv[0]);
    print_usage();
    return 0;
  }

  std::ifstream input(argv[1]);

  if(!input.is_open()){
    std::cout << "unable to open file "<< argv[1] <<std::endl;
    exit(0);
  }

  int size = atoi(argv[2]) * 1024;   //Get cache size from argument
  int linesize = atoi(argv[3]);      //Get line size from argument
  int assoc = atoi(argv[4]);         //Get associativity from argument

  if(size % linesize !=0){
    std::cout << "-----------------------------------\n";
    std::cout << "ERROR: size must be a multiple of line size\n";
    std::cout << "-----------------------------------\n";
    print_usage();
    return 0;  
  }

  int num_lines = size / linesize;    //Calculate total number of cache lines
  if(num_lines < assoc ){
    std::cout << "-----------------------------------\n";
    std::cout << "ERROR: associativity cannot be greater than the number of lines\n";
    std::cout << "-----------------------------------\n";

    print_usage();
    return 0;  
  }
  
 if(num_lines % assoc != 0){
    std::cout << "-----------------------------------\n";
    std::cout << "ERROR: number of lines must be a multiple of accociativity\n";
    std::cout << "-----------------------------------\n";

    print_usage();
    return 0;  
  }

  //Get replacement policy from cli argument
  int replacement = parse_replacement_policy(argv[5]);
  if(replacement==-1)
    return 0;

  //Get write policy from cli argument
  int write_pol = parse_write_policy(argv[6]);
  if(write_pol == -1)  
     return 0;

  /*
  *  get metadata from first line of input file
  */
  unsigned int warp_size;
  unsigned int total_wk;
  char metadata[maxLineSize] ;
  input.getline(metadata,maxLineSize);
  sscanf (metadata,"%u %u",&warp_size,&total_wk);


  //populate vector of workgroups to process
  std::vector<unsigned int> workgroups = get_workgroups(warp_size,total_wk);

  //Prints cache configuration information to stdout
  print_config(size,linesize,assoc, num_lines / assoc); 

  Cache cache(num_lines,linesize, assoc, replacement,write_pol,warp_size);

  
  /*
  *  Reads memory trace from file 
  */  

  
  std::string line;
  unsigned int  wk_id,warp_id,inst,op;
  unsigned long address;
  std::list<Entry> trace;
  while(getline(input,line)){

    sscanf (line.c_str(),"%lX %d %d %d %d\n",&address,&op,&wk_id,&warp_id,&inst);
    Entry e(address,op,wk_id,warp_id,inst);

    trace.push_back(e);
  }


  

  //Runs the trace through the simulator
  exec_trace(trace, cache,workgroups);

  //Prints cache performance data to stdout
  std::cout<<cache.stats;

 
}



/*
  Calculates the ceiling of a over b
*/
unsigned int ceiling(unsigned int a, unsigned int b){
  
   if(!a || !b)
       return 0;

   if(a% b ==0){
     return a/ b;
   }
   else{
     return (a/b) +1;
   }
}
