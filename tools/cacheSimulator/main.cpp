#include <algorithm>  

#include "parse.h"
#include "stats.h"
#include "cache.h"
#include "common.h"


//calculate workgroups to process based on total number of workgroups
std::vector<unsigned int>get_workgroups(unsigned int warp_size,unsigned int total_wk){

  std::vector<unsigned int> workgroups;

  //Seed rand() function
  srand(time(NULL));

  //calculate how many workgroups will be processed on each core
  unsigned int sim_num = ceiling(total_wk,CORES);

  // if only one workgroup per core, process first workgroup
  if(sim_num == 1){
  // std::cout << "wk group zero \n";
   workgroups.push_back(0);
   return workgroups;
  }

  //Picks workgroups at random to simulate, provided they are not duplicates 
  for(unsigned int i=0;i<sim_num;i++){
      unsigned int added = rand() % total_wk;
      while(std::find(workgroups.begin(),workgroups.end(),added) != workgroups.end() ){
          added = rand() % total_wk;
      }
    //  std::cout << "wk group " << added << std::endl;
      workgroups.push_back(added);
  }

  return workgroups;

}

/*
 *  Runs trace through simulator
*/
void exec_trace(TRACE_VEC& executions,Cache& cache){

  unsigned int n=0;
  for(TRACE_VEC::iterator iter = executions.begin(), end = executions.end(); iter != end; ++iter){
      std::cout <<"\nExecuting Trace " << n++ << " of "<<executions.size();
      cache.warp_size = std::get<1>(*iter);
      cache.reset_memory();

      std::vector<unsigned int> workgroups = std::get<0>(*iter);
      std::list<Entry> entries = std::get<2>(*iter);


      for(unsigned int w=0;w<workgroups.size();w++){
        //for every entry in workgroup
        for( std::list<Entry>::iterator e_iter = entries.begin(), \
           e_end = entries.end();e_iter!=e_end;++e_iter){  

            //check if entry is in current workgroup
            if(e_iter->wk_id == workgroups.at(w)){
              //Process with simulator
              if(e_iter->op==1)
                cache.read(e_iter->address,e_iter->warp_id,e_iter->inst);        //Cache read
              else
                cache.write(e_iter->address,e_iter->warp_id,e_iter->inst);       //Cache write
            }
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



  //Prints cache configuration information to stdout
  print_config(size,linesize,assoc, num_lines / assoc); 

  Cache cache(num_lines,linesize, assoc, replacement,write_pol);

  

  /*
  *  parses trace vector into a vector of traces from individual kernel executions
  */

  TRACE_VEC executions = parse(input);


  //Runs the trace through the simulator
  exec_trace(executions, cache);

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
