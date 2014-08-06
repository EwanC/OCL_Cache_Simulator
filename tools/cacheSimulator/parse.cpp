/*

Copyright 2014 Ewan Crawford<ewan.cr@gmail.com>


This file is part of OpenCL Visuliser.

OpenCL Visuliser is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OpenCL Visuliser is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OpenCL Visuliser.  If not, see <http://www.gnu.org/licenses/>
*/

#include "parse.h"
#include "cache.h"



TRACE_VEC parse(std::ifstream& input){
  TRACE_VEC exections;


  unsigned int warp_size;
  unsigned int total_wk;
  std::string line;

  getline(input,line);
  sscanf (line.c_str(),"%u %u",&warp_size,&total_wk);
  std::vector<unsigned int> workgroups = get_workgroups(warp_size,total_wk);
  
  
  /*
  *  Reads memory trace from file 
  */  

  unsigned int  wk_id,warp_id,inst,op;
  unsigned long address; 
  std::list<Entry> trace;
  while(getline(input,line)){
  
    if((line.find("-") < line.length()) && !input.eof()){
         exections.push_back(std::make_tuple(workgroups,warp_size,trace));
         
         getline(input,line);
         if(!input.eof()){

         
           sscanf (line.c_str(),"%u %u",&warp_size,&total_wk);
           workgroups = get_workgroups(warp_size,total_wk);
        } 
         
    }else{

      sscanf (line.c_str(),"%lX %d %d %d %d\n",&address,&op,&wk_id,&warp_id,&inst);
      Entry e(address,op,wk_id,warp_id,inst);

      trace.push_back(e);
    }

  }

  return exections;

}



/*
 *  Parses to cache write policy from cli argument
*/
int parse_write_policy(char* arg){
  if(strcmp("WBWA",arg)==0)         //Write Back Write Allocate
    return CACHE_WRITEPOLICY_WBWA;
  else if(strcmp("WTNA",arg)==0)    //Write Through No-Allocate
    return CACHE_WRITEPOLICY_WTNA;
  else{                             //Invalid Policy
    std::cout << "-----------------------------------\n";
    std::cout << "No valid write policy selected\n";
    std::cout << "-----------------------------------\n";
    print_usage();
    return -1;
  }
}

/*
 *  Parses to cache replacement policy from cli argument
*/
int parse_replacement_policy(char* arg){
  if(strcmp("LRU",arg)==0)                  //Least Recently Used
    return CACHE_REPLACEMENTPOLICY_LRU;
  else if(strcmp("RAND",arg)==0)            //Random
    return CACHE_REPLACEMENTPOLICY_RANDOM;
  else if(strcmp("MRU",arg)==0)             //Most Recently Used
    return CACHE_REPLACEMENTPOLICY_MRU;
  else if(strcmp("LFU",arg)==0){            //Least Frequently Used
    return CACHE_REPLACEMENTPOLICY_LFU;
  }
  else{                                     //Invalid Policy
    std::cout << "-----------------------------------\n";
    std::cout << "No valid replacement policy selected\n";
    std::cout << "-----------------------------------\n";
    print_usage();
    return -1;
  }
}



/*
 *  Prints help on how to use the program
*/
void print_usage(){
    std::cout << "filename\n";
    std::cout << "size in KB\n";
    std::cout << "line size in Bytes\n";
    std::cout << "associativity\n";
    std::cout << "replacement policy: 'LRU','LFU,'MRU', 'RAND'\n";
    std::cout << "writepolicy: 'WBWA','WTNA'\n";
}


/*
 *  prints cache configuration to stdout
*/
void print_config(int size, int line_size, int assoc, int num_sets){ 
  std::cout << "\n==================================\n";
  std::cout << "CACHE CONFIGURATION\n";
  std::cout << "==================================\n";
  std::cout << "SIZE(Bytes):       "<< size << std::endl;
  std::cout << "LINE SIZE(BYTES):  "<< line_size << std::endl;
  std::cout << "ASSOCIATIVITY:     "<< assoc << std::endl;
  std::cout << "NUMBER OF SETS:    "<< num_sets << std::endl;
}