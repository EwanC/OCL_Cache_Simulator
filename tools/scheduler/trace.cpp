#include "trace.h"



void Trace_entry::pushLoopIter(unsigned int label, unsigned int counter){
  loops.counters.push_back(std::tuple<unsigned int,unsigned int>(label,counter));
}

 
void Trace_entry::setThreadIds(unsigned int d1, unsigned int d2, unsigned int d3){
  t_id.push_back(d1);
  t_id.push_back(d2);
  t_id.push_back(d3);
}

/*
  Calculates a unique ID for a thread based
  on it's id in all dimensions and the total 
  number of threads
*/
unsigned int Trace_entry::getThreadVal(const Trace& t)const{
  unsigned int index = 0;
   
  index += t_id[2] * (t.getGlobal(1) * t.getGlobal(0));
  
  index += t_id[1] * t.getGlobal(0);
  
  index += t_id[0] % t.getGlobal(0);
  
  return index; 

}


std::ostream & operator<< (std::ostream & os, const Trace_entry& right){
  os << "T ID: " <<right.t_id.at(0)<< " "<< right.t_id.at(1) << " " <<right.t_id.at(2)<<std::endl;
  os <<"Index: " <<right.index<<std::endl;
  os << "Loop depth " << right.loops.depth<<std::endl;

  for(int i=0;i<right.loops.depth;i++){
    os << "label "<<std::get<0>(right.loops.counters[i])<<" Val " \
    << std::get<1>(right.loops.counters[i])<<std::endl; 
  }

  if(!right.barrier){
    os << "Read: " << right.read <<std::endl;
    os << "ADDR: " <<std::hex << right.mem_addr <<std::dec <<std::endl;
    os << "Instr: "<<right.inst<<std::endl;
  }else{
    os <<"Barrier: " <<right.barrier<<std::endl;
  }

  os <<std::endl;
  return os;
}

std::ostream & operator<< (std::ostream & os, const Trace& right){
  os << "Trace length: "<< right.entries.size() << std::endl;
  os <<"Number of dimensions: "<<right.dims << std::endl;
  os <<"Global Size: "<<right.global_threads.at(0)<< " "\
                      <<right.global_threads.at(1)<< " "\
                      <<right.global_threads.at(2)<<std::endl;
  os <<"Local Size: "<<right.local_threads.at(0)<< " "\
                      <<right.local_threads.at(1)<< " "\
                      <<right.local_threads.at(2)<<std::endl; 
  return os;
}



Trace::Trace(){ 
  warp_size = 1;
  global_threads.push_back(0);
  global_threads.push_back(0);
  global_threads.push_back(0);
}


void Trace::setLocalSize(unsigned int d1, unsigned int d2, unsigned int d3){
  local_threads.push_back(d1);
  local_threads.push_back(d2);
  local_threads.push_back(d3);
}

void Trace::setGlobalSize(unsigned int index, unsigned int val){
   if(index < maxDim)
     global_threads[index] = val;
}

unsigned int Trace::ceiling(unsigned int a, unsigned int b) const{
 if(a % b ==0){
  return a/b;
 }

 return ((a/b) +1);
}

unsigned int Trace::getTotalThreads()const {
    return global_threads[0] * global_threads[1] * global_threads[2];
}

//calculates total number of workgroups
unsigned int Trace::getTotalWorkgroups()const{

  unsigned int num_wk = ceiling(global_threads[0],local_threads[0]);

  if(dims > 1)
    num_wk *= ceiling(global_threads[1],local_threads[1]);

  if(dims > 2) 
   num_wk *= ceiling(global_threads[2],local_threads[2]);

  return num_wk;

}

unsigned int Trace::warpsPerWorkgroup() const {
 
  unsigned int warps_per_workgroup;
 
  //number of warps in the first dimension of a workgroup
  warps_per_workgroup = local_threads.at(0);

  if(dims > 1){
    warps_per_workgroup *= local_threads.at(1);
  }

  if(dims > 2){
    warps_per_workgroup *= local_threads.at(2);
  }

  warps_per_workgroup = ceiling(warps_per_workgroup,warp_size);

  return warps_per_workgroup;
  }

unsigned int Trace::calcNumWarps() const{

  //number of workgroups
  int num_wkgroups = getTotalWorkgroups();

  //number of warps per workgroup
  int warps_per_group =  warpsPerWorkgroup();

  return (num_wkgroups * warps_per_group);
}

unsigned int Trace::workgroupsByDim(int dim)const{
    return ceiling(global_threads.at(dim), local_threads.at(dim));

}



