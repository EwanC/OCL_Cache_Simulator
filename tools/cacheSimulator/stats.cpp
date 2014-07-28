#include "stats.h"
#include <fstream>


Stats::Stats(){
  reads = 0;                          
  readMisses =0;                    
  writes = 0;                         
  writeMisses =0;                    
  writeBacks = 0;                     
  coldMisses = 0;                       
  capacityMisses = 0;                 
  conflictMisses = 0;
}

void Stats::incrementReads(){
     ++reads; 
}

void Stats::incrementReadMisses(unsigned int stack_dist, unsigned int lines){
    ++readMisses;
    

   if(stack_dist  == Infinity){
      ++coldMisses;
   }
   else if(stack_dist >= lines ){
     ++capacityMisses;

   }
   else{
    ++conflictMisses;
   }

}

void Stats::incrementWrites(){
   ++writes;
}

void Stats::incrementWriteMisses(){
	++writeMisses = writeMisses;
}

void Stats::incrementWriteBacks(){
	  ++writeBacks;
}

/*
 * returns the total number of cache accesses
*/
int Stats::getNumAccess()const{
	return (reads + writes);
}

/*
 * returns the read miss rate
*/
double Stats::getReadMissRate()const{
   if(reads == 0)
   	 return 0;
   return ((double) readMisses / reads);
}

/*
 *  returns the write miss rate
*/
double Stats::getWriteMissRate()const {
	if(writes == 0)
   	 return 0;
   return ((double) writeMisses / writes);
}

/*
 *  returns the total miss rate
 */
double Stats::getTotalMissRate()const{
	if(getNumAccess() == 0)
   	 return 0;
   return ((double)(writeMisses + readMisses) / ((double)getNumAccess()));
}

std::ostream & operator<< (std::ostream & os, const Stats& right){
  os<<"\n==================================\n";
  os<<"RESULTS\n";
  os<<"==================================\n";
  os<<"Reads:           "<<right.reads << std::endl; 
  os<<"Read Misses:     "<<right.readMisses << std::endl;
  os<<"Writes:          "<<right.writes << std::endl;
  os<<"Write Backs:     "<<right.writeBacks << std::endl;
  os<<"Cold Misses:     "<<right.coldMisses << std::endl;
  os<<"Capacity Misses: "<<right.capacityMisses << std::endl;
  os<<"Conflict Misses: "<<right.conflictMisses << std::endl;
  os<<"Read  Miss Rate: "<<right.getReadMissRate() << std::endl;
  os<<"Write Miss Rate: "<<right.getWriteMissRate() << std::endl;
  os<<"Total Miss Rate: "<<right.getTotalMissRate() << std::endl<< std::endl;

  return os;
}

/*
* Writes cache performance stats out to file
*/
void Stats::Write(){
  std::ofstream output("results.out",std::ofstream::out);

  if(!output.is_open() ){
    std::cout <<"Error, could not open output file\n";
    return;
  }
  output << "Reads:           " << reads << std::endl;
  output << "Read Misses:     " << readMisses << std::endl;
  output << "Writes:          " << writes << std::endl;
  output << "Write Backs:     " << writeBacks << std::endl;
  output << "Cold Misses:     " << coldMisses << std::endl;
  output << "Capacity Misses: " << capacityMisses << std::endl;
  output << "Conflict Misses: " << conflictMisses << std::endl;
  output << "Read  Miss Rate: " << getReadMissRate() << std::endl;
  output << "Write Miss Rate: " << getWriteMissRate() << std::endl;
  output << "Total Miss Rate: " << getTotalMissRate() << std::endl;

  output.close();
}


bool operator== (const StackEntry& a,const StackEntry& b){

    return (a.tag == b.tag && a.set == b.set);

}



/*
 *  returns stack distance of given line and updates reuse stack
*/
unsigned int Stats::stackRef(intptr_t tag,int set){

  StackEntry access;    //Create stack entry
  access.tag = tag;     //set entry tag to the cache line tag
  access.set = set;    //set entry set to cache line set
 
  int size = 0;

  //stack is empty
  if(stack.empty() == 1){
     stack.push(access);
     return Infinity;
  }
  else{
    size = stack.size();
  }
  
  //stores all stack entries above access
  StackEntry above[size];
  
  //target is on top of stack, distance is zero and no updating is done
  if(stack.top() == access){
    return 0;
  }

  //look for target tag in stack
  for(int i = 0; i<size;i++){
        //pop top element
        StackEntry p = stack.top();
        stack.pop();

        //If element is target tag
        if(p == access){
           //add all the previously seen elemnts to the stack
           for(int j=i-1;j>=0;j--){
              stack.push(above[j]);
           }
           //add the target tag back to the top
           stack.push(access);
           return i;
        }
        else{
          above[i] = p;
        }
  }

  //Cache line is not in stack, so has not been seen before.
  for (int i = size -1; i>=0; i--)
  {
    stack.push(above[i]);
  }
  stack.push(access);

  return Infinity;

}




