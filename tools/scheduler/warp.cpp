#include "trace.h"
#include "warp.h"
#include <limits>


std::ostream & operator<< (std::ostream & os, const Loop_timestamp& right){
  os<<"Loop depth "<<right.depth <<std::endl;
  for ( auto loop : right.counters ){
             os << "label: " << std::get<0>(loop)\
                <<" val "<< std::get<1>(loop) <<std::endl;
         
    }

  return os;
}

/*
 calculates the lowest loop label creater than a paramtized minimum
*/
loopTuple minGreaterThan(std::vector<loopTuple> vec,int minLabel){
  
    unsigned int minVal =0;

    int lowestL= std::numeric_limits<unsigned int>::max();

    for ( auto loop : vec ){
         if(std::get<0>(loop) < lowestL && std::get<0>(loop) > minLabel){
             lowestL = std::get<0>(loop); 
             minVal = std::get<1>(loop);
         }
    }

    return loopTuple(lowestL,minVal);

}

/*
   Returns true if loop a was executed before loop b

*/
bool earlierLoop(const Loop_timestamp& a, const Loop_timestamp& b){
    

    unsigned int minDepth = std::min(a.depth,b.depth);

    loopTuple minA = loopTuple(-1,0);
    loopTuple minB = loopTuple(-1,0);

    for(unsigned int i =0; i< minDepth; i++){
    
     
      //find not processes outermost loop
      minA = minGreaterThan(a.counters,std::get<0>(minA));
      minB = minGreaterThan(b.counters,std::get<0>(minB));
      
     
      //if outmost loops if not to same 
      if(std::get<0>(minA) != std::get<0>(minB))
        return std::get<0>(minA) < std::get<0>(minB);

      //if number of iterations of outermost loop is not the same
      if(std::get<1>(minA) != std::get<1>(minB))
        return std::get<1>(minA) < std::get<1>(minB);


    } 

    return false;

}

/*
  Calculates and workgroup of an access
  based on it thread id and workgroup size 
*/
unsigned int getWorkgroupId(const Trace_entry& a){

 unsigned int D1Workgroups = a.getPointer()->workgroupsByDim(0);
 unsigned int D2Workgroups = a.getPointer()->workgroupsByDim(1);

 unsigned int D1 = a.getThreadId(0) / a.getPointer()->getLocal(0);
 unsigned int D2 = 0;
 unsigned int D3 = 0;

if(a.getPointer()->getDim() > 1){
    D2 = a.getThreadId(1) / a.getPointer()->getLocal(1);
  }
  
if(a.getPointer()->getDim() > 2){
    D3 = a.getThreadId(2) / a.getPointer()->getLocal(2);
} 

unsigned int result = D1 + (D2 * D1Workgroups);
result += D3 * (D1Workgroups * D2Workgroups);

return result;


}


/*
   Caluclates the the warp id of an entry basen on the 
   workgroup of the thread and number of warps per workgroup
*/
unsigned int getWarpId(const Trace_entry& a){
  
  unsigned int workgroup = getWorkgroupId(a);

  //Number of warps in each dimension of a workgroup
  unsigned int warp1D = a.getThreadId(0) % a.getPointer()->getLocal(0);
  unsigned int warp2D = 0;
  unsigned int warp3D = 0;

  
  if(a.getPointer()->getDim() > 1){
    warp2D = a.getThreadId(1) % a.getPointer()->getLocal(1);
  }
  
  if(a.getPointer()->getDim() > 2){
    warp3D = a.getThreadId(2) % a.getPointer()->getLocal(2);
  }

  
  unsigned int result = warp1D + (warp2D * a.getPointer()->getLocal(0));
  result += warp3D * (a.getPointer()->getLocal(0) * a.getPointer()->getLocal(1));
  result /= a.getPointer()->getWarpSize();

 // result = result + (workgroup *  a.getPointer()->warpsPerWorkgroup());  
  result = workgroup + (result * a.getPointer()->getTotalWorkgroups());  

  return result;
}

/*
  Sorts linked list based on the following critrtia in ascending priority
  
  If a access was made in a earlier loop then rank it
  higher

  If a access has an earlier instruction then it's got
  higher priority

  If a access has an earlier warp then it's got
  higher priority. Definition of earlier is aribitary 
  but enusures accesses are sorted according to warp.

  Otherwise rank according to thread id

*/

bool warp_compare( const Trace_entry& a, const Trace_entry& b){
  

  // entry A is executed in a loop before entry B
  if (earlierLoop(a.getLoops(),b.getLoops()) ){
       return true;
  }

  // entry B is executed in a loop before entry A
  if(earlierLoop(b.getLoops(),a.getLoops()) ){
       return false;
  }

  // entry A is executed in a earlier instruction than entry B
  if( a.getName() !=  b.getName())
      return a.getName() < b.getName();

  // entry A is executed in a 'earlier' warp than entry B

  unsigned int warpA = getWarpId(a);
  unsigned int warpB = getWarpId(b);

  if(warpA != warpB)
     return warpA < warpB;
  
  // entry A is has a lower thread id than enry B
  return a.getThreadVal(*(a.getPointer())) < b.getThreadVal(*(b.getPointer()));


}