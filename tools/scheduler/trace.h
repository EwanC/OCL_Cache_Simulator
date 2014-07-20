#ifndef TRACE_H
#define TRACE_H


#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>
#include <list>
#include <algorithm>
#include <vector>
#include <tuple>

const unsigned int maxLineSize = 50; //Maximum input file line size
const unsigned int maxDim = 3;       //Maximum number of thread space dimensions
const unsigned int maxLoops= 3;      //Maximum number of nested loops

/*
  Tuple reprsenting a loop id, iteration pair
*/
typedef std::tuple<int,unsigned int> loopTuple; 

class Trace;

//Strcture representing any loops that the memory access was in
struct Loop_timestamp{

  int depth;                        //Number of nested loops 

  std::vector<loopTuple> counters;  //Loop execution counter 
};


class Trace_entry
{

 private:
  unsigned int index;     // Index in thread execution
  Loop_timestamp loops;   // Loop Data
  std::vector<unsigned int> t_id; // Thread ID
  unsigned int mem_addr;          // Memory Address
  bool barrier;                   // Is a barrier
  bool read;                      // Memory read or write
  unsigned int inst;              // Instrucion which made access
  unsigned int priority;          // priority for random scheduling
  const Trace *trace_ptr;         // pointer to Trace class

 public:

    friend std::ostream & operator<< (std::ostream & os, const Trace_entry& right);

    void setName(unsigned int name){inst= name;}
    unsigned int getName()const{return inst;}
  
    void setPriority(unsigned int p){priority= p;}
    unsigned int getPriority()const{return priority;}

    void setPointer(const Trace* ptr) { trace_ptr = ptr;}
    const Trace* getPointer() const { return trace_ptr;}

    void setBarrier(bool val){barrier = val;}
    bool getBarrier()const {return barrier;}
  
    void setRead(bool val){read = val;}
    bool getRead() const{return read;}
  
    void setIndex(unsigned int val){index = val;}
    unsigned int getIndex()const{return index;}
  
    void setLoopDepth(unsigned int i){loops.depth = i;}
    void pushLoopIter(unsigned int label, unsigned int counter);
 
    unsigned int getLoopDepth() const { return loops.depth;}
    Loop_timestamp getLoops() const { return loops;}

    void setThreadIds(unsigned int, unsigned int, unsigned int);
    unsigned int getThreadId(unsigned int index) const{
      return t_id.at(index);
    }
    unsigned int getThreadVal(const Trace& t)const;
     
    unsigned int getMemAddr() const{return mem_addr;}
    void setMemAddr(unsigned int addr){mem_addr = addr;}

	  Trace_entry() :barrier(false),priority(0) {};
   ~Trace_entry() {};
};



class Trace{
 
  public:
   std::list<Trace_entry> entries;

   enum Algorithm{ //Possible scheduling algorithms
     RR,
     SEQUENTIAL,
     COALESCED,
     RANDOM,
     NONE
   };
   
   Trace();
   friend std::ostream & operator<< (std::ostream & os, const Trace& right);

   unsigned int getDim() const { return dims;}
   void setDim(unsigned int d){ dims = d;}
  
   unsigned int getLocal(unsigned int index) const{return local_threads.at(index);}
   unsigned int getGlobal(unsigned int index) const{return global_threads.at(index);}

   void setLocalSize(unsigned int, unsigned int, unsigned int);
   void setGlobalSize(unsigned int, unsigned int);

   void setWarpSize(unsigned int val){warp_size = val;}
   unsigned int getWarpSize()const{return warp_size;}

   Trace::Algorithm getAlgorithm()const{return algorithm;}
   void setAlgorithm(Trace::Algorithm a){ algorithm = a;}

   unsigned int getTotalWorkgroups()const;
   unsigned int getTotalThreads()const;
   unsigned int calcNumWarps() const;
   unsigned int warpsPerWorkgroup() const;
   unsigned int workgroupsByDim(int dim)const;
   unsigned int calcTotalThreads() const{
    return global_threads.at(0) * global_threads.at(1) * global_threads.at(2);
   }
   
 private:
  unsigned int warp_size;  //Number of threads in warp
  unsigned short dims;     //Number of dimensions
  //Total number of threads in each dimension 
  std::vector<unsigned int> global_threads; 
  //number of threads in each dimension in a workgroup
  std::vector<unsigned int> local_threads;
  Trace::Algorithm algorithm; //Scheduling algoritm
  //returns the ceiling of first param over second param 
  unsigned int ceiling(unsigned int, unsigned int)const;


};


#endif //TRACE_H