#include "trace.h"
#include "schedule.h"
#include <array>
#include <queue>

/*
  Prioritse entries based on index, 
  and if entries are from same index then on thread ID.

  This gives a round robin ordering.
*/
bool rr_compare( const Trace_entry& a, const Trace_entry& b){
  if(a.getIndex() != b.getIndex())
    return a.getIndex() < b.getIndex();
  
  if(a.getThreadId(2)!= b.getThreadId(2)) 
    return a.getThreadId(2) < b.getThreadId(2);
  
  if(a.getThreadId(1) != b.getThreadId(1)) 
    return a.getThreadId(1) < b.getThreadId(1);
   
  return a.getThreadId(0) < b.getThreadId(0);
  
}

/*
  Prioritse entries based on thread ID, 
  and if entries are from same thread then on index.

  This gives a sequential ordering.
*/
bool seq_compare( const Trace_entry& a, const Trace_entry& b){
  if(a.getThreadId(2) != b.getThreadId(2)) 
    return a.getThreadId(2) < b.getThreadId(2);
  
  if(a.getThreadId(1)!= b.getThreadId(1)) 
    return a.getThreadId(1)< b.getThreadId(1);
   
  if(a.getThreadId(0) != b.getThreadId(0)) 
    return a.getThreadId(0) < b.getThreadId(0);
  

  return a.getIndex() < b.getIndex();
  
}

bool random_compare( const Trace_entry& a, const Trace_entry& b){
  
  return a.getPriority() < b.getPriority();
  
}

/*
  Assins random priotities to each of the entries so they can be
  sorted randomly.
*/
void assignPriorities(std::list<Trace_entry>& list){
 
  const Trace *trace = list.begin()->getPointer();
  std::vector<unsigned int> prioirties;
  std::srand(time(0));
  unsigned int NumThreads = trace->getTotalThreads();

  // Give each thread a priority 
  for(unsigned int i=0; i< NumThreads; ++i){
    prioirties.push_back(rand() % ((NumThreads / 4 )+1));
  }

  /* 
    Give each entry a priority so that earlier accesses
     are ranked higher than later ones, to preseve
     intra-thread ordering.
  */
  for( std::list<Trace_entry>::iterator iter = list.begin(), \
        end = list.end();iter!=end;++iter){
       unsigned int p = prioirties.at(iter->getThreadVal(*trace));

      iter->setPriority(p + iter->getIndex());
  }

  // Scramble linked list
  std::vector<Trace_entry> tmpVec = std::vector<Trace_entry>(list.begin(),list.end());
  std::random_shuffle(tmpVec.begin(),tmpVec.end());
  list = std::list<Trace_entry>(tmpVec.begin(),tmpVec.end());

}


/*
   Calls routine to sort linked list depending on algorithm
*/
void sort(std::list<Trace_entry>& list){

    switch(Trace::algorithm){
     case Trace::RR :
        list.sort(rr_compare); 
        break;
     case Trace::SEQUENTIAL :
        list.sort(seq_compare);
        break;
     case Trace::COALESCED :
          list.sort(warp_compare); // defined in warp.cpp
        break;
     case Trace::RANDOM :
        assignPriorities(list); // give each entry a random priority
        list.sort(random_compare);
        break;
     default: 
        break;
   }
}


/*
  Reorders the Trace_entry elements in the Trace linked links
  according the the scheduling algorithm.

*/
void schedule(Trace* trace){
 
 /*
   Count the number of memory barriers in the trace.
   Since every thread must see a barrier, we only
   need to count those seen by one thread.
 
 */


 unsigned int barrier_count = 1;
 for( std::list<Trace_entry>::const_iterator iter = trace->entries.begin(), \
        end = trace->entries.end();iter!=end;++iter)
  {  

      if(iter->getThreadId(0) == 0 && \
        iter->getThreadId(1) == 0 &&\
        iter->getThreadId(2) == 0)
      {
        if(iter->getBarrier())
          ++barrier_count;
      }
  }

  
 
 if(barrier_count == 1){            // No barriers
    sort(trace->entries);
 } else {                           //Barriers are present

   // Pariton list into a linked list for entries between barriers
   std::list<Trace_entry>* split = new std::list<Trace_entry>[barrier_count];
   
   // Record the number of barriers seen by each thread.
   std::vector<int>barriers_seen(trace->getTotalThreads(),0);

   //Assign each entry to a partition.
   for( std::list<Trace_entry>::const_iterator iter = trace->entries.begin(), \
        end = trace->entries.end();iter!=end;++iter)
    {
        unsigned int tVal = iter->getThreadVal(*trace);
        unsigned int index = barriers_seen[tVal];
 
        if(iter->getBarrier()){
          ++barriers_seen[tVal];
          continue;
        }

        split[index].push_back(*iter);
     }

    // Sort each of the patitions
    for(unsigned int i=0;i<barrier_count;i++){
        split[i].sort(rr_compare);
        sort(split[i]);
    }
    
    //combine partitions togeth
    trace->entries.clear();
    for(unsigned int i=0;i<barrier_count;i++){
      trace->entries.splice(trace->entries.end(),split[i]);
    }

   delete[] split;
 }



}