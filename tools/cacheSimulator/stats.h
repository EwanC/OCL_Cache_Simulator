#ifndef STATS_H
#define STATS_H

#include <cstdint>
#include <stack>
#include <iostream>


const unsigned int Infinity = 2000000000;


//reuse stack entry 
class StackEntry
{
  public:
    intptr_t tag;    //cache line tag
    int set;         //cache line set
    
};

bool operator== (const StackEntry& S1,const StackEntry &S2);

// Class for holding stats about cache performance.
class Stats
{  
   private:


    int reads;                          // Number of reads 
    int readMisses;                     // Number of read misses
    int writes;                         // Number of writes
    int writeMisses;                    // Number of write misses
    int writeBacks;                     // Number of write backs
    int coldMisses;                     // Number of cold misses   
    int capacityMisses;                 // Number of capactiy misses
    int conflictMisses;                 // Number of conflict misses
    std::stack<StackEntry> stack;       //cache line reuse distance stack

   public:

   Stats();
   
   friend std::ostream & operator<< (std::ostream & os, const Stats& right);

   void Write();

   void incrementReads();
   void incrementReadMisses(unsigned int,unsigned int);
   void incrementWrites();
   void incrementWriteMisses();
   void incrementWriteBacks();
   int getNumAccess()const;
   double getReadMissRate()const;
   double getWriteMissRate()const;
   double getTotalMissRate()const;

   unsigned int stackRef(intptr_t tag,int set);


};



#endif //STATS_H