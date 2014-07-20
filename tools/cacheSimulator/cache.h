/*
 * cache.h
 *
 * Definition of the structure used to represent a cache.
 */
#ifndef CACHE_H
#define CACHE_H
#include <cstdlib>
#include "stats.h"
#include <vector>

/*
 * Replacement policies.
 */

const unsigned int CACHE_REPLACEMENTPOLICY_LRU =  0;  //LEAST RECENTLY USED
const unsigned int CACHE_REPLACEMENTPOLICY_RANDOM =1;  //RANDOM REPLACEMENT
const unsigned int CACHE_REPLACEMENTPOLICY_MRU =2;    //MOST RECENTLY USED
const unsigned int CACHE_REPLACEMENTPOLICY_LFU =3;     //LESASR FREQUENTLY USED

/*
 * Write policies.
 */

const unsigned int CACHE_WRITEPOLICY_WBWA   = 0;       //WRITE BACK WRITE ALLOCATE
const unsigned int CACHE_WRITEPOLICY_WTNA   = 1;        //WRITE THROUGH NO-ALLOCATE


//Class used to store a single cache line.
class CacheLine
{
  public:

   CacheLine();

   enum State{
      INVALID =0,         
      VALID,
      MODIFIED             //Dirty bit, line has been modified 
   };

    CacheLine::State state;        // State of cache line.

    intptr_t tag;            // The tag. 

    int ctr;                 // Counter used to implement LFU replacement,
                             // incremented on every access.

};


/*
 * Class used to store a cache set: a cache set contains a pointer
 * to an array of pointers to cache lines.
*/
class CacheSet
{
  public:
    CacheSet(unsigned int associativity);

    CacheLine** lines;

};

//Class used to store a cache.
class Cache
{ 

	
  public:

    Cache(unsigned int num_lines, unsigned int line_size, unsigned int associativity, unsigned int rep_policy, unsigned int write_policy,unsigned int w_size);
    
    unsigned int num_sets;             // Number of sets in the cache. 

    unsigned int associativity;        // Number of lines in each set. 

    unsigned int line_size;                  // Size of each line. 

    unsigned int line_offset_mask;     // Mask for line offset. 

    unsigned int cache_index_mask;     // Mask for cache index. 

    unsigned int cache_index_shift;    // Shift for cache index and tag. 

    unsigned int tag_shift;            // Shift for tag. 

    unsigned int replacement_policy;   // Replacement policy. 

    unsigned int write_policy;         // Write policy. 

    std::vector<CacheSet> sets;        // Array of sets, each of which is an array of blocks,
                                       // each of which is an array of bytes. 
    Stats stats;              // Statistics about the cache accesses

    unsigned int warp_size;            // Size of a warp 

};




/*
 * Read a single integer from the cache.
 */
void cache_read(Cache& cache, unsigned long address,int t_id, int inst);

/*
 * Write a single integer to memory and/or the cache.
 */
void cache_write(Cache& cache, unsigned long address,int t_id, int inst);


unsigned int ceiling(unsigned int a, unsigned int b);


#endif