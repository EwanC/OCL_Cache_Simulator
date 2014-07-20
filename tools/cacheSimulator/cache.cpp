#include "cache.h"
#include "stats.h"
#include "cache.h"


  
//counter for number of accesses processed from warp
static unsigned int warp_counter = 0;

//Thread id of last memory access
static int last_id = 0;

//instruction identifier of last memory access
static int last_inst = 0;


/*
 * Initialize a new cache line with a given line size.
*/
CacheLine::CacheLine()
{
    state = INVALID;
    ctr = 0;
}


/*
 * Initialize a new cache set with the given associativity and line size.
*/
CacheSet::CacheSet(unsigned int associativity)
{

    //allocate space for all the lines in the set
    lines = new CacheLine*[associativity];

   

    for (unsigned int i = 0; i < associativity; i++){
        
        lines[i] = new CacheLine(); 
    }
}



/*
 * Compute the shift and mask given the number of bytes/sets.
*/
static void get_shift_and_mask(int value, unsigned int *shift, unsigned int *mask, int init_shift)
{
    *mask = 0;
    *shift = init_shift;

    while (value > 1){
        (*shift)++;
        value >>= 1;
        *mask = (*mask << 1) | 1;
    }
}



/*
 * Create a new cache that contains a total of num_lines lines, each of which is line_size
 * bytes long, with the given associativity, and the given set of cache policies for replacement
 * and write operations.
 */
Cache::Cache(unsigned int num_lines, unsigned int lineSize, unsigned int assoc, unsigned int repPolicy, unsigned int writePolicy,unsigned int wSize)
{

    /*
     * Create the cache and initialize constant fields.
    */
    
    stats = Stats();
    

    /*
     * Initialize size fields.
    */
    write_policy = writePolicy;
    replacement_policy = repPolicy;
    line_size = lineSize;
    associativity = assoc;
    num_sets = num_lines / assoc;
    warp_size = wSize;


    /*
     * Initialize shifts and masks.
    */
    get_shift_and_mask(line_size, &cache_index_shift, &line_offset_mask, 0);
    get_shift_and_mask(num_sets, &tag_shift, &cache_index_mask, cache_index_shift);

    /*
     * Initialize cache sets.
    */
    
     for (unsigned int i = 0; i < num_sets; i++){
         
         sets.push_back(CacheSet (associativity));
     }

  
}

/*
 * Determine whether or not a cache line is valid for a given tag.
*/
static int cache_line_check_validity_and_tag(CacheLine* line, intptr_t tag)
{

     return (line->state != 0 && line->tag == tag );       
}


/*
 * Move the cache lines inside a cache set so the cache line with the given index is
 * tagged as the most recently used one. The most recently used cache line will be the 
 * 0'th one in the set, the second most recently used cache line will be next, etc.
 * Cache lines whose valid bit is 0 will occur after all cache lines whose valid bit
 * is 1.
*/
static CacheLine* cache_line_make_mru(CacheSet& cache_set, size_t line_index)
{
    CacheLine *line = cache_set.lines[line_index];
    
    for (int i = line_index - 1; i >= 0; i--){
        cache_set.lines[i + 1] = cache_set.lines[i];
    }

    cache_set.lines[0] = line;
    
    return line;
}

/*
 * Retrieve a matching cache line from a set, if one exists.
*/
static CacheLine* cache_set_find_matching_line(const Cache& cache, CacheSet& cache_set, intptr_t tag)
{

     for(unsigned int i=0; i<cache.associativity;i++){
           CacheLine* curr_line = cache_set.lines[i];

           if(cache_line_check_validity_and_tag(curr_line, tag)){         
                return cache_line_make_mru(cache_set, i);             
           }
     }
     
     return NULL;  
}

/*
 * Finds the index of the line which has been least frequently accessed.
*/
static int find_LFU_line(CacheSet& cache_set, int assoc){

 int min_ctr = 100000;
 int index = 0;
 for(int i=0;i<assoc;i++){
   if(cache_set.lines[i]->ctr < min_ctr){
     min_ctr = cache_set.lines[i]->ctr;
     index = i;
   }
 }

 return index;

}

/*
 * Function to find a cache line to use for new data.
*/
static CacheLine *find_available_cache_line(const Cache& cache, CacheSet& cache_set)
{

     unsigned int N = cache.associativity;    

     //Least recently used replacement
     if (cache.replacement_policy == CACHE_REPLACEMENTPOLICY_LRU){  
        return cache_line_make_mru(cache_set,(N-1));
     }
     //Most recently used replacement
     else if (cache.replacement_policy == CACHE_REPLACEMENTPOLICY_MRU){                           //For MRU replacement policies 
         return cache_set.lines[0];
     }
     //Least frequently used replacement
     else if (cache.replacement_policy == CACHE_REPLACEMENTPOLICY_LFU){    //For LFU replacement
         int i = find_LFU_line(cache_set,N);
         return cache_line_make_mru(cache_set,i);
      }     
      
     //Random replacement
     int randomIndex = rand() % N;
     return cache_line_make_mru(cache_set,randomIndex);       //For random replacement policies
}

/*
 * Add a line to a given cache set.
 */
static CacheLine *cache_set_add(const Cache& cache, CacheSet& cache_set, intptr_t address, intptr_t tag)
{
    /*
     * First locate the cache line to use.
     */
    CacheLine *line = find_available_cache_line(cache, cache_set);
   
    /*
     * Now set it up.
     */
    line->tag = tag;
    line->state = CacheLine::VALID;
    line->ctr=0;
    
    return line;
}

/*
 *  Cache write from thread t_id, at instruction inst to address 
 */
void cache_write(Cache& cache, unsigned long address,int warp_id, int inst){
   
    unsigned int warp_size = cache.warp_size;

    /*
       Get shifts and masks
    */
    unsigned int index_mask = cache.cache_index_mask;
    unsigned int index_shift = cache.cache_index_shift; 
    unsigned int tag_shift = cache.tag_shift;  

    //get set index, tag, and line offest from address
    int set_index = (address >> index_shift) & index_mask;
    intptr_t tag = address >> tag_shift;

  
    //find cache set of access
    CacheSet cache_set = cache.sets.at(set_index);
    
    //find if there is a matching cache line in cache
    CacheLine *matching_line = cache_set_find_matching_line(cache,cache_set,tag);
    
    //update find stack distance of cache line
    unsigned int stack_dist = cache.stats.stackRef(tag,set_index);
  
   
    //CASE: Write through, no allocate
    if(cache.write_policy == CACHE_WRITEPOLICY_WTNA){
       if(matching_line == NULL){   //Write Miss

         //if cache line hasn't been accessed this warp
         if(!(warp_counter > stack_dist)){                
            cache.stats.incrementWrites();
            cache.stats.incrementWriteMisses();
         }

        }
        else{                                            //Write hit
        matching_line->ctr = matching_line->ctr + 1;
         
      
        //if cache line hasn't been accessed this warp
        if(!(warp_counter > stack_dist)){
              cache.stats.incrementWrites();
        }
       }
    }
    //CASE: Write back allocate
    else{
       if(matching_line == NULL){                    //Write miss
        
         //if cache line hasn't been accessed this warp
         if(!(warp_counter > stack_dist)){      
            cache.stats.incrementWrites();
            cache.stats.incrementWriteMisses();
            }

         //find line for write
         matching_line = cache_set_add(cache,cache_set,address, tag);
         
         //Line is dirty, needs to be written back to memory
        if(matching_line->state == CacheLine::MODIFIED){
             cache.stats.incrementWriteBacks();
        }

       }
       else{   //Write hit
        
          //if cache line hasn't been accessed this warp
          if(!(warp_counter > stack_dist)){
               cache.stats.incrementWrites();
          }
         
          matching_line->ctr = matching_line->ctr + 1;
       
       }
       matching_line->state = CacheLine::MODIFIED;            //Set to dirty
    }
     
    //increment counter, resetting when all warp accesses have been made 
    if(warp_counter >= warp_size-1 || (warp_counter != 0 &&(warp_id != last_id ||  last_inst != inst)))
      warp_counter=0;
    else
      warp_counter++;
 
    //update details of previous access
    last_inst = inst;
    last_id = warp_id;   
}

/*
 * Cache read from thread t_id from instruction inst to address
 */
void cache_read(Cache& cache, unsigned long address,int warp_id,int inst)
{

    unsigned int warp_size = cache.warp_size;

    /*
       Setup shifts and masks
    */
    unsigned int index_mask = cache.cache_index_mask;
    unsigned int index_shift = cache.cache_index_shift; 
    unsigned int tag_shift = cache.tag_shift;  

   /*
     use address to get line offset, tag, and set index
   */
    int set_index = (address >> index_shift) & index_mask;
    intptr_t tag = address >> tag_shift;
    
    //finds cache set of access
    CacheSet cache_set = cache.sets.at(set_index);
    
    //finds matching line in cache set
    CacheLine *matching_line = cache_set_find_matching_line(cache,cache_set,tag);
    
    //finds stack distance of cache line and updates stack distance histogram
    unsigned int stack_dist = cache.stats.stackRef(tag,set_index);
    
    //CASE: Write through no-allocate
    if(cache.write_policy == CACHE_WRITEPOLICY_WTNA){
        if(matching_line == NULL){                         //Read miss
           matching_line = cache_set_add(cache,cache_set,address, tag);
      

          //if line has not been accessed this warp
          if(!(warp_counter > stack_dist)){
              cache.stats.incrementReads();
              cache.stats.incrementReadMisses(stack_dist,cache.num_sets * cache.associativity);
          }

        }
        else{    //Read hit
          
          //if line has not been accessed this warp
          if(!(warp_counter > stack_dist)){
             cache.stats.incrementReads();
          }
          
           matching_line->ctr = matching_line->ctr + 1;
        }
    }
    //CASE: Write back allocate
    else{
        if(matching_line == NULL){                     //Read miss
         
        
         //if line has not been accessed this warp
         if(!(warp_counter > stack_dist)){
           cache.stats.incrementReads();
           cache.stats.incrementReadMisses(stack_dist,cache.num_sets * cache.associativity);
         }
        
         matching_line = cache_set_add(cache,cache_set,address, tag);
        
         //cache line is dirty and needs to be written back
         if(matching_line->state == CacheLine::MODIFIED){
              cache.stats.incrementWriteBacks();
         }

           matching_line->state = CacheLine::VALID;              //Set line to valid
        }
        else{     
         
          //if line has not been accessed this warp
          if(!(warp_counter > stack_dist)){
            cache.stats.incrementReads();
          }

          matching_line->ctr = matching_line->ctr + 1;
        }
    }

    //update warp counter, resetting if all warp accessed have been made
    if(warp_counter >= warp_size-1 || (warp_counter != 0 &&(warp_id != last_id ||  last_inst != inst))){
      warp_counter=0;
    }
    else
      warp_counter++;
    
    //update details of previous access
    last_inst = inst;
    last_id = warp_id;

}

