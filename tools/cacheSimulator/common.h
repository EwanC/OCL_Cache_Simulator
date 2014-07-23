#ifndef COMMON_H
#define COMMON_H

#include <vector>

std::vector<unsigned int>get_workgroups(unsigned int warp_size,unsigned int total_wk);


const unsigned int CORES = 15;  //Number of cores on GTX 480 architecture simulated against

unsigned int ceiling(unsigned int a, unsigned int b);

class Entry{
 public:
	Entry(unsigned int addr,bool _op,unsigned int wk,unsigned int warp,unsigned int i):
	      address(addr),op(_op),wk_id(wk),warp_id(warp),inst(i) {}
	
	unsigned int address;
	bool op;
	unsigned int wk_id;
	unsigned int warp_id; 
	unsigned int inst;

};

#endif