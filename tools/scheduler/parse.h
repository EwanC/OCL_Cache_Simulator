#ifndef PARSE_H
#define PARSE_H

// Converts a file line into a Trace_entry object.
void parseInput(std::string line,Trace& trace);

// Populate loop data in the Trace)entry object.
void getLoops(std::string line,Trace_entry& curr);

// Populate memory address data in the entry object.
void getMemoryAddress(std::string line,Trace_entry& curr); 

// Populate instuction name in the entry object.
void getInstruction(std::string line,Trace_entry& curr);  

// Populate Thread Id in entry object.
void getThreadIds(std::string line, Trace_entry& curr,Trace& t); 

unsigned int decodeRadix(int input);

#endif //PARSE_H