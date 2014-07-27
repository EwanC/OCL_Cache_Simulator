#include "trace.h"
#include "parse.h"

/*
  Takes file line in the and converts it at a Trace_entry object
  defined in 'trace.h'

  file line in the form:
  'memory address,read/write,instruction | Thread id | loop data'
   -------------- ---------- -----------   ----------  ---------
    8 characters    1 char    7 char        16 chars    16 chars
*/
bool parseInput(std::string line,Trace& trace){

  // Consecutive hypens indicate end of the trace.
  if(line.find("-") < line.length()){
   	return true;
  }

  // Creates Trace_entry object to represent to current file line
  Trace_entry curr_entry;

  // Populate loop data in the entry object.
  getLoops(line,curr_entry);

  // Populate Thread Id in entry object.
  getThreadIds(line,curr_entry,trace); 

  if(line.find("|") <= 1){         // Checks if file line signals a memory barrier.
	  curr_entry.setBarrier(true);
  }else{                           // Otherwise the file line is a memory access.

    // Populate memory address data in the entry object.
    getMemoryAddress(line,curr_entry); 

    // Populate instuction name in the entry object.
    getInstruction(line,curr_entry);  
  }
  
  // Give the entry a pointer to Trace object
  curr_entry.setPointer(&trace);

  // Add entry to linked list field of Trace object.
  trace.entries.push_back(curr_entry);

  return false;
}


/*
  Populates loop data in Trace_entry object
  from data after second '|' 
*/
void getLoops(std::string line,Trace_entry& curr){
   
  size_t pos = 0; // Position of last '|'

  // Get substring containing loop data
  while((pos = line.find("|")) != std::string::npos){
    line.erase(0,pos+1);
  }

  // If access was not inside a loop return
  if(strtol(line.c_str(),NULL,16) == 0){
    curr.setLoopDepth(0);
    return;
  }

  // Copy line into char*
  int length = line.length();   
  char str_cpy[length];
  strcpy(str_cpy,line.c_str());
   
  // Attributes that need to be populated
  unsigned int label[maxLoops],loop_val[maxLoops];
  unsigned int loop_num = 0;
  
   
  char* substr;        // Substring for label bits
  unsigned int hex= 0; // Conversion to hex from string

  //For each possible nested loop
  for(int i=maxLoops-1; i >= 0; --i){
    if(i==2){
     	// Gets loop data of the 3rd nested loop, if one exists, from the first 20 bits
      substr = str_cpy + (length-5);
      hex = strtol(substr,NULL,16);  //convert hex string to int
    }
    if(i==1){
      // Gets loop data of second level nested loop, if exists, from bits 20 - 39
      *substr = '\0';
      substr = str_cpy + (length - 10);
      hex = strtol(substr,NULL,16);  //convert hex string to int
    }
    if(i==0){
      // Gets loop data, from top level loop, from bits 40 - 59
      *substr = '\0';
      substr = str_cpy;
      hex = strtol(substr,NULL,16); //convert hex string to int
    }

    // Label is first 4 bits
    label[i] = (0xF << 16) & hex;
    label[i]= label[i] >> 16;
   
    // Loop value is next 16 bits 65536 = 2^16 
    loop_val[i] = (65535) & hex;
   
    // If loop value is non zero update the number of loops the instuction is in
    if(loop_val[i] != 0)
      loop_num++;

  }

  // Assign data to object
  curr.setLoopDepth(loop_num);
  curr.pushLoopIter(label[0],loop_val[0]);
  curr.pushLoopIter(label[1],loop_val[1]);
  curr.pushLoopIter(label[2],loop_val[2]);
 
 
}

/*
  Get the memory address of the line
*/
void getMemoryAddress(std::string line,Trace_entry& curr){
  
  // Get first 8 characters as hex digits
   line.erase(line.find("|"));
  if(line.length() < 16){
     line.insert(0,std::string(16-line.length(),'0'));
  }

  unsigned int addr = strtoul(line.substr(0,8).c_str(),NULL,16);

  // Converts string to base 16 int and stores address in struct
  curr.setMemAddr(addr); 

} 

/*
  Get the whether the memory access was a read or write
  from 9th bit, and decode instruction name from last 7 
  bits before first '|'
*/
void getInstruction(std::string line,Trace_entry& curr){
  line.erase(line.find("|"));


  line = line.substr(line.length()-8);

  // Read or write access type
  if(line[0] == 'F')
      curr.setRead(true);
  else
      curr.setRead(false);

  line = line.substr(1);

  // Convert string from base 16 to base 10 
  long int as_int = strtol(line.c_str(),NULL,16); 
                
  // Decode from radix to get character string from integer
  curr.setName(as_int);


} 

/*
  Populate thread id data in Trace_entry struct
*/
void getThreadIds(std::string line, Trace_entry& curr,Trace& trace){
     
  // Get characters between two '|' 
  line.erase(0,line.find("|")+1);
  size_t pos = line.find("|");
  line.erase(pos,line.length()-pos);

  // Only one dimension
  if(line.length() <= 5){
    unsigned int hex =strtol(line.c_str(),NULL,16);      //convert string id to hex int
    curr.setThreadIds(hex,0,0);

    /*
      if id is greater than number of threads then set the maximum
      number of threads in the trace to the id
    */
    if(hex >= trace.getGlobal(0)){
       trace.setGlobalSize(0,hex+1);
    }    
 
    	
  } 
  else if(line.length() <= 10){ // Two dimensions
    unsigned int length = line.length();
    std::string id1,id2;
    id2 = line.substr(0,length-5);
    id1 = line.substr(length-5,5);
     
    unsigned int hex1 =strtol(id1.c_str(),NULL,16);

    if(hex1 >= trace.getGlobal(0)){ 
      trace.setGlobalSize(0, hex1+1);
    }

    //get second dimension id
    unsigned int hex2 =strtol(id2.c_str(),NULL,16);
    if(hex2 >= trace.getGlobal(1)){
      trace.setGlobalSize(1,hex2+1);
    }

    curr.setThreadIds(hex1,hex2,0);

   }
   else{ //Three dimensions
     unsigned int length = line.length();
     std::string id1,id2,id3;
     id3 = line.substr(0,length-10);
     id2 = line.substr(length-10,5);
     id1 = line.substr(length-5,5);
      
     unsigned int hex1 = strtol(id1.c_str(),NULL,16);
     if(hex1 >= trace.getGlobal(0)){ 
       trace.setGlobalSize(0, hex1+1);
     }


     //get second dimension id
     unsigned int hex2 =strtol(id2.c_str(),NULL,16);
     if(hex2 >= trace.getGlobal(1)){
       trace.setGlobalSize(1,hex2+1);

     }

     //get third dimension id
     unsigned int hex3 =strtol(id3.c_str(),NULL,16);
     if(hex3 >= trace.getGlobal(2)){
       trace.setGlobalSize(2,hex3+1);
     }

     curr.setThreadIds(hex1,hex2,hex3);

   }
  
} 

