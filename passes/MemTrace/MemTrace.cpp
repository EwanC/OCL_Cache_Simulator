/*

Copyright 2014 Ewan Crawford<ewan.cr@gmail.com>


This file is part of OpenCL Visuliser.

OpenCL Visuliser is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OpenCL Visuliser is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OpenCL Visuliser.  If not, see <http://www.gnu.org/licenses/>
*/

#include "trace.h"

using namespace llvm;
namespace {

 struct Loop {
 

   Value* counter; //pointer to loop iteration counter
   BasicBlock* exit; 
   int id;

 };


   struct MemTrace : public ModulePass {
   static char ID; 

   std::map<BasicBlock*,Loop> loop_map; 

   int inst_ctr;                //counts number of memory access instructions
   int loop_ctr;                //counts number of loops

   Function *thread_id_hook;    //Function hook for global thread id
   Function *atomicINC_hook;    //Function hook for atomic_inc with global param
   Function *barrier_hook;      //Function hook for opencl barrier

   Type* int32_Ty;              //32 bit int type
   Type* int64_Ty;              //64 bit int type

  
   MemTrace() : ModulePass(ID) {} 

    virtual bool runOnModule(Module &M){    

      //Initalises information about block structure
      init_pass(M);
      inst_ctr=1;
      Module::iterator A;
      Module::iterator E;
      
      for( A = M.begin(),E = M.end(); A!= E; ++A){
        loop_ctr=0;
        MemTrace::runOnFunction(A); 
      }
      return false;  
    }

    virtual bool runOnBasicBlock(Function::iterator &bb) {  
      //Check if basic block is a loop body
      if(strstr(bb->getName().data(),".body")){
        //checks if this basic block has been added before
        if(loop_map.find(bb) == loop_map.end()){
          loop_map[bb].id = loop_ctr++;
                    
          //Finds basic block predecessor
          BasicBlock* pred  = bb->getUniquePredecessor();
          BasicBlock::iterator I;
          BasicBlock::iterator E;

          //Finds end basic block for selected loop by checking branches
          for(I = pred->begin(),E=pred->end();I!=E;I++){
              Instruction* i = &*I;
              if(isa<BranchInst>(i)){
                int n = cast<BranchInst>(i)->getNumSuccessors();
                for(int itr=0;itr<n;itr++){
                  BasicBlock* curr = cast<BranchInst>(i)->getSuccessor(itr);
                  if(strstr(curr->getName().data(),".end")){
                    loop_map[bb].exit = curr;
                    break;
                  }
                }
                break;
              }

          }
        }  
      }
      return false;
  }


  /*
     Looks at each function
  */
  virtual bool runOnFunction(Module::iterator &F) {  

    Value* trace_arg;            //'trace' argument storing mem address and instruction name
    Value* t_id_arg;             //'ids'   argument storing the string global id
    Value* loop_arg;             //'loop'  argument storing loops and their iterations
  

    loop_map.clear();            //Forget about loops in other kernel functions

    inst_iterator I,E;  
    Function::arg_iterator a =  F->getArgumentList().begin(); 
    Function::arg_iterator z =  F->getArgumentList().end();  

    //Find all athe dded parameters by searching kernel parameters
    for(; a!=z; a++){
      if( a->getName().equals("trace")){  
        trace_arg = cast<Value> (a);    
      }
      else if(a->getName().equals("ids")){
        t_id_arg = cast<Value>(a);
      }
      else if( a->getName().equals("loop_ctr")){  
        loop_arg = cast<Value> (a);    
      }            
    } 

    //Identify loops in basic blocks
    for(Function::iterator func_itr = F->begin(), func_end = F->end(); func_itr != func_end; ++func_itr){
      MemTrace::runOnBasicBlock(func_itr);
    }

    create_loop_counters(F->begin());   
    inc_loop_counters(F);
    reset_loop_counters(F);

    //for each instruction
    for(I = inst_begin(F), E = inst_end(F); I != E;){    
      Instruction *i = &*I++;                           
      IRBuilder<> builder(i);

      //if instruction is load or store
      if((isa<StoreInst>(i)) || (isa<LoadInst>(i)) ){  
        if(is_global(i)){  //if instrcution access global memory
         
          //get pointer to counter
          Value *GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(builder.getInt32(0)));
          GEP = builder.CreateBitCast(GEP,PointerType::get(int32_Ty,1));
         
          //atomically load and increment the trace index
          Value *counter= builder.CreateCall(atomicINC_hook,GEP,"atomic_inc");

          //add trace information
          add_trace(i,counter,trace_arg,t_id_arg,loop_arg);

          inst_ctr++;
        }
      }
      //Check for memory barriers
      else if((isa<CallInst>(i))){
        Function* F = cast<CallInst>(i)->getCalledFunction();
        if((F->getName().equals("barrier"))  || (F->getName().equals("mem_fence")) || (F->getName().equals("work_group_barrier")) ){
          mem_fence(i,trace_arg,t_id_arg,loop_arg);
        }
      }
    }

    return true;                                                      
  }

  //Add memory barrier pos to trace
  void mem_fence(Instruction* i,Value* trace_arg,Value* t_id_arg,Value* loop_arg){
    IRBuilder<> builder(i);

    //get pointer to counter
    Value *GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(builder.getInt32(0)));
    GEP = builder.CreateBitCast(GEP,PointerType::get(int32_Ty,1));
               
    //atomically load and increment the trace index
    Value *counter= builder.CreateCall(atomicINC_hook,GEP);
    Value* arg = cast<CallInst>(i)->getArgOperand(0);
    arg  = builder.CreateZExt(arg,int64_Ty);
    GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(counter),"mem_fence"); 
    builder.CreateStore(arg,GEP);

    //get id of current thread in first dimension
    Value* dim_id = builder.CreateCall(thread_id_hook,builder.getInt32(0),"dim0");
    Value* thread_id  = builder.CreateZExt(dim_id,int64_Ty);

    //get id of current thread in second dimension
    dim_id = builder.CreateCall(thread_id_hook,builder.getInt32(1),"dim1");
    dim_id  = builder.CreateZExt(dim_id,int64_Ty);
    dim_id = builder.CreateShl(dim_id,20,"dim_sl",false,false);
    thread_id= builder.CreateOr(dim_id,thread_id);

     //get id of current thread in third dimension
    dim_id = builder.CreateCall(thread_id_hook,builder.getInt32(2),"dim2");
    dim_id  = builder.CreateZExt(dim_id,int64_Ty);
    dim_id = builder.CreateShl(dim_id,40,"dim_sl",false,false);
    thread_id= builder.CreateOr(dim_id,thread_id);
   
    GEP = builder.CreateInBoundsGEP(t_id_arg,ArrayRef<Value*>(counter),"mem_fence"); 
    builder.CreateStore(thread_id,GEP);
  }


  //checks that memory access is to global memory
  bool is_global(Instruction* i){
    Value* pointer_op;
    if(isa<StoreInst>(i)){
      pointer_op= cast<StoreInst>(i)->getPointerOperand();
    }
    else{
      pointer_op= cast<LoadInst>(i)->getPointerOperand();
    }

   Type *addr_space = (pointer_op->getType());

   if(cast<PointerType>(addr_space)->getAddressSpace() == 1)
      return true;
   else
      return false;
  
  }


  //adds trace information about instuction i to the trace parameter at specified index
  void add_trace(Instruction* i, Value* index,Value* trace_arg, Value* t_id_arg,Value* loop_arg){
    Value* addr;
    IRBuilder<> builder(i);

    StringRef name;
    
    char *inst_name = (char*)malloc(sizeof(char) * 10);

    //Rename instruction for it easier to process later
    sprintf(inst_name,"M%d",inst_ctr);
    if((isa<StoreInst>(i))) {

      //convert memory address to 64 bit and shift left 4 bits to make room for R/W indicator 
      addr = builder.CreatePointerCast(cast<StoreInst>(i)->getPointerOperand(),int64_Ty,inst_name);
      name = addr->getName();
      addr = builder.CreateShl(addr,32,"addr_sl",false,false);
     
      Value* type = builder.getInt64(10);

      type= builder.CreateShl(type,28);
      addr = builder.CreateOr(addr,type);

    }
    else{ //load instruction
      //sets inst name so we know which load this trace elemnent came form
      cast<Value>(i)->setName(inst_name);
      name = i->getName();
     
      //get address
      addr = builder.CreatePointerCast(cast<LoadInst>(i)->getPointerOperand(),int64_Ty);
      addr = builder.CreateShl(addr,32,"addr_sl",false,false);

      Value* type = builder.getInt64(15);

      type= builder.CreateShl(type,28);
      addr = builder.CreateOr(addr,type);
    }

    //use a radix to encode the name
    Value* name_val= builder.getInt64(inst_ctr);

    //combine first 31 bits with second 33
    Value* trace_item = builder.CreateOr(addr,name_val,"trace_arg");

    //get pointer to index in trace, i.e trace[i]
    Value *GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(index),"trc"); 
    builder.CreateStore(trace_item,GEP);

    //get id of current thread
    Value* dim_id = builder.CreateCall(thread_id_hook,builder.getInt32(0),"dim0");
    Value* thread_id  = builder.CreateZExt(dim_id,int64_Ty);

    dim_id = builder.CreateCall(thread_id_hook,builder.getInt32(1),"dim1");
    dim_id  = builder.CreateZExt(dim_id,int64_Ty);
    dim_id = builder.CreateShl(dim_id,20,"dim_sl",false,false);
    thread_id= builder.CreateOr(dim_id,thread_id);

    dim_id = builder.CreateCall(thread_id_hook,builder.getInt32(2),"dim2");
    dim_id  = builder.CreateZExt(dim_id,int64_Ty);
    dim_id = builder.CreateShl(dim_id,40,"dim_sl",false,false);
    thread_id= builder.CreateOr(dim_id,thread_id);
   
    GEP = builder.CreateInBoundsGEP(t_id_arg,ArrayRef<Value*>(index),"id_trc"); 
    builder.CreateStore(thread_id,GEP);

    //Add loop counter info
    Value* loop_val= builder.getInt64(0);
    Value* tmp;
    Value* tmp2;
    Value* tmp3;
    Value* mask = builder.getInt1(1);
    Value* loop_offset = builder.getInt64(3);
  
    //Loop over all the loop counters
    std::map<BasicBlock*, Loop>::iterator iter;
    for(iter = loop_map.begin(); iter != loop_map.end(); iter++) {
        
      //Cacluate mask if loop counter is zero
      Value* loop_var = builder.CreateLoad(iter->second.counter);
      mask = builder.CreateICmpNE(loop_var,builder.getInt64(0));
      mask = builder.CreateSExt(mask,int64_Ty);
      mask = builder.CreateMul(mask,builder.getInt64(-1));

      //find offset in kernel buffer
      tmp = builder.CreateICmpNE(loop_var,builder.getInt64(0));
      tmp = builder.CreateZExt(tmp,int64_Ty);
      loop_offset = builder.CreateSub(loop_offset,tmp);

      //find counters offset
      tmp2 = builder.CreateMul(loop_offset,builder.getInt64(20));
      tmp  = builder.CreateShl(loop_var,tmp2);

      //find loop index offset
      tmp2 = builder.CreateAdd(tmp2,builder.getInt64(16));
      //AND index with offset
      tmp3 = builder.getInt64(iter->second.id);
      tmp3 = builder.CreateAnd(tmp3,mask);
      tmp3 = builder.CreateShl(tmp3,tmp2);

      //combine label and offset with existing buffer
      tmp3 = builder.CreateOr(tmp3,tmp);
      loop_val = builder.CreateOr(loop_val,tmp3);
   }     
   
   //Store loop offset back in buffer
   GEP = builder.CreateInBoundsGEP(loop_arg,ArrayRef<Value*>(index),"LOOP"); 
   builder.CreateStore(loop_val,GEP);

  }

  void create_loop_counters(BasicBlock *bb){
   
    Instruction* i = bb->begin();
    IRBuilder<> builder(i);
   
           
    //Create loop counter variable
    for(int k=0;k<loop_ctr;k++){
      Value *loop_var = builder.CreateAlloca(int64_Ty);   
      cast<AllocaInst>(loop_var)->setAlignment(8);
      Value* GEP = builder.CreateInBoundsGEP(loop_var,ArrayRef<Value*>(builder.getInt64(0)));
      builder.CreateStore(builder.getInt64(0),GEP);
      for(std::map<BasicBlock*,Loop>::iterator iter = loop_map.begin();
          iter != loop_map.end();++iter){
          if(iter->second.id == k)
           iter->second.counter = GEP;
      }
    }

  }

  //Add loop counter incrementation instruction
  void inc_loop_counters(Module::iterator &F){
    Function::iterator func_itr,func_end;
    for( func_itr= F->begin(), func_end = F->end(); func_itr != func_end; ++func_itr){
       BasicBlock* bb = &*func_itr;
       if(loop_map.find(bb) != loop_map.end()){
          IRBuilder<> builder(bb->begin());
          Value* inc = builder.CreateLoad(loop_map[bb].counter);
          inc = builder.CreateNSWAdd(inc,builder.getInt64(1));
          builder.CreateStore(inc,loop_map[bb].counter);
       }
    }
  }
  
  //Reset loop counter to zero after termination basic block is seen
  void reset_loop_counters(Module::iterator &F){

    Function::iterator func_itr,func_end;
    std::map<BasicBlock*,Loop>::iterator map_itr,map_end; 
    
    for(func_itr = F->begin(), func_end = F->end(); func_itr != func_end; ++func_itr){
       BasicBlock* bb = &*func_itr;
       for(map_itr = loop_map.begin(),map_end = loop_map.end();map_itr != map_end;++map_itr){
          if(map_itr->second.exit == bb){
            IRBuilder<> builder(bb->begin());
            builder.CreateStore(builder.getInt64(0),map_itr->second.counter);
          }
       }
    }
  }

  //sets inital information
  void init_pass(Module &M){


    int32_Ty = Type::getInt32Ty(M.getContext()); 
    int64_Ty = Type::getInt64Ty(M.getContext()); 

    //Sets hook to get_global_id() function, for thread id
    Constant *hookFunc;
    std::vector<Type *> params;
    params.push_back(int32_Ty);

    FunctionType *MTy=FunctionType::get(int32_Ty,params,false);
    hookFunc = M.getOrInsertFunction("get_global_id",MTy);

    thread_id_hook = cast<Function>(hookFunc);
    thread_id_hook->addFnAttr(Attribute::NoBuiltin);

    //Sets hook to atomic_inc function with __global param
    Constant *hookInc;
    std::vector<Type *> inc_params;
    inc_params.push_back(PointerType::get(int32_Ty,1));

    MTy=FunctionType::get(int32_Ty,inc_params,false);
    hookInc = M.getOrInsertFunction("atomic_inc",MTy);
    atomicINC_hook = cast<Function>(hookInc);
    atomicINC_hook->addFnAttr(Attribute::NoBuiltin);

    }

  };
} //end of namespace

char MemTrace::ID = 0;
static RegisterPass<MemTrace> X("trace", "Memory Trace");  //registers trace so '-trace' can be used to execute the pass
