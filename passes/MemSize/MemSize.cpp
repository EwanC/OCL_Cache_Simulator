//
// Memory size pass for OpenCL programs
// Ewan Crawford     s1032792
// 

#include "trace.h"

using namespace llvm;
namespace {

 struct Loop {
 

   Value* counter; //pointer to loop iteration counter
   BasicBlock* exit; 
   int id;

 };

  


  struct MemSize : public ModulePass {
    static char ID; 

   std::map<BasicBlock*,Loop> loop_map; 

   //Has this called function been seen before
   std::map<Function*, int> seen;

   //Function call hooks
   Function *thread_id_hook;
   Function *atomicINC_hook;

   std::vector<Instruction*>toDelete;

   Type* int32_Ty;    //32 bit int type
   Type* int64_Ty;    //64 bit int type
   int loop_ctr;      //counter the number of loops

   MemSize() : ModulePass(ID) {} 

   virtual bool runOnModule(Module &M){    

      //Initalises information about block structure
      init_pass(M);
      Module::iterator A;
      Module::iterator E;

      for( A = M.begin(),E = M.end(); A!= E; ++A){
        loop_ctr=0;
        MemSize::runOnFunction(A); 
      }


      while(!toDelete.empty()){
        toDelete.back()->eraseFromParent();
        toDelete.pop_back();
      }

      return false;  
   }

    //checks each basic block to see if it is a loop body
    virtual bool runOnBasicBlock(Function::iterator &bb) {  
      //check name for '.body' substring
      if(strstr(bb->getName().data(),".body")){
         if(loop_map.find(bb) == loop_map.end()){
          
           loop_map[bb].id = loop_ctr++;

           //get predecesssor
           BasicBlock* pred  = bb->getUniquePredecessor();
           BasicBlock::iterator I;
           BasicBlock::iterator E;

           //find basic block for loop exit
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
    
    //Executes on every kernel in kernel file
    virtual bool runOnFunction(Module::iterator &F) {  


      loop_map.clear();
      Function::arg_iterator a =  F->getArgumentList().begin(); 
      Function::arg_iterator z =  F->getArgumentList().end();  
      Value* trace_arg;  //trace parameter in kernel

      //Check correct kernel
      inst_iterator I,E;  
      //locate the kernel argument where we will write our trace
      for(; a!=z; a++){
        if( a->getName().equals("trace")){  
         trace_arg = cast<Value> (a);    
        }      
      } 
     
      //Look at each basic block to see if it contains a loop body
      for(Function::iterator func_itr = F->begin(), func_end = F->end(); func_itr != func_end; ++func_itr){
         MemSize::runOnBasicBlock(func_itr);
      }

      create_loop_counters(F->begin(),trace_arg);   
      inc_loop_counters(F,trace_arg);
      reset_loop_counters(F,trace_arg);
      

      //for each instruction
      for(I = inst_begin(F), E = inst_end(F); I != E; ++I){    
             
        Instruction *i = &*I;
        IRBuilder<> builder(i);
    
        //if instruction is a load or store
        if((isa<StoreInst>(i)) || (isa<LoadInst>(i)) ){  
          if(is_global(i)){  //if instruction acesses global memory
    
            //get pointer to counter
            Value *GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(builder.getInt32(0)));
            GEP = builder.CreateBitCast(GEP,PointerType::get(int32_Ty,1));
            //atomically load and increment the trace index
            builder.CreateCall(atomicINC_hook,GEP,"a_inc");

            if(isa<StoreInst>(i))
              toDelete.push_back(i);

          }
        }
        else if((isa<CallInst>(i))){
          Function* F = cast<CallInst>(i)->getCalledFunction();
          //check for memory barriers
          if((F->getName().equals("barrier"))  || (F->getName().equals("mem_fence")) || (F->getName().equals("work_group_barrier")) ){
            Value *GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(builder.getInt32(0)));
            GEP = builder.CreateBitCast(GEP,PointerType::get(int32_Ty,1));
            //atomically load and increment the trace index
            builder.CreateCall(atomicINC_hook,GEP,"a_inc");
          }
          //add warning if function called with global memory params   
          if(!F->isIntrinsic()){
            a =  F->getArgumentList().begin(); 
            z =  F->getArgumentList().end();  
            for(; a!=z; a++){
              if( a->getType()->isPointerTy()){
                //Helper function is being called with global memory as param
                if(cast<PointerType>(a->getType())->getAddressSpace() == 1){
                  if(seen.find(F) == seen.end()){
                    //add trace params to called function
                    Value *GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(builder.getInt32(2)));
                    builder.CreateStore(builder.getInt64(0x15),GEP);
                    seen[F] = 1;
                  }
                }                         
              }
            }
          }
        }

    }
    return true;                                                     
  }

  void create_loop_counters(BasicBlock *bb,Value* trace_arg){
   
    Instruction* i = bb->begin();
    IRBuilder<> builder(i);
   
    // check if more than 16 loops are present and add warning
    if(loop_ctr > 15){
      Value *GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(builder.getInt32(1)));
      builder.CreateStore(builder.getInt64(0x15),GEP);
    }
              
    //Create loop counter variable
    for(int k=0;k<loop_ctr;k++){
      Value *loop_var = builder.CreateAlloca(int32_Ty);   
      cast<AllocaInst>(loop_var)->setAlignment(8);
      Value* GEP = builder.CreateInBoundsGEP(loop_var,ArrayRef<Value*>(builder.getInt64(0)));
      builder.CreateStore(builder.getInt32(0),GEP);
      for(std::map<BasicBlock*,Loop>::iterator iter = loop_map.begin();
          iter != loop_map.end();++iter){
          if(iter->second.id == k)
           iter->second.counter = GEP;
      }
    }

  }

  void inc_loop_counters(Module::iterator &F,Value *trace_arg){
    Function::iterator func_itr,func_end;
    for( func_itr= F->begin(), func_end = F->end(); func_itr != func_end; ++func_itr){
       BasicBlock* bb = &*func_itr;
       if(loop_map.find(bb) != loop_map.end()){
          IRBuilder<> builder(bb->begin());
          Value* inc = builder.CreateLoad(loop_map[bb].counter);
          inc = builder.CreateNSWAdd(inc,builder.getInt32(1));
          builder.CreateStore(inc,loop_map[bb].counter);
       }
    }

  }

  void reset_loop_counters(Module::iterator &F,Value* trace_arg){
    Function::iterator func_itr,func_end;
    std::map<BasicBlock*,Loop>::iterator map_itr,map_end; 
    for(func_itr = F->begin(), func_end = F->end(); func_itr != func_end; ++func_itr){
       BasicBlock* bb = &*func_itr;
       for(map_itr = loop_map.begin(),map_end = loop_map.end();map_itr != map_end;++map_itr){
          if(map_itr->second.exit == bb){
            IRBuilder<> builder(bb->begin());
            Value* exit_ctr = builder.getInt32(0);
            
            //Check how many loops are currently running
            std::map<BasicBlock*, Loop>::iterator iter;
            for(iter = loop_map.begin(); iter != loop_map.end(); iter++) {
              Value* loop = builder.CreateLoad(iter->second.counter);
              loop = builder.CreateICmpNE(loop,builder.getInt32(0));
              loop = builder.CreateZExt(loop,int32_Ty);
              exit_ctr = builder.CreateNSWAdd(loop,exit_ctr);
            }


            //if more than 3 nested loops add warning
            exit_ctr = builder.CreateICmpUGT(exit_ctr,builder.getInt32(3));
            Value *GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(builder.getInt32(3)));
            Value* index = builder.CreateLoad(GEP);
            exit_ctr = builder.CreateZExt(exit_ctr,int64_Ty);
            index = builder.CreateOr(index,exit_ctr);
            builder.CreateStore(index,GEP);

            //adds warning if a loop executes more than 2^16 times
            Value* maxIter = builder.CreateLoad(map_itr->second.counter);
            maxIter = builder.CreateICmpUGE(maxIter,builder.getInt32(65535));
            maxIter = builder.CreateZExt(maxIter,int64_Ty);

            GEP = builder.CreateInBoundsGEP(trace_arg,ArrayRef<Value*>(builder.getInt32(4)));
            index = builder.CreateLoad(GEP);
               
            index = builder.CreateOr(index,maxIter);
            builder.CreateStore(index,GEP);

            builder.CreateStore(builder.getInt32(0),map_itr->second.counter);
          }
       }
    }
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

  //sets up inital information
  void init_pass(Module &M){

   int32_Ty = Type::getInt32Ty(M.getContext()); 
   int64_Ty = Type::getInt64Ty(M.getContext()); 
  
   //Sets hook to atomic_inc function
   Constant *hookInc;
   std::vector<Type *> inc_params;
   inc_params.push_back(PointerType::get(int32_Ty,1));

   FunctionType* MTy=FunctionType::get(int32_Ty,inc_params,false);
   hookInc = M.getOrInsertFunction("atomic_inc",MTy);

   atomicINC_hook = cast<Function>(hookInc);
   atomicINC_hook->addFnAttr(Attribute::NoBuiltin);
       
  }

};

} //end of namespace

char MemSize::ID = 0;
static RegisterPass<MemSize> X("size", "Memory Size"); //registers trace so '-trace' can be used to execute the pass
