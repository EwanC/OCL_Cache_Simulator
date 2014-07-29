==============================================================
           
                     OpenCL Visualiser
           
==============================================================

July 2014  Ewan Crawford[ewan.cr@gmail.com]

//////////////////////// Introduction ////////////////////////////

Tool for visualising the memory accesses of OpenCL programs. Extended to predict the cache performance of the
OpenCL programs on FERMI GPUs by rearranging the memory accesses to be platform independent. Cache prediction
is still under development however as the results can not yet be validated against a complete benchmark suite.

OpenCL programs are written using a wrapper created by Alberto Magni[alberto.magni86@gmail.com]. This wrapper 
performs instrumentation passes on the OpenCL kernels using the LLVM optimizer tool opt. These passes allow
instructions to be inserted that record information about global memory accesses. Axtor is then used as
a OpenCL backend to convert the instrumented LLVM IR back into an OpenCL kernel that can be executed. The 
wrapper allows this is process to be hidden from the user.

After execution has finished a trace of memory accesses is available dependent executed by the users 
hardware. In order to generalise this to a machine independent representation of GPU OpenCL execution these
memory accesses are then reordered using a scheduler. Additionally this trace can be put through a NVIDIA FERMI
cache simulation to predict it's performance. 

However by default the original trace is plotted as a graph without any cache prediction or scheduling.
This is done using R[http://www.r-project.org/]. However the coalescing done by GPU's is not taken into
account and all plots are of individual accesses.


This project is based on my undergraduate project at Edinburgh University. The dissertation for which can 
be found in /support/dissertation.pdf bitbucket.org/gnarf/axtor/


Example graphs from polybench benchmarks that are rearranged using the scheduler.
![Alt text](/examples/2dconv.png?raw=true "2D Convolution")
![Alt text](/examples/3mm.png?raw=true "3 Matrix Multiplication")
![Alt text](/examples/fdtd2d.png?raw=true "2-D Finite Different Time Domain Kernel(FDTD-2D)")



//////////////////////// Dependencies //////////////////////////// 

LLVM 3.3[http://llvm.org/]
Axtor[bitbucket.org/gnarf/axtor/, Simon Moll]

This tool is only available for Linux, and additionally has only been tested on Ubuntu and Mint distributions.

/////////////////////// Build Instructions ///////////////////////
 
 # First build the LLVM passes by copying them into your llvm source tree, and rebuilding LLVM

 $> cp SOURCE_DIR/passes/*   LLVM_SOURCE_DIR/lib/Transforms/
 $> cd LLVM_SOURCE_DIR/lib/Transforms
 $> append 'add_subdirectory(MemTrace), add_subdirectory(MemSize)' to CMakeLists.txt 
 $> rebuild LLVM
 $> export VIS_PASSES as path to shared object files for passes

 # Next build the tool and export location of built 'tools/'directory as VIS_TOOLS

 $> mkdir build
 $> cd build
 $> cmake [SOURCE_DIR]
 $> make
 $> export VIS_TOOLS

 # Axtor needs to be built from bitbucket.org/gnarf/axtor/ and executable added to $PATH
 # R graphing tool also needs to be installed so that 'Rscipt' is in $PATH
  

///////////////////// Project structure ///////////////////////////

visualise.py      --Links components together so the tool can be run easily                  

benchmarks/       --Benchmarks implemented using wrapper
      
        matrix/   -- Matrix based benchmarks from NVIDIA
           mm/
           mt/
           mv/

        parboil/   -- contains selected benchmarks from parboil benchmark suite
                      [http://impact.crhc.illinois.edu/Parboil/parboil.aspx]
           stencil/
           bfs/
  
        polybench/ --contains complete polybench benchmark suite
                     [http://web.cse.ohio-state.edu/~pouchet/software/polybench/]
           
             
passes/
    MemSize/        --LLVM pass that calculates the length of the trace
                      so that memory can be allocated

    MemTrace/       --LLVM pass that adds instructions to record memory accesses

support/

    scripts/        -- Bash scripts called by OpenCL wrapper for executing LLVM passes
                       as well as R graphing

    wrapper/        --Modified version of Alberto Magni's OpenCL wrapper

tools/

    scheduler/      --Rearranges machine dependent memory accesses so they represent a GPU.
 
    cacheSimulator/ --simulates cache performance of memory accesses

examples/           --Examples of graphs that can be produced using the tool.

