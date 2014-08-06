
#Copyright 2014 Ewan Crawford<ewan.cr@gmail.com>


#This file is part of OpenCL Visuliser.

#OpenCL Visuliser is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#OpenCL Visuliser is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with OpenCL Visuliser.  If not, see <http://www.gnu.org/licenses/>


#!/usr/bin/python

from optparse import OptionParser
from distutils.spawn import find_executable
import os

def Dependencies():
  clang = find_executable("clang",os.environ["PATH"])
  if not clang:
    print "Please install clang frontend and add to $PATH"
    exit(0)

  opt = find_executable("opt",os.environ["PATH"])
  if not opt:
     print "Please install LLVM Optimizer,opt, and add to $PATH"
     exit(0)

  R= find_executable("R",os.environ["PATH"])
  if not R:
    print "Please install R graphing software and add to $PATH"
    exit(0)

  axtor= find_executable("axtor",os.environ["PATH"])
  if not axtor:
    print "Please install axtor backend and add to $PATH"
    exit(0)


def parseArgs():
    parser = OptionParser(usage="Usage: %prog [options] 'OpenCL Program' ",
                        version="Visualizer 1.0")

    parser.add_option("-s","--simulate",
                                      action="store_true",
                                      dest="sim",
                                      default=False,
                                      help="Simulate kernel cache performance")

    parser.add_option("-w","--warp",
                         action="store",
                         type="int",
                         default=32,
                         dest="warp",
                         help="Number of threads per warp")

    parser.add_option("-a","--algorithm",
                         action="store",
                         type="choice",
                         choices=["coalesced","rr","seq","rand","none"],
                         default="none",
                         dest="alg",
                         help="Scheduling algorithm")

    (options,args) = parser.parse_args()
    if len(args) < 1:
       parser.error("wrong number of arguments")
    
    return (options,args)

def setup_env():

   tools  = os.getenv("VIS_TOOLS")
   if not tools:
      print "please set environmental variable VIS_TOOLS to directory where the llvm tools are built"
      exit(0)

   passes = os.getenv("VIS_PASSES")
   if not passes:
      print "please set environmental variable VIS_PASSES to directory where the llvm passes are built"
      exit(0)
  
   return tools


def main():
    (opts,args) = parseArgs()
    Dependencies()

    program = os.path.basename(args[0])
    directory =  os.path.dirname(args[0])

    build_dir = setup_env()
   
    if (program == "mt" or program == "mv"):
       os.system(args[0]+" --kernelName " + args[1])
    else:
       os.system(args[0])

    trace = os.getcwd() + "/trace.txt"
    scheduler_path = build_dir+"/scheduler/scheduler "
   
    os.system(scheduler_path + trace + " " +opts.alg + " "+str(opts.warp));
    os.system("rm " + trace);
    
    graph = os.getcwd()+ "/graph.out"
    cache = os.getcwd() + "/cache.out"
    
    R_path = os.getcwd()+"/support/scripts/graph.R "
    cache_path = build_dir+"/cacheSimulator/cachSim "   

    os.system("Rscript " + R_path + graph)
    if opts.sim:
      os.system(cache_path + cache + " 16 128 4 LRU WTNA")
   
    os.system("rm " + graph)
    os.system("rm " + cache)
    os.system("mv " + graph+".png " + os.getcwd()+"/"+program+".png") 

if __name__ == "__main__":
  main()
