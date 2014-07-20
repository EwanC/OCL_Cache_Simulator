PROJ=bfs

CC=gcc

CFLAGS= -Wall -DUNIX -g -DDEBUG

# Check for 32-bit vs 64-bit
PROC_TYPE = $(strip $(shell uname -m | grep 64))
 
# Check for Mac OS
OS = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN = $(strip $(findstring DARWIN, $(OS)))


# MacOS System
ifneq ($(DARWIN),)
	CFLAGS += -DMAC
	LIBS=-framework OpenCL

	ifeq ($(PROC_TYPE),)
		CFLAGS+=-arch i386
	else
		CFLAGS+=-arch x86_64
	endif
else

# Linux OS
LIBS=-lOpenCL -lopenclwrapper -lboost_system -lboost_program_options
ifeq ($(PROC_TYPE),)
	CFLAGS+=-m32
else
	CFLAGS+=-m64
endif

AMDAPPSDKROOT = /opt/AMDAPP


# Check for Linux-AMD
ifdef AMDAPPSDKROOT
   INC_DIRS=. $(AMDAPPSDKROOT)/include
   INC_DIRS +=../../../../OpenCL/graphics/include/OpenCLWrapper
   INC_DIRS +=../../lib/bench_library/include/

	ifeq ($(PROC_TYPE),)
		LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86
	else
		LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86_64
	endif
else


endif
endif
LIB_DIRS += ../../../../OpenCL/graphics/lib/OpenCLWrapper


$(PROJ): $(PROJ).cpp
	$(CC) $(CFLAGS) ../../lib/bench_library/src/bench_support.cpp  -o $@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) /usr/lib/libboost_program_options-mt.so $(LIBS)

.PHONY: clean

clean:
	rm $(PROJ)
