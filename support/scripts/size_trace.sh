#! /bin/bash

CLANG=clang
OPT=opt
AXTOR=axtor

INPUT_FILE=$1
OUTPUT_FILE=/tmp/output.cl



if [ -z "$VIS_PASSES" ]; then
  echo "Need to export $VIS_PASSES"
  exit 1
fi

if [ -z "$SCRIPT_PATH" ]; then
  echo "Need to export $SCRIPT_PATH"
  exit 1
fi

OCLDEF=$SCRIPT_PATH/ocldef_intel.h
OPTIMIZATION=-O0
TMP_1=/tmp/tc_tmp${RANDOM}.cl
TMP_2=/tmp/ax_tmp${RANDOM}.cl
PASS=$VIS_PASSES/libMemSize.so

$CLANG -x cl \
       -target nvptx \
       -include ${OCLDEF} \
       ${OPTIMIZATION} \
       ${INPUT_FILE} \
       -S -emit-llvm -fno-builtin -o ${TMP_1}


$OPT -S -mem2reg -load ${PASS} -size <${TMP_1}> ${TMP_2}
${AXTOR} ${TMP_2} -m OCL -o ${OUTPUT_FILE} 

rm ${TMP_1}
rm ${TMP_2}
