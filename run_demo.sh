#!/bin/bash

TARGET_EXE=${1:-Add}

# update TARGET_EXE for the operator

# ----------- FAIL -----------
# TARGET_EXE=Resize
# TARGET_EXE=ResizeD

# ----------- WORK -----------
# TARGET_EXE=Add
# TARGET_EXE=ResizeNearestNeighborV2
# TARGET_EXE=BroadcastTo
# TARGET_EXE=BroadcastToD
# TARGET_EXE=BinaryCrossEntropy
# TARGET_EXE=ArgMaxV2
# TARGET_EXE=BNTrainingReduce
# TARGET_EXE=BNTrainingUpdate
# TARGET_EXE=FillV2D
# TARGET_EXE=DeformableOffsets
# TARGET_EXE=BatchMatMul
# TARGET_EXE=ReduceSumD
TARGET_EXE=ReduceSum

echo "----------- buiding target : ${TARGET_EXE} --------------"

build_dir="$(pwd)/${TARGET_EXE}/build"
if [ -d ${build_dir} ];then
  rm -rf ${build_dir}
fi
mkdir -p ${build_dir} && cd ${build_dir} 

# build
mkdir -p ${build_dir} && cd ${build_dir} 
cmake -DTARGET_EXE=${TARGET_EXE} -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../../
make

# run
echo "-------------- start running op : ${TARGET_EXE} --------------"
export ASCEND_GLOBAL_LOG_LEVEL=0 # debug level
./${TARGET_EXE}
