# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ascend_demo_native)

# set(TARGET_EXE resize)
message(STATUS "Building target TARGET_EXE=${TARGET_EXE}")

# 1. path to Ascend Path
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_DIR /usr/local/Ascend)
endif()

# 2.  Add compile options
add_compile_options(-std=c++14)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# 4. Ascend include directory
set(ACL_INCLUDE_DIR "${HUAWEI_ASCEND_NPU_DDK_ROOT}/acllib/include")
set(ATC_INCLUDE_DIR "${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/include")
set(OPP_INCLUDE_DIR "${HUAWEI_ASCEND_NPU_DDK_ROOT}/opp/op_proto/built-in/inc")
include_directories(${ACL_INCLUDE_DIR})
include_directories(${ATC_INCLUDE_DIR})
include_directories(${OPP_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 3. ascend driver, atc, acllib
set(ASCEND_DRIVER_DIR ${ASCEND_DIR}/driver/lib64)
set(ASCEND_DRIVER_COMMON_DIR ${ASCEND_DIR}/driver/lib64/common)
set(ASCEND_DRIVER_SHARE_DIR ${ASCEND_DIR}/driver/lib64/share)
set(ASCEND_RUNTIME_DIR ${ASCEND_DIR}/fwkacllib/lib64)
set(ASCEND_ATC_DIR ${ASCEND_DIR}/atc/lib64)
set(ASCEND_ACL_DIR ${ASCEND_DIR}/acllib/lib64)
set(STATIC_ACL_LIB ${ASCEND_ACL_DIR})

set(ASCEND_MS_RUNTIME_PATH ${ASCEND_RUNTIME_DIR} ${ASCEND_ACL_DIR} ${ASCEND_ATC_DIR})
set(ASCEND_MS_DRIVER_PATH ${ASCEND_DRIVER_DIR} ${ASCEND_DRIVER_COMMON_DIR})
set(ATLAS_RUNTIME_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/lib64)
set(ATLAS_RUNTIME_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include)
set(ATLAS_ACL_DIR ${ASCEND_DIR}/ascend-toolkit/latest/acllib/lib64)
set(ATLAS_ATC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/atc/lib64)
set(ATLAS_MS_RUNTIME_PATH ${ATLAS_RUNTIME_DIR} ${ATLAS_ACL_DIR} ${ATLAS_ATC_DIR})

set(atlas_graph_lib ${ATLAS_RUNTIME_DIR}/libgraph.so)
set(atlas_ge_runner_lib ${ATLAS_RUNTIME_DIR}/libge_runner.so)
set(atlas_acl_lib ${ATLAS_RUNTIME_DIR}/libascendcl.so)
INCLUDE_DIRECTORIES(${ATLAS_RUNTIME_INC_DIR})


ADD_LIBRARY(ascend_ge SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ascend_ge PROPERTY IMPORTED_LOCATION ${atlas_ge_runner_lib})

ADD_LIBRARY(ascend_graph SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ascend_graph PROPERTY IMPORTED_LOCATION ${atlas_graph_lib})

ADD_LIBRARY(atlas_acl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET atlas_acl PROPERTY IMPORTED_LOCATION ${atlas_acl_lib})

set(extern_ascend ascend_ge ascend_graph atlas_acl CACHE INTERNAL "acllib runtime libs")

# 4. ascend-toolkit
set(ASCEND_CL_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/lib64)

set(ascend_hccl_lib ${ASCEND_CL_DIR}/libhccl.so)
set(ascendcl_lib ${ASCEND_CL_DIR}/libascendcl.so)
set(acl_op_compiler_lib ${ASCEND_CL_DIR}/libacl_op_compiler.so)
set(FWKACLLIB_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include)
set(ACLLIB_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/acllib/include)

message(STATUS "FWKACLLIB_INC_DIR ${FWKACLLIB_INC_DIR}")
message(STATUS "ASCEND_CL_DIR ${ASCEND_CL_DIR}")
INCLUDE_DIRECTORIES(${FWKACLLIB_INC_DIR})
INCLUDE_DIRECTORIES(${ACLLIB_INC_DIR})

ADD_LIBRARY(ascendcl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ascendcl PROPERTY IMPORTED_LOCATION ${ascendcl_lib})

ADD_LIBRARY(ascend_hccl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ascend_hccl PROPERTY IMPORTED_LOCATION ${ascend_hccl_lib})

ADD_LIBRARY(acl_op_compiler SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET acl_op_compiler PROPERTY IMPORTED_LOCATION ${acl_op_compiler_lib})

set(extern_ascend_cl ascendcl acl_op_compiler CACHE INTERNAL "acltoolkit libs")

# 5. Final target
add_executable(${TARGET_EXE} ${TARGET_EXE}/${TARGET_EXE}.cc common/logging.cc)
target_link_libraries(${TARGET_EXE} ${extern_ascend} ${extern_ascend_cl})

