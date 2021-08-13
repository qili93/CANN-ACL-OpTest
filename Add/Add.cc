#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "common/logging.h"

#define ACL_CALL(msg) CHECK_EQ(reinterpret_cast<aclError>(msg), ACL_SUCCESS)

int main() {
  // Init
  ACL_CALL(aclInit(nullptr));
  ACL_CALL(aclrtSetDevice(0));

  // op type
  const std::string op_type = "Add";
  // common dims
  const std::vector<int64_t> dims{2, 3};
  // inputs
  const std::vector<float> x1{1, 2, 3, 4, 5, 6};
  const std::vector<float> x2{1, 2, 3, 4, 5, 6};
  // output
  std::vector<float> y{0, 0, 0, 0, 0, 0};

  // input - x1
  auto x1_desc = aclCreateTensorDesc(ACL_FLOAT, dims.size(), dims.data(), ACL_FORMAT_ND);
  auto x1_size = aclGetTensorDescSize(x1_desc);
  void* x1_device_ptr;
  ACL_CALL(aclrtMalloc(&x1_device_ptr,x1_size,ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x1_device_ptr, x1_size, x1.data(), x1_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x1_buffer = aclCreateDataBuffer(x1_device_ptr, x1_size);
  // input - x2
  auto x2_desc = aclCreateTensorDesc(ACL_FLOAT, dims.size(), dims.data(), ACL_FORMAT_ND);
  auto x2_size = aclGetTensorDescSize(x1_desc);
  void* x2_device_ptr;
  ACL_CALL(aclrtMalloc(&x2_device_ptr,x2_size,ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x2_device_ptr, x2_size, x2.data(), x2_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x2_buffer = aclCreateDataBuffer(x2_device_ptr, x2_size);
  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x1_desc);
  input_descs.emplace_back(x2_desc);
  input_buffers.emplace_back(x1_buffer);
  input_buffers.emplace_back(x2_buffer);

  // output - y
  auto y_desc = aclCreateTensorDesc(ACL_FLOAT, dims.size(), dims.data(), ACL_FORMAT_ND);
  auto y_size = aclGetTensorDescSize(y_desc);
  void* y_device_ptr;
  ACL_CALL(aclrtMalloc(&y_device_ptr, y_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto y_buffer = aclCreateDataBuffer(y_device_ptr, y_size);
  // outputs
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y_desc);
  output_buffers.emplace_back(y_buffer);

  // attributes
  auto attr = aclopCreateAttr();

  // create stream
  aclrtStream stream = nullptr;
  ACL_CALL(aclrtCreateStream(&stream));

  // run operator
  std::cout << "aclopCompileAndExecute : " << op_type << std::endl;
  ACL_CALL(aclopCompileAndExecute(op_type.c_str(), 
            input_descs.size(), input_descs.data(), input_buffers.data(), 
            output_descs.size(), output_descs.data(), output_buffers.data(), 
            attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));


  // sync and destroy stream
  ACL_CALL(aclrtSynchronizeStream(stream));
  ACL_CALL(aclrtDestroyStream(stream));

  ACL_CALL(aclrtMemcpy(y.data(), y_size, y_device_ptr, y_size, ACL_MEMCPY_DEVICE_TO_HOST));

  std::cout << "y = [";
  for (int i = 0; i < y.size(); ++i) {
    std::cout << y[i] << ", ";
  }
  std::cout << "]" << std::endl;

  ACL_CALL(aclDestroyDataBuffer(x1_buffer));
  ACL_CALL(aclDestroyDataBuffer(x2_buffer));
  ACL_CALL(aclDestroyDataBuffer(y_buffer));
  ACL_CALL(aclrtFree(x1_device_ptr));
  ACL_CALL(aclrtFree(x2_device_ptr));
  ACL_CALL(aclrtFree(y_device_ptr));

  aclDestroyTensorDesc(x1_desc);
  aclDestroyTensorDesc(x2_desc);
  aclDestroyTensorDesc(y_desc);
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}