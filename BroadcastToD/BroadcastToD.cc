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

  // Get Run Mode - ACL_HOST
  aclrtRunMode runMode;
  ACL_CALL(aclrtGetRunMode(&runMode));
  std::string run_mode_str = (runMode == ACL_DEVICE) ? "ACL_DEVICE" : "ACL_HOST";
  std::cout << "aclrtRunMode is : " << run_mode_str << std::endl;

  // op type
  const std::string op_type = "BroadcastToD";
  // input - x
  const std::vector<int64_t> x_dims{3, 1, 1};
  const std::vector<float> x{1, 2, 3};
  // output - y
  const std::vector<int64_t> y_dims{3, 2, 4};
  std::vector<float> y(24, 0.0);
  // attr - shape
  const std::vector<int64_t> shape{3, 2, 4};

  // input0 - x - should use device buffer
  auto x_desc = aclCreateTensorDesc(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW);
  auto x_size = aclGetTensorDescSize(x_desc);
  // allocate device mem and copy date to device
  void* x_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&x_device_ptr, x_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x_device_ptr, x_size, x.data(), x_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x_device_buffer = aclCreateDataBuffer(x_device_ptr, x_size);
  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x_desc);
  input_buffers.emplace_back(x_device_buffer);

  // output0 - y - should use device buffer
  auto y_desc = aclCreateTensorDesc(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW);
  auto y_size = aclGetTensorDescSize(y_desc);
  // allocate device mem
  void* y_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&y_device_ptr, y_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto y_device_buffer = aclCreateDataBuffer(y_device_ptr, y_size);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y_desc);
  output_buffers.emplace_back(y_device_buffer);
  
  // attr - shape
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrListInt(attr, "shape", shape.size(), shape.data()));

  // create stream
  aclrtStream stream = nullptr;
  ACL_CALL(aclrtCreateStream(&stream));

  std::cout << "aclopCompileAndExecute : " << op_type << std::endl;
  ACL_CALL(aclopCompileAndExecute(op_type.c_str(), 
            input_descs.size(), input_descs.data(), input_buffers.data(), 
            output_descs.size(), output_descs.data(), output_buffers.data(), 
            attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));

  // sync and destroy stream
  ACL_CALL(aclrtSynchronizeStream(stream));
  ACL_CALL(aclrtDestroyStream(stream));

  // copy output from device to host
  ACL_CALL(aclrtMemcpy(y.data(), y_size, y_device_ptr, y_size, ACL_MEMCPY_DEVICE_TO_HOST));

  // print output
  std::cout << "y = [";
  for (size_t i = 0; i < y.size(); ++i) {
    std::cout << y[i] << ", ";
  }
  std::cout << "]" << std::endl;

  // destroy
  ACL_CALL(aclDestroyDataBuffer(x_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(y_device_buffer));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFree(y_device_ptr));
  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(y_desc);
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}