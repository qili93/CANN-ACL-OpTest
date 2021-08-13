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
  const std::string op_type = "BinaryCrossEntropy";
  // input - x
  const std::vector<int64_t> x_dims{3};
  const std::vector<float> x_data{0,5, 0.6, 0.7};
  // input - y
  const std::vector<int64_t> y_dims{3};
  const std::vector<float> y_data{1.0, 0.0, 1.0};
  // output - output
  const std::vector<int64_t> out_dims{3};
  std::vector<float> out_data{0.0, 0.0, 0.0};
  // attr - shape
  const std::string reduction = "none";

  // input0 - x - should use device buffer
  auto x_desc = aclCreateTensorDesc(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW);
  auto x_size = aclGetTensorDescSize(x_desc);
  // allocate device mem and copy date to device
  void* x_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&x_device_ptr, x_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x_device_ptr, x_size, x_data.data(), x_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x_device_buffer = aclCreateDataBuffer(x_device_ptr, x_size);

  // input1 - y - should use device buffer
  auto y_desc = aclCreateTensorDesc(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW);
  auto y_size = aclGetTensorDescSize(x_desc);
  // allocate device mem and copy date to device
  void* y_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&y_device_ptr, y_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(y_device_ptr, y_size, y_data.data(), y_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto y_device_buffer = aclCreateDataBuffer(y_device_ptr, y_size);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x_desc);
  input_descs.emplace_back(y_desc);
  input_buffers.emplace_back(x_device_buffer);
  input_buffers.emplace_back(y_device_buffer);

  // output - should use device buffer
  auto out_desc = aclCreateTensorDesc(ACL_FLOAT, out_dims.size(), out_dims.data(), ACL_FORMAT_NCHW);
  auto out_size = aclGetTensorDescSize(y_desc);
  // allocate device mem
  void* out_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&out_device_ptr, out_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto out_device_buffer = aclCreateDataBuffer(out_device_ptr, out_size);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(out_desc);
  output_buffers.emplace_back(out_device_buffer);
  
  // attr - shape
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrString(attr, "reduction", reduction.c_str()));

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
  ACL_CALL(aclrtMemcpy(out_data.data(), out_size, out_device_ptr, out_size, ACL_MEMCPY_DEVICE_TO_HOST));

  // print output
  std::cout << "out = [";
  for (size_t i = 0; i < out_data.size(); ++i) {
    std::cout << out_data[i] << ", ";
  }
  std::cout << "]" << std::endl;

  // destroy
  ACL_CALL(aclDestroyDataBuffer(x_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(y_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(out_device_buffer));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFree(y_device_ptr));
  ACL_CALL(aclrtFree(out_device_ptr));
  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(y_desc);
  aclDestroyTensorDesc(out_desc);
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}