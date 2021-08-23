#include <iostream>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <numeric>

#include "acl/acl.h"
// #include "acl/acl_op.h" // aclopExecuteV2 可以支持动态Shape算子
#include "acl/acl_op_compiler.h" // aclopCompileAndExecute 只能支持固定Shape算子
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
  const std::string op_type = "BNTrainingReduce";

  // input - x
  const std::vector<int64_t> x_dims{1, 2, 3, 4};
  std::vector<float> x_data(24);
  std::iota(x_data.begin(), x_data.end(), 0);
  // output - sum
  const std::vector<int64_t> sum_dims{2};
  std::vector<float> sum_data{0.0, 0.0};
  // output - square_sum
  std::vector<float> square_sum_data{0.0, 0.0};
  // attr - epsilon
  const float epsilon = 1e-5;

  // input - x
  auto x_desc = aclCreateTensorDesc(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW);
  auto x_size = aclGetTensorDescSize(x_desc);
  // allocate device mem and copy date to device
  void* x_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&x_device_ptr, x_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x_device_ptr, x_size, x_data.data(), x_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x_device_buffer = aclCreateDataBuffer(x_device_ptr, x_size);

  // output - sum
  auto sum_desc = aclCreateTensorDesc(ACL_FLOAT, sum_dims.size(), sum_dims.data(), ACL_FORMAT_NCHW);
  auto sum_size = aclGetTensorDescSize(sum_desc);
  // allocate device mem
  void* sum_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&sum_device_ptr, sum_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto sum_device_buffer = aclCreateDataBuffer(sum_device_ptr, sum_size);

  // output - square_sum
  auto square_sum_desc = aclCreateTensorDesc(ACL_FLOAT, sum_dims.size(), sum_dims.data(), ACL_FORMAT_NCHW);
  auto square_sum_size = aclGetTensorDescSize(square_sum_desc);
  // allocate device mem
  void* square_sum_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&square_sum_device_ptr, square_sum_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto square_sum_device_buffer = aclCreateDataBuffer(square_sum_device_ptr, square_sum_size);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x_desc);
  input_buffers.emplace_back(x_device_buffer);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(sum_desc);
  output_descs.emplace_back(square_sum_desc);
  output_buffers.emplace_back(sum_device_buffer);
  output_buffers.emplace_back(square_sum_device_buffer);

  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrFloat(attr, "epsilon", epsilon));

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
  ACL_CALL(aclrtMemcpy(sum_data.data(), sum_size, sum_device_ptr, sum_size, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CALL(aclrtMemcpy(square_sum_data.data(), square_sum_size, square_sum_device_ptr, square_sum_size, ACL_MEMCPY_DEVICE_TO_HOST));

  // print output
  std::cout << "sum_data = [";
  for (size_t i = 0; i < sum_data.size(); ++i) {
    std::cout << sum_data[i] << ", ";
  }
  std::cout << "]" << std::endl;

  std::cout << "square_sum_data = [";
  for (size_t i = 0; i < square_sum_data.size(); ++i) {
    std::cout << square_sum_data[i] << ", ";
  }
  std::cout << "]" << std::endl;


  // destroy
  ACL_CALL(aclDestroyDataBuffer(x_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(sum_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(square_sum_device_buffer));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFree(sum_device_ptr));
  ACL_CALL(aclrtFree(square_sum_device_ptr));
  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(sum_desc);
  aclDestroyTensorDesc(square_sum_desc);
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}