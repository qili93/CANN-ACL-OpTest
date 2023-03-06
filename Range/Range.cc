#include <iostream>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <numeric>

#include "common/nputensor.h"

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
  const std::string op_type = "Range";
  // inputs
  const std::vector<int64_t> input_dims{1};
  const std::vector<float> input_data_start{0}; // start
  const std::vector<float> input_data_limit{7}; // limit
  const std::vector<float> input_data_delta{1}; // delta
  // output - y
  const std::vector<int64_t> output_dims{7};

  // input - dims
  auto input_start = new npuTensor<float>(ACL_FLOAT, input_dims.size(), input_dims.data(), ACL_FORMAT_NCHW, input_data_start.data(), memType::DEVICE);
  auto input_limit = new npuTensor<float>(ACL_FLOAT, input_dims.size(), input_dims.data(), ACL_FORMAT_NCHW, input_data_limit.data(), memType::DEVICE);
  auto input_delta = new npuTensor<float>(ACL_FLOAT, input_dims.size(), input_dims.data(), ACL_FORMAT_NCHW, input_data_delta.data(), memType::DEVICE);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_start->desc);
  input_descs.emplace_back(input_limit->desc);
  input_descs.emplace_back(input_delta->desc);
  input_buffers.emplace_back(input_start->buffer);
  input_buffers.emplace_back(input_limit->buffer);
  input_buffers.emplace_back(input_delta->buffer);

  // output - out
  auto output = new npuTensor<float>(ACL_FLOAT, output_dims.size(), output_dims.data(), ACL_FORMAT_NCHW, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // attr
  auto attr = aclopCreateAttr();
  
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

  // print output
  output->Print("y");

  // destroy - outputs
  input_start->Destroy();
  input_limit->Destroy();
  input_delta->Destroy();
  output->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}