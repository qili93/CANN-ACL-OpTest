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
  const std::string op_type = "Fill";
  // input 0 - dims
  const std::vector<int64_t> input_0_dims{1};
  const std::vector<int64_t> input_0_data{1};
  // input 1 - value
  const std::vector<int64_t> input_1_dims{1};
  const std::vector<int64_t> input_1_data{1};
  // output - y
  const std::vector<int64_t> output_dims{1};

  // input tensor 0 - dims
  auto input_0 = new npuTensor<int64_t>(ACL_INT64, input_0_dims.size(), input_0_dims.data(), ACL_FORMAT_NCHW, input_0_data.data(), memType::HOST);
  // input tensor 1 - value
  auto input_1 = new npuTensor<int64_t>(ACL_INT64, input_1_dims.size(), input_1_dims.data(), ACL_FORMAT_NCHW, input_1_data.data(), memType::HOST);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_0->desc);
  input_descs.emplace_back(input_1->desc);
  input_buffers.emplace_back(input_0->buffer);
  input_buffers.emplace_back(input_1->buffer);

  // output - out
  auto output = new npuTensor<int64_t>(ACL_INT64, output_dims.size(), output_dims.data(), ACL_FORMAT_NCHW, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // attr
  auto attr = aclopCreateAttr();
  // ACL_CALL(aclopSetAttrFloat(attr, "value", value));
  
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
  input_0->Destroy();
  input_1->Destroy();
  output->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}