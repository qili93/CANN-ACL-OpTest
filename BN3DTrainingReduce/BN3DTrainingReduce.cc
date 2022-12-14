#include <iostream>
#include <vector>

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
  const std::string op_type = "BN3DTrainingReduce";

  // input - x
  const std::vector<int64_t> x_dims{2, 4, 6, 6, 6}; // NCDHW
  std::vector<float> x_data(2*4*6*6*6, 1.0);
  // std::iota(x_data.begin(), x_data.end(), 1.0);
  // output - sum & square_sum
  const std::vector<int64_t> sum_dims{4}; // ND
  std::vector<float> sum_data(4, 0.0);
  std::vector<float> square_sum_data(4, 0.0);
  // attr - epsilon
  const float epsilon = 1e-5;

  // input - x
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCDHW, x_data.data());
  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_buffers.emplace_back(input_x->buffer);

  // output - sum, square_sum
  // NOTE: will fail if change ACL_FORMAT_ND to ACL_FORMAT_NCHW
  auto output_sum = new npuTensor<float>(ACL_FLOAT, sum_dims.size(), sum_dims.data(), ACL_FORMAT_NCHW, nullptr);
  auto output_square_sum = new npuTensor<float>(ACL_FLOAT, sum_dims.size(), sum_dims.data(), ACL_FORMAT_NCHW, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_sum->desc);
  output_descs.emplace_back(output_square_sum->desc);
  output_buffers.emplace_back(output_sum->buffer);
  output_buffers.emplace_back(output_square_sum->buffer);

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

  // print output
  output_sum->Print("output_sum");
  output_square_sum->Print("output_square_sum");


  // destroy
  // destroy
  input_x->Destroy();
  output_sum->Destroy();
  output_square_sum->Destroy();
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}