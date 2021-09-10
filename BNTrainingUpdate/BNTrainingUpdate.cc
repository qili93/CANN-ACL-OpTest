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
  const std::string op_type = "BNTrainingUpdate";

  // input - x
  const std::vector<int64_t> x_dims{1, 2, 4};
  std::vector<float> x_data(8);
  std::iota(x_data.begin(), x_data.end(), 0);
  // input - sum
  const std::vector<int64_t> c_dims{2};
  const std::vector<float> sum_data{6.0, 22.0};
  // input - square_sum
  const std::vector<float> square_sum_data{14.0, 126.0};
  // input - scale
  const std::vector<float> scale_data{1.0, 1.0};
  // input - offset
  const std::vector<float> offset_data{0.0, 0.0};
  // input - mean
  const std::vector<float> mean_data{0.0, 0.0};
  // input - var
  const std::vector<float> var_data{1.0, 1.0};
  // attr - epsilon
  const float epsilon = 1e-5;
  // attr - factor
  const float factor = 0.9;

  // input - x
  auto x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, x_data.data());
  auto sum = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, sum_data.data());
  auto square_sum = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, square_sum_data.data());
  auto scale = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, scale_data.data());
  auto offset = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, offset_data.data());
  auto mean = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, mean_data.data());
  auto var = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, var_data.data());

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x->desc);
  input_descs.emplace_back(sum->desc);
  input_descs.emplace_back(square_sum->desc);
  input_descs.emplace_back(scale->desc);
  input_descs.emplace_back(offset->desc);
  input_descs.emplace_back(mean->desc);
  input_descs.emplace_back(var->desc);
  input_buffers.emplace_back(x->buffer);
  input_buffers.emplace_back(sum->buffer);
  input_buffers.emplace_back(square_sum->buffer);
  input_buffers.emplace_back(scale->buffer);
  input_buffers.emplace_back(offset->buffer);
  input_buffers.emplace_back(mean->buffer);
  input_buffers.emplace_back(var->buffer);

  // output - y
  auto y = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, nullptr);
  auto mean_out = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, nullptr);
  auto var_out = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, nullptr);
  auto saved_mean = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, nullptr);
  auto saved_var = new npuTensor<float>(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y->desc);
  output_descs.emplace_back(mean_out->desc);
  output_descs.emplace_back(var_out->desc);
  output_descs.emplace_back(saved_mean->desc);
  output_descs.emplace_back(saved_var->desc);
  output_buffers.emplace_back(y->buffer);
  output_buffers.emplace_back(mean_out->buffer);
  output_buffers.emplace_back(var_out->buffer);
  output_buffers.emplace_back(saved_mean->buffer);
  output_buffers.emplace_back(saved_var->buffer);

  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrFloat(attr, "factor", factor));
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
  y->Print("y");
  mean_out->Print("mean_out");
  var_out->Print("var_out");
  saved_mean->Print("saved_mean");
  saved_var->Print("saved_var");

  // destroy - inputs
  x->Destroy();
  sum->Destroy();
  square_sum->Destroy();
  scale->Destroy();
  offset->Destroy();
  mean->Destroy();
  var->Destroy();
  // destroy - outputs
  y->Destroy();
  mean_out->Destroy();
  var_out->Destroy();
  saved_mean->Destroy();
  saved_var->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}