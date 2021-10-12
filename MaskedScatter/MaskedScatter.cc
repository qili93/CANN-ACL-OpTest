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
  const std::string op_type = "MaskedScatter";
  // input - x
  const std::vector<int64_t> x_dims{5};
  const std::vector<float> x_data(5, 1);
  // input - mask
  const std::vector<int64_t> mask_dims{5};
  bool mask_data[5] = {false};
  mask_data[0] = true;
  mask_data[1] = true;
  // input - value
  const std::vector<int64_t> value_dims{5};
  const std::vector<float> value_data{2, 3, 0, 0, 0};
  // output
  const std::vector<int64_t> y_dims{5};

  // inputs
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_ND, x_data.data());
  auto input_mask = new npuTensor<bool>(ACL_BOOL, mask_dims.size(), mask_dims.data(), ACL_FORMAT_ND, mask_data);
  auto input_value = new npuTensor<float>(ACL_FLOAT, value_dims.size(), value_dims.data(), ACL_FORMAT_ND, value_data.data());

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_descs.emplace_back(input_mask->desc);
  input_descs.emplace_back(input_value->desc);
  input_buffers.emplace_back(input_x->buffer);
  input_buffers.emplace_back(input_mask->buffer);
  input_buffers.emplace_back(input_value->buffer);

  // output
  auto output_y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_ND, nullptr);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y->desc);
  output_buffers.emplace_back(output_y->buffer);
  
  // attributes
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
  output_y->Print("y");

  // destroy
  input_x->Destroy();
  input_mask->Destroy();
  input_value->Destroy();
  output_y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}