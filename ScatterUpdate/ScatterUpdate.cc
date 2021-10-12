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
  const std::string op_type = "ScatterUpdate";
  // input - var
  const std::vector<int64_t> var_dims{24};
  std::vector<float> var_data(24, 1);
  // input - value
  const std::vector<int64_t> indices_dims{12};
  std::vector<int64_t> indices_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  // input - value
  const std::vector<int64_t> updates_dims{12};
  std::vector<float> updates_data(12, 6);
  // output
  const std::vector<int64_t> y_dims{24};

  // inputs
  auto input_var = new npuTensor<float>(ACL_FLOAT, var_dims.size(), var_dims.data(), ACL_FORMAT_ND, var_data.data());
  auto input_indices = new npuTensor<int64_t>(ACL_INT64, indices_dims.size(), indices_dims.data(), ACL_FORMAT_ND, indices_data.data());
  auto input_updates = new npuTensor<float>(ACL_FLOAT, updates_dims.size(), updates_dims.data(), ACL_FORMAT_ND, updates_data.data());
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_var->desc);
  input_descs.emplace_back(input_indices->desc);
  input_descs.emplace_back(input_updates->desc);
  input_buffers.emplace_back(input_var->buffer);
  input_buffers.emplace_back(input_indices->buffer);
  input_buffers.emplace_back(input_updates->buffer);

  // output
  auto output_y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_ND, nullptr);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y->desc);
  output_buffers.emplace_back(output_y->buffer);
  
  // attributes
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrBool(attr, "use_locking", false));

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
  input_var->Destroy();
  input_indices->Destroy();
  input_updates->Destroy();
  output_y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}