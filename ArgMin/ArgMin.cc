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
  const std::string op_type = "ArgMin";
  // input - x
  const std::vector<int64_t> x_dims{2, 3, 4};
  std::vector<float> x_data(24);
  std::iota(x_data.begin(), x_data.end(), 0);
  // input - axis
  const std::vector<int64_t> axis_dims{1};
  const std::vector<int64_t> axis_data{1};
  // output - y
  std::vector<int64_t> y_dims{2, 4};
  // attr - shape
  const bool keep_dims = false;

  // inputs
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_ND, x_data.data());
  auto input_axis = new npuTensor<int64_t>(ACL_INT64, axis_dims.size(), axis_dims.data(), ACL_FORMAT_ND, axis_data.data(), memType::HOST);
  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_descs.emplace_back(input_axis->desc);
  input_buffers.emplace_back(input_x->buffer);
  input_buffers.emplace_back(input_axis->buffer);

  // output
  auto output_y = new npuTensor<int64_t>(ACL_INT64, y_dims.size(), y_dims.data(), ACL_FORMAT_ND, nullptr);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y->desc);
  output_buffers.emplace_back(output_y->buffer);
  
  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrInt(attr, "dtype", 9));
  // ACL_CALL(aclopSetAttrDataType(attr, "dtype", ACL_INT64));

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
  input_x->Print("x");
  output_y->Print("y");

  // destroy
  input_x->Destroy();
  input_axis->Destroy();
  output_y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}