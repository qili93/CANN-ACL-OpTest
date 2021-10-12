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
  const std::string op_type = "StridedSliceAssign";
  // input - var
  const std::vector<int64_t> var_dims{4, 4};
  std::vector<float> var_data(16, 1);
  // input - value
  const std::vector<int64_t> value_dims{2, 2};
  std::vector<float> value_data{2, 3, 4, 5};
  // input - begin
  const std::vector<int64_t> begin_dims{2};
  std::vector<int64_t> begin_data{0, 0};
  // input - end
  const std::vector<int64_t> end_dims{2};
  std::vector<int64_t> end_data{2, 2};
  // input - stride
  const std::vector<int64_t> stride_dims{2};
  std::vector<int64_t> stride_data{1, 1};
  // output
  const std::vector<int64_t> y_dims{4, 4};

  // inputs
  auto input_var = new npuTensor<float>(ACL_FLOAT, var_dims.size(), var_dims.data(), ACL_FORMAT_ND, var_data.data());
  auto input_value = new npuTensor<float>(ACL_FLOAT, value_dims.size(), value_dims.data(), ACL_FORMAT_ND, value_data.data());
  auto input_begin = new npuTensor<int64_t>(ACL_INT64, begin_dims.size(), begin_dims.data(), ACL_FORMAT_ND, begin_data.data());
  auto input_end = new npuTensor<int64_t>(ACL_INT64, end_dims.size(), end_dims.data(), ACL_FORMAT_ND, end_data.data());
  auto input_stride = new npuTensor<int64_t>(ACL_INT64, stride_dims.size(), stride_dims.data(), ACL_FORMAT_ND, stride_data.data());
  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_var->desc);
  input_descs.emplace_back(input_value->desc);
  input_descs.emplace_back(input_begin->desc);
  input_descs.emplace_back(input_end->desc);
  input_descs.emplace_back(input_stride->desc);
  input_buffers.emplace_back(input_var->buffer);
  input_buffers.emplace_back(input_value->buffer);
  input_buffers.emplace_back(input_begin->buffer);
  input_buffers.emplace_back(input_end->buffer);
  input_buffers.emplace_back(input_stride->buffer);

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
  input_var->Destroy();
  input_value->Destroy();
  input_begin->Destroy();
  input_end->Destroy();
  input_stride->Destroy();
  output_y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}