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
  const std::string op_type = "Sort";
  // input - x
  const std::vector<int64_t> x_dims{2, 3};
  std::vector<int64_t> x_data{1, 2, 3, 4, 5, 6};
  // output - y
  std::vector<int64_t> y_dims{2, 3};

  // inputs
  auto input_x = new npuTensor<int64_t>(ACL_INT64, x_dims.size(), x_dims.data(), ACL_FORMAT_ND, x_data.data());
  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_buffers.emplace_back(input_x->buffer);

  // output
  auto output_y1 = new npuTensor<int64_t>(ACL_INT64, y_dims.size(), y_dims.data(), ACL_FORMAT_ND, nullptr);
  auto output_y2 = new npuTensor<int32_t>(ACL_INT32, y_dims.size(), y_dims.data(), ACL_FORMAT_ND, nullptr);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y1->desc);
  output_descs.emplace_back(output_y2->desc);
  output_buffers.emplace_back(output_y1->buffer);
  output_buffers.emplace_back(output_y2->buffer);

  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrInt(attr, "axis", -1));
  ACL_CALL(aclopSetAttrBool(attr, "descending", false));

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
  output_y1->Print("y1");
  output_y2->Print("y2");

  // destroy
  input_x->Destroy();
  output_y1->Destroy();
  output_y2->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}