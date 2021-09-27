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
  const std::string op_type = "ReduceSum";

  // input - X
  const std::vector<int64_t> x_dims{3, 2, 3, 2};
  std::vector<float> x_data(36);
  std::iota(x_data.begin(), x_data.end(), 0);
  // input - axes
  const std::vector<int64_t> a_dims{1};
  const std::vector<int64_t> axes{1};
  // attr
  const bool keep_dims = true;
  // output - y
  const std::vector<int64_t> y_dims{3, 1, 3, 2};

  // input - x
  auto x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, x_data.data());
  auto a =  new npuTensor<int64_t>(ACL_INT64, a_dims.size(), a_dims.data(), ACL_FORMAT_ND, axes.data(), memType::HOST);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x->desc);
  input_descs.emplace_back(a->desc);
  input_buffers.emplace_back(x->buffer);
  input_buffers.emplace_back(a->buffer);

  // output - y
  auto y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y->desc);
  output_buffers.emplace_back(y->buffer);

  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrBool(attr, "keep_dims", keep_dims));
  
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
  x->Print("x");
  y->Print("y");

  // destroy - inputs
  x->Destroy();
  // destroy - outputs
  y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}