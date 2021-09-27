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
  const std::string op_type = "BatchMatMul";
  // shape
  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 5;
  // input - x1
  const std::vector<int64_t> x1_dims{3, 1, M, K};
  std::vector<float> x1_data(3 * M * K);
  std::iota(x1_data.begin(), x1_data.end(), 0);
  // input - x2
  const std::vector<int64_t> x2_dims{1, 2, K, N};
  std::vector<float> x2_data(2 * K * N);
  std::iota(x2_data.begin(), x2_data.end(), 0);
  // attr
  const bool trans_x1 = false;
  const bool trans_x2 = false;
  // output - y
  const std::vector<int64_t> y_dims{3, 2, M, N};

  // input - x
  auto x1 = new npuTensor<float>(ACL_FLOAT, x1_dims.size(), x1_dims.data(), ACL_FORMAT_NCHW, x1_data.data());
  auto x2 = new npuTensor<float>(ACL_FLOAT, x2_dims.size(), x2_dims.data(), ACL_FORMAT_NCHW, x2_data.data());

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x1->desc);
  input_descs.emplace_back(x2->desc);
  input_buffers.emplace_back(x1->buffer);
  input_buffers.emplace_back(x2->buffer);

  // output - y
  auto y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y->desc);
  output_buffers.emplace_back(y->buffer);

  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrBool(attr, "adj_x1", trans_x1));
  ACL_CALL(aclopSetAttrBool(attr, "adj_x2", trans_x2));
  
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
  x1->Print("x1");
  x2->Print("x2");
  y->Print("y");

  // destroy - inputs
  x1->Destroy();
  x2->Destroy();
  // destroy - outputs
  y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}