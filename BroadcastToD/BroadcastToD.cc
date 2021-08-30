#include <iostream>
#include <vector>

#include "common/nputensor.h"

// [ERROR] GE(32344,BroadcastToD):2021-08-23-19:16:43.726.946 [/home/jenkins/agent/workspace/Compile_GraphEngine_Centos_X86/graphengine/ge/engine_manager/dnnengine_manager.cc:273]32344 GetDNNEngineName: ErrorNo: 1343242282(assign engine failed) [COMP][SUB_OPT][Check][OpSupported]Op type BroadcastToD of ops kernel AIcoreEngine is unsupported, reason : The reason why this op BroadcastToD is not supported by op information library[tbe-custom] is that op type BroadcastToD is not found.
// The reason why this op BroadcastToD is not supported by op information library[tbe-builtin] is that [Dynamic check]: data type DT_INT64 of input [x] is not supported. All supported data type and format of tensor input0.x is:
// Data Type: {}
// Format:{}
// [Static check]: data type DT_INT64 of input [x] is not supported. All supported data type and format of tensor input0.x is:
// Data Type: {DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}
// Format:{ND,ND,ND,ND,ND}

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
  const std::string op_type = "BroadcastToD";
  // input - x
  const std::vector<int64_t> x_dims{3, 1, 1};
  const std::vector<float>   x_data{1, 2, 3};
  // output - y
  const std::vector<int64_t> y_dims{3, 2, 4};
  // attr - shape
  const std::vector<int64_t> shape{3, 2, 4};

  // input - x
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_ND, x_data.data(), memType::DEVICE);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_buffers.emplace_back(input_x->buffer);

  // output - out
  auto output_y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_ND, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y->desc);
  output_buffers.emplace_back(output_y->buffer);
  
  // attr - shape
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrListInt(attr, "shape", shape.size(), shape.data()));

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
  output_y->Destroy();

  // destrpy - attr
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}