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
  const std::string op_type = "ResizeBilinearV2";
  // input - x
  const std::vector<int64_t> x_dims{1, 1, 2, 3};
  const std::vector<float> x_data{1, 2, 3, 4, 5, 6};
  // input - sizes
  const std::vector<int64_t> sizes_dims{2};
  const std::vector<int64_t> sizes_data{3, 3};
  // output
  const std::vector<int64_t> y_dims{1, 1, 3, 3};

  // input - x
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, x_data.data());
  auto input_sizes = new npuTensor<int64_t>(ACL_INT64, sizes_dims.size(), sizes_dims.data(), ACL_FORMAT_ND, sizes_data.data(), memType::HOST);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_descs.emplace_back(input_sizes->desc);
  input_buffers.emplace_back(input_x->buffer);
  input_buffers.emplace_back(input_sizes->buffer);

  // output - out
  auto output_y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y->desc);
  output_buffers.emplace_back(output_y->buffer);

  // attributes
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrBool(attr, "align_corners", false));
  ACL_CALL(aclopSetAttrBool(attr, "half_pixel_centers", false));

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
  input_sizes->Print("sizes");
  output_y->Print("y");

  // destroy
  input_x->Destroy();
  input_sizes->Destroy();
  output_y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}