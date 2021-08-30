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
  const std::string op_type = "DeformableOffsets";
  const int64_t kernel_h = 3;
  const int64_t kernel_w = 3;
  // input - x
  const std::vector<int64_t> x_dims{4, 3, 10, 10};
  const int64_t x_numel = get_numel(x_dims);
  std::vector<float> x_data(x_numel);
  std::iota(x_data.begin(), x_data.end(), 0);
  // input - offset
  const std::vector<int64_t> offset_dims{4, 3 * kernel_h * kernel_w, 8, 8};
  const int64_t offset_numel = get_numel(offset_dims);
  const std::vector<float> offset_data(offset_numel, 0.0);
  // output
  const std::vector<int64_t> y_dims{4, 3, 8 * 3, 8 * 3};
  const int64_t y_numel = get_numel(y_dims);
  // attr
  const std::vector<int64_t> ksize{kernel_h, kernel_w};
  const std::vector<int64_t> strides{1, 1, 1, 1};
  const std::vector<int64_t> pads{0, 0, 0, 0};
  const std::vector<int64_t> dilations{1, 1, 1, 1};

  // inputs
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, x_data.data());
  auto input_offset = new npuTensor<float>(ACL_FLOAT, offset_dims.size(), offset_dims.data(), ACL_FORMAT_NCHW, offset_data.data());
  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_descs.emplace_back(input_offset->desc);
  input_buffers.emplace_back(input_x->buffer);
  input_buffers.emplace_back(input_offset->buffer);

  // output
  auto output_y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW, nullptr);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y->desc);
  output_buffers.emplace_back(output_y->buffer);
  
  // attributes
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrListInt(attr, "ksize", ksize.size(), ksize.data()));
  ACL_CALL(aclopSetAttrListInt(attr, "strides", strides.size(), strides.data()));
  ACL_CALL(aclopSetAttrListInt(attr, "pads", pads.size(), pads.data()));
  ACL_CALL(aclopSetAttrListInt(attr, "dilations", dilations.size(), dilations.data()));
  ACL_CALL(aclopSetAttrInt(attr, "deformable_groups", 1));
  ACL_CALL(aclopSetAttrString(attr, "data_format", "NCHW"));
  ACL_CALL(aclopSetAttrBool(attr, "modulated", true));

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
//   output_y->Print("y");

  // destroy
  input_x->Destroy();
  input_offset->Destroy();
  output_y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}