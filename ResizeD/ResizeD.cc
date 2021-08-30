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
  const std::string op_type = "ResizeD";
  // input - x
  const std::vector<int64_t> x_dims{1, 1, 2, 3};
  std::vector<float> x_data{1, 2, 3, 4, 5, 6};
  // output - y
  const std::vector<int64_t> y_dims{1, 1, 3, 3};
  // attr - sizes
  const std::vector<int64_t> sizes{3, 3};
  const std::vector<float>   scales{1.5, 1.0};
  const std::vector<int64_t> roi{0, 0};

  // input - x
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, x_data.data(), memType::HOST);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_buffers.emplace_back(input_x->buffer);

  // output - out
  auto output_y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW, nullptr, memType::HOST);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y->desc);
  output_buffers.emplace_back(output_y->buffer);

  // attributes
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrListInt(attr, "sizes", sizes.size(), sizes.data()));
  ACL_CALL(aclopSetAttrListFloat(attr, "scales", scales.size(), scales.data()));
  ACL_CALL(aclopSetAttrListInt(attr, "roi", roi.size(), roi.data()));
  ACL_CALL(aclopSetAttrString(attr, "coordinate_transformation_mode", "align_corners"));
  ACL_CALL(aclopSetAttrFloat(attr, "cubic_coeff_a", -0.75));
  ACL_CALL(aclopSetAttrInt(attr, "exclude_outside", 0));
  ACL_CALL(aclopSetAttrFloat(attr, "extrapolation_value", 0.0));
  ACL_CALL(aclopSetAttrString(attr, "mode", "nearest"));
  ACL_CALL(aclopSetAttrString(attr, "nearest_mode", "round_prefer_floor"));

  std::cout << "aclopInferShape : " << op_type << std::endl;
  ACL_CALL(aclopInferShape(op_type.c_str(), 
            input_descs.size(), input_descs.data(), input_buffers.data(), 
            output_descs.size(), output_descs.data(), attr));

  size_t out_dim_num = aclGetTensorDescNumDims(output_descs[0]);
  std::cout << "out_dim_num = " << out_dim_num << std::endl;
  for (size_t i = 0; i < out_dim_num; ++i) {
    int64_t dim_value;
		ACL_CALL(aclGetTensorDescDimV2(output_descs[0], i, &dim_value));
    std::cout << "dim_value[" << i << "] = " << dim_value << std::endl;
  }


  // // create stream
  // aclrtStream stream = nullptr;
  // ACL_CALL(aclrtCreateStream(&stream));

  // std::cout << "aclopExecuteV2 : " << op_type << std::endl;
  // ACL_CALL(aclopExecuteV2(op_type.c_str(), 
  //           input_descs.size(), input_descs.data(), input_buffers.data(), 
  //           output_descs.size(), output_descs.data(), output_buffers.data(), 
  //           attr, stream));

  // std::cout << "aclopCompileAndExecute : " << op_type << std::endl;
  // ACL_CALL(aclopCompileAndExecute(op_type.c_str(), 
  //           input_descs.size(), input_descs.data(), input_buffers.data(), 
  //           output_descs.size(), output_descs.data(), output_buffers.data(), 
  //           attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));

  // sync and destroy stream
  // ACL_CALL(aclrtSynchronizeStream(stream));
  // ACL_CALL(aclrtDestroyStream(stream));

  // print output
  // output_y->Print("y");

  // destroy
  input_x->Destroy();
  output_y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}