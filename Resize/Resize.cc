#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "common/logging.h"

#define ACL_CALL(msg) CHECK_EQ(reinterpret_cast<aclError>(msg), ACL_SUCCESS)

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
  const std::string op_type = "Resize";
  // input - x
  const std::vector<int64_t> x_dims{1, 1, 2, 3};
  std::vector<float> x{1, 2, 3, 4, 5, 6};
  // input - roi
  const std::vector<int64_t> roi_dims{1};
  std::vector<float> roi{0.1};
  // input - scales
  const std::vector<int64_t> scales_dims{2};
  std::vector<float> scales{1.5, 1.5};
  // input - sizes
  const std::vector<int64_t> sizes_dims{2};
  std::vector<int64_t> sizes{3, 3};
  // output
  const std::vector<int64_t> y_dims{1, 1, 3, 3};
  std::vector<float> y{0, 0, 0, 0, 0, 0, 0, 0, 0};


  // input - x
  auto x_desc = aclCreateTensorDesc(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW);
  ACL_CALL(aclSetTensorStorageFormat(x_desc, ACL_FORMAT_NCHW));
  ACL_CALL(aclSetTensorOriginFormat(x_desc, ACL_FORMAT_NCHW));
  ACL_CALL(aclSetTensorFormat(x_desc, ACL_FORMAT_NCHW));
  auto x_size = aclGetTensorDescSize(x_desc);
  auto x_host_buffer = aclCreateDataBuffer(x.data(), x_size);
  // void* x_host_ptr;
  // ACL_CALL(aclrtMallocHost(&x_host_ptr, x_size));
  // ACL_CALL(aclrtMemcpy(x_host_ptr, x_size, x.data(), x_size, ACL_MEMCPY_HOST_TO_HOST));
  // auto x_host_buffer = aclCreateDataBuffer(x_host_ptr, x_size);
  void* x_device_ptr;
  ACL_CALL(aclrtMalloc(&x_device_ptr, x_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x_device_ptr, x_size, x.data(), x_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x_device_buffer = aclCreateDataBuffer(x_device_ptr, x_size);
  // input - roi
  auto roi_desc = aclCreateTensorDesc(ACL_FLOAT, roi_dims.size(), roi_dims.data(), ACL_FORMAT_ND);
  auto roi_size = aclGetTensorDescSize(roi_desc);
  // void* roi_host_ptr;
  // ACL_CALL(aclrtMallocHost(&roi_host_ptr, roi_size));
  // ACL_CALL(aclrtMemcpy(roi_host_ptr, roi_size, roi.data(), roi_size, ACL_MEMCPY_HOST_TO_HOST));
  auto roi_host_buffer = aclCreateDataBuffer(roi.data(), roi_size);
  // input - scales
  auto scales_desc = aclCreateTensorDesc(ACL_FLOAT, scales_dims.size(), scales_dims.data(), ACL_FORMAT_ND);
  auto scales_size = aclGetTensorDescSize(scales_desc);
  // ACL_CALL(aclSetTensorPlaceMent(scales_desc, ACL_MEMTYPE_HOST));
  auto scales_host_buffer = aclCreateDataBuffer(scales.data(), scales_size);
  // input - sizes
  auto sizes_desc = aclCreateTensorDesc(ACL_INT64, sizes_dims.size(), sizes_dims.data(), ACL_FORMAT_ND);
  ACL_CALL(aclSetTensorPlaceMent(sizes_desc, ACL_MEMTYPE_HOST));
  auto sizes_size = aclGetTensorDescSize(sizes_desc);
  auto sizes_host_buffer = aclCreateDataBuffer(sizes.data(), sizes_size);
  void* sizes_device_ptr;
  ACL_CALL(aclrtMalloc(&sizes_device_ptr, sizes_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(sizes_device_ptr, sizes_size, sizes.data(), sizes_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto sizes_device_buffer = aclCreateDataBuffer(sizes_device_ptr, sizes_size);
  

  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_host_buffers;
  input_descs.emplace_back(x_desc);
  input_descs.emplace_back(roi_desc);
  input_descs.emplace_back(scales_desc);
  input_descs.emplace_back(sizes_desc);
  input_host_buffers.emplace_back(x_host_buffer);
  input_host_buffers.emplace_back(roi_host_buffer);
  input_host_buffers.emplace_back(scales_host_buffer);
  input_host_buffers.emplace_back(sizes_host_buffer);

  // output - y
  auto y_desc = aclCreateTensorDesc(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW);
  auto y_size = aclGetTensorDescSize(y_desc);
  void* y_device_ptr;
  ACL_CALL(aclrtMalloc(&y_device_ptr, y_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto y_device_buffer = aclCreateDataBuffer(y_device_ptr, y_size);
  // outputs
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y_desc);
  output_buffers.emplace_back(y_device_buffer);

  // attributes
  auto attr = aclopCreateAttr();
  // aclopSetAttrListInt(attr, "sizes", sizes.size(), sizes.data());
  ACL_CALL(aclopSetAttrString(attr, "coordinate_transformation_mode", "align_corners"));
  ACL_CALL(aclopSetAttrFloat(attr, "cubic_coeff_a", -0.75));
  ACL_CALL(aclopSetAttrInt(attr, "exclude_outside", 0));
  ACL_CALL(aclopSetAttrFloat(attr, "extrapolation_value", 0));
  ACL_CALL(aclopSetAttrString(attr, "mode", "nearest"));
  ACL_CALL(aclopSetAttrString(attr, "nearest_mode", "round_prefer_floor"));

  // std::cout << "aclopCompileAndExecute : " << op_type << std::endl;

  // set aclSetTensorDescName for dynamic input
  // aclSetTensorDescName(input_descs[0], "x");
  // aclSetTensorDescName(input_descs[1], "roi");
  // aclSetTensorDescName(input_descs[2], "scales");
  // aclSetTensorDescName(input_descs[3], "sizes");
  // aclSetTensorDescName(output_descs[0], "y");

  std::cout << "input x data format is: " << aclGetTensorDescFormat(input_descs[0]) << std::endl;

  // infer shape
  ACL_CALL(aclopInferShape(op_type.c_str(), 
            input_descs.size(), input_descs.data(), input_host_buffers.data(), 
            output_descs.size(), output_descs.data(), attr));

  size_t dim_size = aclGetTensorDescNumDims(output_descs[0]);
  std::cout << "dim_size = " << dim_size << std::endl;
  for (size_t i = 0; i < dim_size; ++i) {
    int64_t dim_value;
    ACL_CALL(aclGetTensorDescDimV2(output_descs[0], i, &dim_value));
    std::cout << "dim[" << i << "] = " << dim_value << std::endl;
  }

  // // create stream
  // aclrtStream stream = nullptr;
  // ACL_CALL(aclrtCreateStream(&stream));
  // ACL_CALL(aclopCompileAndExecute(op_type.c_str(), 
  //           input_descs.size(), input_descs.data(), input_buffers.data(), 
  //           output_descs.size(), output_descs.data(), output_buffers.data(), 
  //           attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));
  // ACL_CALL(aclrtSynchronizeStream(stream));
  // ACL_CALL(aclrtDestroyStream(stream));

  // ACL_CALL(aclrtMemcpy(y.data(), y_size, y_device_ptr, y_size, ACL_MEMCPY_DEVICE_TO_HOST));

  std::cout << "y = [";
  for (int i = 0; i < y.size(); ++i) {
    std::cout << y[i] << ", ";
  }
  std::cout << "]" << std::endl;

  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(roi_desc);
  // aclDestroyTensorDesc(scales_desc);
  aclDestroyTensorDesc(sizes_desc);
  aclDestroyTensorDesc(y_desc);
  aclopDestroyAttr(attr);

  ACL_CALL(aclDestroyDataBuffer(x_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(x_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(roi_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(scales_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(sizes_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(sizes_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(y_device_buffer));

  // ACL_CALL(aclrtFreeHost(x_host_ptr));
  // ACL_CALL(aclrtFreeHost(roi_host_ptr));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFree(sizes_device_ptr));
  ACL_CALL(aclrtFree(y_device_ptr));

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}