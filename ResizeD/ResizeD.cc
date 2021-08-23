#include <iostream>
#include <vector>

#include "acl/acl.h"
// #include "acl/acl_op.h" // aclopExecuteV2 可以支持动态Shape算子
#include "acl/acl_op_compiler.h" // aclopCompileAndExecute 只能支持固定Shape算子
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
  const std::string op_type = "ResizeD";
  // input - x
  const std::vector<int64_t> x_dims{1, 1, 2, 3};
  std::vector<float> x{1, 2, 3, 4, 5, 6};
  // output - y
  const std::vector<int64_t> y_dims{1, 1, 3, 3};
  std::vector<float> y{0, 0, 0, 0, 0, 0, 0, 0, 0};
  // attr - sizes
  const std::vector<int64_t> sizes{3, 3};
  const std::vector<float> scales{1.5, 1.0};
  const std::vector<int64_t> roi{};

  // input0 - x - should use device buffer
  auto x_desc = aclCreateTensorDesc(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW);
  auto x_size = aclGetTensorDescSize(x_desc);
  // alloc host memory
  // void* x_host_ptr = nullptr;
  // ACL_CALL(aclrtMallocHost(&x_host_ptr, x_size));
  // ACL_CALL(aclrtMemcpy(x_host_ptr, x_size, x.data(), x_size, ACL_MEMCPY_HOST_TO_HOST));
  auto x_host_buffer = aclCreateDataBuffer(x.data(), x_size);
  // alloc device memory
  void* x_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&x_device_ptr, x_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x_device_ptr, x_size, x.data(), x_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x_device_buffer = aclCreateDataBuffer(x_device_ptr, x_size);
  // inputs
  std::vector<aclTensorDesc *> input_descs;
  // std::vector<aclDataBuffer *> input_host_buffers;
  std::vector<aclDataBuffer *> input_device_buffers;
  input_descs.emplace_back(x_desc);
  // input_host_buffers.emplace_back(x_host_buffer);
  input_device_buffers.emplace_back(x_device_buffer);

  // output - y
  auto y_desc = aclCreateTensorDesc(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW);
  auto y_size = aclGetTensorDescSize(y_desc);
  // alloc device memory
  void* y_device_ptr;
  ACL_CALL(aclrtMalloc(&y_device_ptr, y_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto y_device_buffer = aclCreateDataBuffer(y_device_ptr, y_size);
  // outputs
  std::vector<aclTensorDesc *> output_descs;
  // std::vector<aclDataBuffer *> output_host_buffers;
  std::vector<aclDataBuffer *> output_device_buffers;
  output_descs.emplace_back(y_desc);
  output_device_buffers.emplace_back(y_device_buffer);

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

  // infer shape
  // ACL_CALL(aclopInferShape(op_type.c_str(), 
  //           input_descs.size(), input_descs.data(), input_host_buffers.data(), 
  //           output_descs.size(), output_descs.data(), attr));

  // size_t dim_size = aclGetTensorDescNumDims(output_descs[0]);
  // std::cout << "dim_size = " << dim_size << std::endl;
  // for (size_t i = 0; i < dim_size; ++i) {
  //   int64_t dim_value;
  //   ACL_CALL(aclGetTensorDescDimV2(output_descs[0], i, &dim_value));
  //   std::cout << "dim[" << i << "] = " << dim_value << std::endl;
  // }

  std::cout << "aclopCompileAndExecute : " << op_type << std::endl;

  // create stream
  aclrtStream stream = nullptr;
  ACL_CALL(aclrtCreateStream(&stream));
  ACL_CALL(aclopCompileAndExecute(op_type.c_str(), 
            input_descs.size(), input_descs.data(), input_device_buffers.data(), 
            output_descs.size(), output_descs.data(), output_device_buffers.data(), 
            attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  ACL_CALL(aclrtDestroyStream(stream));

  ACL_CALL(aclrtMemcpy(y.data(), y_size, y_device_ptr, y_size, ACL_MEMCPY_DEVICE_TO_HOST));

  std::cout << "y = [";
  for (size_t i = 0; i < y.size(); ++i) {
    std::cout << y[i] << ", ";
  }
  std::cout << "]" << std::endl;
  
  ACL_CALL(aclDestroyDataBuffer(x_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(x_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(y_device_buffer));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFree(y_device_ptr));

  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(y_desc);
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}