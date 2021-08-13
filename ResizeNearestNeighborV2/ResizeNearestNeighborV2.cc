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
  const std::string op_type = "ResizeNearestNeighborV2";
  // input - x
  const std::vector<int64_t> x_dims{1, 1, 2, 3};
  const std::vector<float> x{1, 2, 3, 4, 5, 6};
  // input - sizes
  const std::vector<int64_t> sizes_dims{2};
  const std::vector<int32_t> sizes{3, 3};
  // output
  const std::vector<int64_t> y_dims{1, 1, 3, 3};
  std::vector<float> y{0, 0, 0, 0, 0, 0, 0, 0, 0};

  // input0 - x - should use device buffer
  auto x_desc = aclCreateTensorDesc(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW);
  auto x_size = aclGetTensorDescSize(x_desc);
  // assign host value
  void* x_host_ptr = nullptr;
  ACL_CALL(aclrtMallocHost(&x_host_ptr, x_size));
  ACL_CALL(aclrtMemcpy(x_host_ptr, x_size, x.data(), x_size, ACL_MEMCPY_HOST_TO_HOST));
  auto x_host_buffer = aclCreateDataBuffer(x_host_ptr, x_size);
  // memcopy to device
  void* x_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&x_device_ptr, x_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x_device_ptr, x_size, x.data(), x_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x_device_buffer = aclCreateDataBuffer(x_device_ptr, x_size);

  // input1 - sizes - should use host buffer
  auto sizes_desc = aclCreateTensorDesc(ACL_INT32, sizes_dims.size(), sizes_dims.data(), ACL_FORMAT_NCHW);
  ACL_CALL(aclSetTensorPlaceMent(sizes_desc, ACL_MEMTYPE_HOST));
  auto sizes_size = aclGetTensorDescSize(sizes_desc);
  // host buffer
  void* sizes_host_ptr = nullptr;
  ACL_CALL(aclrtMallocHost(&sizes_host_ptr, sizes_size));
  ACL_CALL(aclrtMemcpy(sizes_host_ptr, sizes_size, sizes.data(), sizes_size, ACL_MEMCPY_HOST_TO_HOST));
  auto sizes_host_buffer = aclCreateDataBuffer(sizes_host_ptr, sizes_size);
  // device buffer
  void* sizes_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&sizes_device_ptr, sizes_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(sizes_device_ptr, sizes_size, sizes.data(), sizes_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto sizes_device_buffer = aclCreateDataBuffer(sizes_device_ptr, sizes_size);

  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x_desc);
  input_descs.emplace_back(sizes_desc);

  // input_buffers.emplace_back(x_host_buffer);
  input_buffers.emplace_back(x_device_buffer);
  input_buffers.emplace_back(sizes_host_buffer);
  // input_buffers.emplace_back(sizes_device_buffer);

  // output0 - y - should use device buffer
  auto y_desc = aclCreateTensorDesc(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW);
  auto y_size = aclGetTensorDescSize(y_desc);
  // host
  void* y_host_ptr = nullptr;
  ACL_CALL(aclrtMallocHost(&y_host_ptr, y_size));
  auto y_host_buffer = aclCreateDataBuffer(y_host_ptr, y_size);
  // device
  void* y_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&y_device_ptr, y_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto y_device_buffer = aclCreateDataBuffer(y_device_ptr, y_size);
  // outputs
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y_desc);
  // output_buffers.emplace_back(y_host_buffer);
  output_buffers.emplace_back(y_device_buffer);
  
  // attributes
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrBool(attr, "align_corners", true));
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
  // copy output from device to host
  ACL_CALL(aclrtMemcpy(y.data(), y_size, y_device_ptr, y_size, ACL_MEMCPY_DEVICE_TO_HOST));

  // print output
  std::cout << "y = [";
  for (size_t i = 0; i < y.size(); ++i) {
    std::cout << y[i] << ", ";
  }
  std::cout << "]" << std::endl;

  ACL_CALL(aclDestroyDataBuffer(x_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(x_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(sizes_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(sizes_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(y_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(y_device_buffer));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFree(sizes_device_ptr));
  ACL_CALL(aclrtFree(y_device_ptr));
  ACL_CALL(aclrtFreeHost(x_host_ptr));
  ACL_CALL(aclrtFreeHost(sizes_host_ptr));
  ACL_CALL(aclrtFreeHost(y_host_ptr));

  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(sizes_desc);
  aclDestroyTensorDesc(y_desc);

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}