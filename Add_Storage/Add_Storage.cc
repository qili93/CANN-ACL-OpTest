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

  // op type
  const std::string op_type = "Add";

  // common
  const aclFormat origin_format = ACL_FORMAT_NCHW;
  const aclFormat storage_format = ACL_FORMAT_NC1HWC0;

  // input - x
  const std::vector<int64_t> x_origin_dims{4, 6, 4, 4};
  const std::vector<int64_t> x_storage_dims{4, 1, 4, 4, 16};
  const std::vector<float> x_data(64*16, 1);

  // input - y
  const std::vector<int64_t> y_origin_dims{6};
  const std::vector<int64_t> y_storage_dims{1, 1, 1, 1, 16};
  const std::vector<float> y_data(16, 1);

  // output - out
  // const std::vector<int64_t> origin_dims{4, 6, 4, 4};
  // const std::vector<int64_t> storage_dims{4, 1, 4, 4, 16};
  std::vector<float> out_data(64*16, 0); // output = 2

  // input - x
  auto x_desc = aclCreateTensorDesc(ACL_FLOAT, x_origin_dims.size(), x_origin_dims.data(), origin_format);
  ACL_CALL(aclSetTensorFormat(x_desc, storage_format));
  ACL_CALL(aclSetTensorShape(x_desc, x_storage_dims.size(), x_storage_dims.data()));
  auto x_size = aclGetTensorDescSize(x_desc);
  std::cout << "x_size = " << x_size << std::endl;
  void* x_device_ptr;
  ACL_CALL(aclrtMalloc(&x_device_ptr, x_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x_device_ptr, x_size, x_data.data(), x_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x_buffer = aclCreateDataBuffer(x_device_ptr, x_size);

  // input - x
  auto y_desc = aclCreateTensorDesc(ACL_FLOAT, y_origin_dims.size(), y_origin_dims.data(), origin_format);
  ACL_CALL(aclSetTensorFormat(y_desc, storage_format));
  ACL_CALL(aclSetTensorShape(y_desc, y_storage_dims.size(), y_storage_dims.data()));
  auto y_size = aclGetTensorDescSize(y_desc);
  std::cout << "y_size = " << y_size << std::endl;
  void* y_device_ptr;
  ACL_CALL(aclrtMalloc(&y_device_ptr, y_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(y_device_ptr, y_size, y_data.data(), y_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto y_buffer = aclCreateDataBuffer(y_device_ptr, y_size);

  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x_desc);
  input_descs.emplace_back(y_desc);
  input_buffers.emplace_back(x_buffer);
  input_buffers.emplace_back(y_buffer);


  // output - out
  auto out_desc = aclCreateTensorDesc(ACL_FLOAT, x_origin_dims.size(), x_origin_dims.data(), origin_format);
  ACL_CALL(aclSetTensorFormat(out_desc, storage_format));
  ACL_CALL(aclSetTensorShape(out_desc, x_storage_dims.size(), x_storage_dims.data()));
  auto out_size = aclGetTensorDescSize(out_desc);
  std::cout << "out_size = " << out_size << std::endl;
  void* out_device_ptr;
  ACL_CALL(aclrtMalloc(&out_device_ptr, out_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto out_buffer = aclCreateDataBuffer(out_device_ptr, out_size);

  // outputs
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(out_desc);
  output_buffers.emplace_back(out_buffer);

  // attributes
  auto attr = aclopCreateAttr();

  // create stream
  aclrtStream stream = nullptr;
  ACL_CALL(aclrtCreateStream(&stream));

  // run operator
  std::cout << "aclopCompileAndExecute : " << op_type << std::endl;
  ACL_CALL(aclopCompileAndExecute(op_type.c_str(), 
            input_descs.size(), input_descs.data(), input_buffers.data(), 
            output_descs.size(), output_descs.data(), output_buffers.data(), 
            attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));

  // sync and destroy stream
  ACL_CALL(aclrtSynchronizeStream(stream));
  ACL_CALL(aclrtDestroyStream(stream));

  ACL_CALL(aclrtMemcpy(out_data.data(), out_size, out_device_ptr, out_size, ACL_MEMCPY_DEVICE_TO_HOST));

  std::cout << "y = [";
  for (int i = 0; i < out_data.size(); ++i) {
    std::cout << out_data[i] << ", ";
  }
  std::cout << "]" << std::endl;

  ACL_CALL(aclDestroyDataBuffer(x_buffer));
  ACL_CALL(aclDestroyDataBuffer(y_buffer));
  ACL_CALL(aclDestroyDataBuffer(out_buffer));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFree(y_device_ptr));
  ACL_CALL(aclrtFree(out_device_ptr));

  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(y_desc);
  aclDestroyTensorDesc(out_desc);
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}