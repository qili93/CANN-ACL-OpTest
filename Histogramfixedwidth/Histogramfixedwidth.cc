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

  // op trangepe
  const std::string op_trangepe = "HistogramFixedWidth";

  // common
  const aclFormat origin_format = ACL_FORMAT_NCHW;
  const aclFormat storage_format = ACL_FORMAT_NCHW;

  // input - x
  const std::vector<int64_t> x_origin_dims{1, 3};
  const std::vector<int64_t> x_storage_dims{1, 3};
  const std::vector<float> x_data(3, 1);
  // input - range
  const std::vector<int64_t> range_origin_dims{1, 2};
  const std::vector<int64_t> range_storage_dims{1, 2};
  const std::vector<float> range_data(2, 0);

  // input - nbins
  const std::vector<int64_t> nbins_origin_dims{1,1};
  const std::vector<int64_t> nbins_storage_dims{1,1};
  const std::vector<int> nbins_data(1, 100);

  // output - out
  const std::vector<int64_t> out_origin_dims{1, 4};
  const std::vector<int64_t> out_storage_dims{1, 4};
  std::vector<int> out_data(4, 0); // output = 2

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

  // input - ranges 
  auto range_desc = aclCreateTensorDesc(ACL_FLOAT,  range_origin_dims.size(), range_origin_dims.data(), origin_format);
  ACL_CALL(aclSetTensorFormat(range_desc, storage_format));
  ACL_CALL(aclSetTensorShape(range_desc, range_storage_dims.size(), range_storage_dims.data()));
  auto range_size = aclGetTensorDescSize(range_desc);
  std::cout << "range_size = " << range_size << std::endl;
  void* range_device_ptr;
  ACL_CALL(aclrtMalloc(&range_device_ptr, range_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(range_device_ptr, range_size, range_data.data(), range_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto range_buffer = aclCreateDataBuffer(range_device_ptr, range_size);

  // input - nbins 
  auto nbins_desc = aclCreateTensorDesc(ACL_INT32,  nbins_origin_dims.size(), nbins_origin_dims.data(), origin_format);
  ACL_CALL(aclSetTensorFormat(nbins_desc, storage_format));
  ACL_CALL(aclSetTensorShape(nbins_desc, nbins_storage_dims.size(), nbins_storage_dims.data()));
  auto nbins_size = aclGetTensorDescSize(nbins_desc);
  std::cout << "nbins_size = " << nbins_size << std::endl;
  void* nbins_device_ptr;
  ACL_CALL(aclrtMalloc(&nbins_device_ptr, nbins_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(nbins_device_ptr, nbins_size, nbins_data.data(), nbins_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto nbins_buffer = aclCreateDataBuffer(nbins_device_ptr, nbins_size);

  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x_desc);
  input_descs.emplace_back(range_desc);
  input_descs.emplace_back(nbins_desc);
  input_buffers.emplace_back(x_buffer);
  input_buffers.emplace_back(range_buffer);
  input_buffers.emplace_back(nbins_buffer);


  // output - out
  auto out_desc = aclCreateTensorDesc(ACL_INT32, out_origin_dims.size(), out_origin_dims.data(), origin_format);
  ACL_CALL(aclSetTensorFormat(out_desc, storage_format));
  ACL_CALL(aclSetTensorShape(out_desc, out_storage_dims.size(), out_storage_dims.data()));
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
  std::cout << "aclopCompileAndExecute : " << op_trangepe << std::endl;
  ACL_CALL(aclopCompileAndExecute(op_trangepe.c_str(), 
            input_descs.size(), input_descs.data(), input_buffers.data(), 
            output_descs.size(), output_descs.data(), output_buffers.data(), 
            attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));

  // srangenc and destrorange stream
  ACL_CALL(aclrtSynchronizeStream(stream));
  ACL_CALL(aclrtDestroyStream(stream));

  ACL_CALL(aclrtMemcpy(out_data.data(), out_size, out_device_ptr, out_size, ACL_MEMCPY_DEVICE_TO_HOST));

  std::cout << "range = [";
  for (int i = 0; i < out_data.size(); ++i) {
    std::cout << out_data[i] << ", ";
  }
  std::cout << "]" << std::endl;

  ACL_CALL(aclDestroyDataBuffer(x_buffer));
  ACL_CALL(aclDestroyDataBuffer(range_buffer));
  ACL_CALL(aclDestroyDataBuffer(nbins_buffer));
  ACL_CALL(aclDestroyDataBuffer(out_buffer));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFree(range_device_ptr));
  ACL_CALL(aclrtFree(nbins_device_ptr));
  ACL_CALL(aclrtFree(out_device_ptr));

  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(range_desc);
  aclDestroyTensorDesc(nbins_desc);
  aclDestroyTensorDesc(out_desc);
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}
