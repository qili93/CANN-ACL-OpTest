#include <iostream>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <numeric>

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
  const std::string op_type = "ArgMaxV2";
  // input - x
  const std::vector<int64_t> x_dims{2, 3, 4};
  std::vector<float> x_data(24);
  std::iota(x_data.begin(), x_data.end(), 0);
  // input - dimension
  const std::vector<int64_t> axis_dims{1};
  const std::vector<int64_t> axis_data{1};
  // output - y
  // std::vector<int64_t> out_dims{2, 1, 4};
  std::vector<int64_t> out_dims{2, 4};
  std::vector<int64_t> out_data(8, 0);
  // attr - shape
  const bool keep_dims = true;

  // input0 - x
  auto x_desc = aclCreateTensorDesc(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_ND);
  auto x_size = aclGetTensorDescSize(x_desc);
  // allocate device memory
  void* x_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&x_device_ptr, x_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CALL(aclrtMemcpy(x_device_ptr, x_size, x_data.data(), x_size, ACL_MEMCPY_HOST_TO_DEVICE));
  auto x_device_buffer = aclCreateDataBuffer(x_device_ptr, x_size);

  // input1 - axis
  auto axis_desc = aclCreateTensorDesc(ACL_INT64, axis_dims.size(), axis_dims.data(), ACL_FORMAT_ND);
  // ACL_CALL(aclSetTensorPlaceMent(axis_desc, ACL_MEMTYPE_HOST));
  auto axis_size = aclGetTensorDescSize(axis_desc);
  // allocate host memory
  void* axis_host_ptr = nullptr;
  ACL_CALL(aclrtMallocHost(&axis_host_ptr, axis_size));
  ACL_CALL(aclrtMemcpy(axis_host_ptr, axis_size, axis_data.data(), axis_size, ACL_MEMCPY_HOST_TO_HOST));
  auto axis_host_buffer = aclCreateDataBuffer(axis_host_ptr, axis_size);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x_desc);
  input_descs.emplace_back(axis_desc);
  input_buffers.emplace_back(x_device_buffer);
  input_buffers.emplace_back(axis_host_buffer);
  // input_buffers.emplace_back(axis_device_buffer);

  // output
  auto out_desc = aclCreateTensorDesc(ACL_INT64, out_dims.size(), out_dims.data(), ACL_FORMAT_ND);
  auto out_size = aclGetTensorDescSize(out_desc);
  // allocate device mem
  void* out_device_ptr = nullptr;
  ACL_CALL(aclrtMalloc(&out_device_ptr, out_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto out_device_buffer = aclCreateDataBuffer(out_device_ptr, out_size);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(out_desc);
  output_buffers.emplace_back(out_device_buffer);
  
  // attr - shape
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrBool(attr, "keep_dims", keep_dims));
  ACL_CALL(aclopSetAttrDataType(attr, "dtype", ACL_INT64));

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
  ACL_CALL(aclrtMemcpy(out_data.data(), out_size, out_device_ptr, out_size, ACL_MEMCPY_DEVICE_TO_HOST));

  // print input data
  std::cout << "input_data = [";
  std::copy(x_data.begin(), x_data.end(), std::ostream_iterator<float>(std::cout, ", "));
  std::cout << "]" << std::endl;

  // print input shape
  std::cout << "input_shape = [";
  std::copy(x_dims.begin(), x_dims.end(), std::ostream_iterator<int64_t>(std::cout, ", "));
  std::cout << "]" << std::endl;

  // print output data
  std::cout << "output_data = [";
  std::copy(out_data.begin(), out_data.end(), std::ostream_iterator<int64_t>(std::cout, ", "));
  std::cout << "]" << std::endl;

  // get output shape
  std::cout << "output_shape = [";
  int output_dim_size = aclGetTensorDescNumDims(output_descs[0]);
  for (int i = 0; i < output_dim_size; i++){
    int64_t dim_value;
    ACL_CALL(aclGetTensorDescDimV2(output_descs[0], i, &dim_value));
    std::cout << dim_value << ", ";
	}
	std::cout << "]" << std::endl;

  // destroy
  ACL_CALL(aclDestroyDataBuffer(x_device_buffer));
  ACL_CALL(aclDestroyDataBuffer(axis_host_buffer));
  ACL_CALL(aclDestroyDataBuffer(out_device_buffer));
  ACL_CALL(aclrtFree(x_device_ptr));
  ACL_CALL(aclrtFreeHost(axis_host_ptr));
  ACL_CALL(aclrtFree(out_device_ptr));
  aclDestroyTensorDesc(x_desc);
  aclDestroyTensorDesc(axis_desc);
  aclDestroyTensorDesc(out_desc);
  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}