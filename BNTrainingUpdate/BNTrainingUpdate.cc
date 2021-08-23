#include <iostream>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <numeric>

#include "acl/acl.h"
// #include "acl/acl_op.h" // aclopExecuteV2 可以支持动态Shape算子
#include "acl/acl_op_compiler.h" // aclopCompileAndExecute 只能支持固定Shape算子
#include "common/logging.h"

#define ACL_CALL(msg) CHECK_EQ(reinterpret_cast<aclError>(msg), ACL_SUCCESS)

class npuTensor {
 public:
  npuTensor(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format, const float *ptr) {
    desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    size = aclGetTensorDescSize(desc);
    device_ptr = nullptr;
    ACL_CALL(aclrtMalloc(&device_ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    if (ptr != nullptr) {
      ACL_CALL(aclrtMemcpy(device_ptr, size, ptr, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    buffer =  aclCreateDataBuffer(device_ptr, size);
  }
  ~npuTensor() {}
  void Destroy() {
    ACL_CALL(aclDestroyDataBuffer(buffer));
    ACL_CALL(aclrtFree(device_ptr));
    aclDestroyTensorDesc(desc);
  }
  void Print(std::string msg) {
    size_t numel = size / sizeof(float);
    std::vector<float> cpu_data(numel, 0);
    ACL_CALL(aclrtMemcpy(cpu_data.data(), size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST));
    std::cout << msg << " = [";
    for (size_t i = 0; i < cpu_data.size(); ++i) {
      std::cout << cpu_data[i] << ", ";
    }
    std::cout << "]" << std::endl;
  }
public:
  size_t size;
  void * device_ptr;
  aclTensorDesc* desc;
  aclDataBuffer* buffer;
};


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
  const std::string op_type = "BNTrainingUpdate";

  // input - x
  const std::vector<int64_t> x_dims{1, 2, 3, 4};
  std::vector<float> x_data(24);
  std::iota(x_data.begin(), x_data.end(), 0);
  // input - sum
  const std::vector<int64_t> c_dims{2};
  const std::vector<float> sum_data{66.0, 210.0};
  // input - square_sum
  const std::vector<float> square_sum_data{506.0, 3818.0};
  // input - scale
  const std::vector<float> scale_data{1.0, 1.0};
  // input - offset
  const std::vector<float> offset_data{0.0, 0.0};
  // input - mean
  const std::vector<float> mean_data{0.0, 0.0};
  // input - var
  const std::vector<float> var_data{1.0, 1.0};
  // attr - epsilon
  const float epsilon = 1e-5;
  // attr - factor
  const float factor = 0.9;

  // input - x
  auto x = new npuTensor(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, x_data.data());
  auto sum = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, sum_data.data());
  auto square_sum = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, square_sum_data.data());
  auto scale = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, scale_data.data());
  auto offset = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, offset_data.data());
  auto mean = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, mean_data.data());
  auto var = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, var_data.data());

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x->desc);
  input_descs.emplace_back(sum->desc);
  input_descs.emplace_back(square_sum->desc);
  input_descs.emplace_back(scale->desc);
  input_descs.emplace_back(offset->desc);
  input_descs.emplace_back(mean->desc);
  input_descs.emplace_back(var->desc);
  input_buffers.emplace_back(x->buffer);
  input_buffers.emplace_back(sum->buffer);
  input_buffers.emplace_back(square_sum->buffer);
  input_buffers.emplace_back(scale->buffer);
  input_buffers.emplace_back(offset->buffer);
  input_buffers.emplace_back(mean->buffer);
  input_buffers.emplace_back(var->buffer);

  // output - y
  auto y = new npuTensor(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, nullptr);
  auto mean_out = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, nullptr);
  auto var_out = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, nullptr);
  auto saved_mean = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, nullptr);
  auto saved_var = new npuTensor(ACL_FLOAT, c_dims.size(), c_dims.data(), ACL_FORMAT_ND, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y->desc);
  output_descs.emplace_back(mean_out->desc);
  output_descs.emplace_back(var_out->desc);
  output_descs.emplace_back(saved_mean->desc);
  output_descs.emplace_back(saved_var->desc);
  output_buffers.emplace_back(y->buffer);
  output_buffers.emplace_back(mean_out->buffer);
  output_buffers.emplace_back(var_out->buffer);
  output_buffers.emplace_back(saved_mean->buffer);
  output_buffers.emplace_back(saved_var->buffer);

  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrFloat(attr, "factor", factor));
  ACL_CALL(aclopSetAttrFloat(attr, "epsilon", epsilon));
  
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
  y->Print("y");
  mean_out->Print("mean_out");
  var_out->Print("var_out");
  saved_mean->Print("saved_mean");
  saved_var->Print("saved_var");

  // destroy - inputs
  x->Destroy();
  sum->Destroy();
  square_sum->Destroy();
  scale->Destroy();
  offset->Destroy();
  mean->Destroy();
  var->Destroy();
  // destroy - outputs
  y->Destroy();
  mean_out->Destroy();
  var_out->Destroy();
  saved_mean->Destroy();
  saved_var->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}