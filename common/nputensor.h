#include "acl/acl.h"
// #include "acl/acl_op.h" // aclopExecuteV2 可以支持动态Shape算子
#include "acl/acl_op_compiler.h" // aclopCompileAndExecute 只能支持固定Shape算子

#include "common/logging.h"

#define ACL_CALL(msg) CHECK_EQ(reinterpret_cast<aclError>(msg), ACL_SUCCESS)

typedef enum {
    DEVICE = 0,
    HOST = 1,
} memType;

template <typename T>
class npuTensor {
 public:
  npuTensor(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format, 
            const T *ptr, const memType mem_type = memType::DEVICE) {
    desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    size = aclGetTensorDescSize(desc);
    device_ptr = nullptr;
    host_ptr = nullptr;
    mem_type_ = mem_type;

    if (mem_type == memType::DEVICE) {
      ACL_CALL(aclrtMalloc(&device_ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
      if (ptr != nullptr) {
        ACL_CALL(aclrtMemcpy(device_ptr, size, ptr, size, ACL_MEMCPY_HOST_TO_DEVICE));
      }
      buffer =  aclCreateDataBuffer(device_ptr, size);
    }

    if (mem_type == memType::HOST) {
      ACL_CALL(aclSetTensorPlaceMent(desc, ACL_MEMTYPE_HOST));
      ACL_CALL(aclrtMallocHost(&host_ptr, size));
      if (ptr != nullptr) {
        ACL_CALL(aclrtMemcpy(host_ptr, size, ptr, size, ACL_MEMCPY_HOST_TO_HOST));
      }
      buffer =  aclCreateDataBuffer(host_ptr, size);
      // ACL_CALL(aclSetTensorConst(desc, buffer, size));
    }
  }

  ~npuTensor() {}
  void Destroy() {
    ACL_CALL(aclDestroyDataBuffer(buffer));
    if(device_ptr != nullptr) {
      ACL_CALL(aclrtFree(device_ptr));
    }
    if (host_ptr != nullptr) {
      ACL_CALL(aclrtFreeHost(host_ptr));
    }
    aclDestroyTensorDesc(desc);
  }

  void Print(std::string msg) {
    size_t numel = size / sizeof(T);
    std::vector<T> cpu_data(numel, 0);
    if (mem_type_ == memType::DEVICE) {
      ACL_CALL(aclrtMemcpy(cpu_data.data(), size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
      ACL_CALL(aclrtMemcpy(cpu_data.data(), size, host_ptr, size, ACL_MEMCPY_HOST_TO_HOST));
    }
    std::cout << msg << " = [";
    for (size_t i = 0; i < cpu_data.size(); ++i) {
      std::cout << cpu_data[i] << ", ";
    }
    std::cout << "]" << std::endl;
  }
public:
  size_t size;
  void * host_ptr;
  void * device_ptr;
  aclTensorDesc* desc;
  aclDataBuffer* buffer;
  memType mem_type_;
};

static int64_t get_numel(const std::vector<int64_t>& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
}