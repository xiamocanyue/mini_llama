#pragma once 

#include <cstdint>
#include <string>
#include <vector>

namespace mini_llama {
enum GgmlType : uint32_t {  
    kQuantTypeF32 = 0,
    kQuantTypeF16 = 1,
    kQuantTypeQ40 = 2,
    kQuantTypeQ41 = 3,
    kQuantTypeQ80 = 8,
};

const char* GetQuantTypeName(uint32_t type);


enum GgufValueType : uint32_t {
    kGgufTypeUint8 = 0,
    kGgufTypeInt8 = 1,
    kGgufTypeUint16 = 2,
    kGgufTypeInt16 = 3,
    kGgufTypeUint32 = 4,
    kGgufTypeInt32 = 5,
    kGgufTypeFloat32 = 6,
    kGgufTypeBool = 7,
    kGgufTypeString = 8,
    kGgufTypeArray = 9,
    kGgufTypeUint64 = 10,
    kGgufTypeInt64 = 11,
    kGgufTypeFloat64 = 12,
};

const char* GetGgufValueTypeName(uint32_t type);


struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t quant_type = 0;
    uint64_t offset = 0;
};

struct GgufMetaKv {
    std::string key;
    std::string value_str;
    uint32_t data_type = 0;
};

struct GgufReader {
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_metadatas = 0;
    std::vector<GgufMetaKv> metadatas;
    std::vector<GgufTensorInfo> tensor_infos;
    uint64_t data_offset = 0;  //记录tensor data的起点

    bool Load(const std::string& path);

    std::string load_error;
};

void InspectGguf(const GgufReader& reader);

bool GetGgufTensorByteSize(const GgufTensorInfo& info, uint64_t& out);

bool GetGgufTensorByteSize(const GgufTensorInfo& info, uint64_t& out);

std::vector<uint8_t> GetGgufSingleTensor(const std::string& path, const GgufTensorInfo& info, uint64_t data_offset);

// 函数执行后，找到的meta的key,type之后，f指向了value的位置
bool OpenAndFindMetaDataKey(std::ifstream& f, const std::string& path, const std::string& key, uint32_t& out_type);

bool GetGgufMetaDataInt(const std::string& path, const std::string& key, int64_t& out);

bool GetGgufMetaDataFloat(const std::string& path, const std::string& key, double& out);

bool GetGgufMetaDataString(const std::string& path, const std::string& key, std::string& out);

bool GetGgufMetaDataStringArray(const std::string& path, const std::string& key, std::vector<std::string>& out);

bool GetGgufMetaDataIntArray(const std::string& path, const std::string& key, std::vector<int32_t>& out);
}