#include "include/gguf.h"

#include <cstring>
#include <limits>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <iostream>


namespace mini_llama {
namespace {

constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;


template <typename T>
bool ReadLe(std::ifstream& f, T& out) {
    char buf[sizeof(T)];
    if (!f.read(buf, sizeof(T))) {
        return false;
    }
    std::memcpy(&out, buf, sizeof(T));
    return true;
}

bool ReadU32(std::ifstream& f, uint32_t& out) {
    return ReadLe(f, out);
}

bool ReadU64(std::ifstream& f, uint64_t& out) {
    return ReadLe(f, out);
}

bool ReadString(std::ifstream& f, std::string& out) {
    uint64_t len = 0;
    if (!ReadLe(f, len)) {
        return false;
    }
    if (len < 0 || len > kMaxStringBytes || len > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    
    out.resize(len);
    if (!f.read(out.data(), static_cast<std::streamsize>(len))) {
        return false;
    }
    return true;
}

bool SkipValue(std::ifstream& f, uint32_t type) {
    switch (type) {
        case kGgufTypeUint8:
        case kGgufTypeInt8: {
            char dummy;
            return f.read(&dummy, 1).good();
        }
        case kGgufTypeUint16:
        case kGgufTypeInt16: {
            char dummy[2];
            return f.read(dummy, 2).good();
        }
        case kGgufTypeUint32:
        case kGgufTypeInt32:
        case kGgufTypeFloat32: {
            char dummy[4];
            return f.read(dummy, 4).good();
        }
        case kGgufTypeBool: {
            char dummy;
            return f.read(&dummy, 1).good();
        }
        case kGgufTypeString: {
            std::string dummy;
            return ReadString(f, dummy);
        }
        case kGgufTypeUint64:
        case kGgufTypeInt64:
        case kGgufTypeFloat64: {
            char dummy[8];
            return f.read(dummy, 8).good();
        }
        case kGgufTypeArray: {
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
                return false;
            }
            if (len > kMaxArrayElements) {
                return false;
            }
            for (uint64_t i = 0; i < len; ++i) {
                if (!SkipValue(f, elem_type)) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    } 
}

bool CheckAndGetMultiplyOut(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool CheckAndGetAddOut(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool GetTensorElementNum(const GgufTensorInfo& info, uint64_t& out) {
    out = 1;
    for (auto dim : info.shape) {
        if (dim == 0) {
            return false;
        }
        if (!CheckAndGetMultiplyOut(dim, out, out)) {
            return false;
        }
    }
    return true;
}

bool GetTensorByteSize(const GgufTensorInfo& info, uint64_t& out) {
    uint64_t element_num = 0;
    if (!GetTensorElementNum(info, element_num)) {
        return false;
    }

    switch (info.quant_type) {
        case kQuantTypeF32: {
            return CheckAndGetMultiplyOut(element_num, 4, out);
        }   
        case kQuantTypeF16: {
            return CheckAndGetMultiplyOut(element_num, 2, out);
        }    
        case kQuantTypeQ80: {
            if (element_num % kQuantBlockSize != 0) {
                return false;
            }   
            uint64_t block_size = element_num / kQuantBlockSize;
            return CheckAndGetMultiplyOut(block_size, 34, out);            
        }
        case kQuantTypeQ40: {
            if (element_num % kQuantBlockSize != 0) {
                return false; 
            }
            uint64_t block_size = element_num / kQuantBlockSize;
            return CheckAndGetMultiplyOut(block_size, 18, out);            
        }
        case kQuantTypeQ41: {
            if (element_num % kQuantBlockSize != 0) {
                return false;
            }
            uint64_t block_size = element_num / kQuantBlockSize;
            return CheckAndGetMultiplyOut(block_size, 20, out);            
        }
    }
    out = 0;
    return true;
}

bool ReadValueAsString(std::ifstream& f, uint32_t type, std::string& out) {
    std::ostringstream oss;
    switch (type) {
        case kGgufTypeUint8: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case kGgufTypeInt8: {
            int8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;            
        }
        case kGgufTypeUint16: {
            uint16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;            
        }
        case kGgufTypeInt16: {
            int16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;            
        }
        case kGgufTypeUint32: {
            uint32_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;            
        }
        case kGgufTypeInt32: {
            int32_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;            
        }
        case kGgufTypeFloat32: {
            float v = 0.0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;            
        }
        case kGgufTypeBool: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << (v ? "true" : "false");
            break;            
        }
        case kGgufTypeString: {
            if (!ReadString(f, out)) {
                return false;
            }
            return true;            
        }
        case kGgufTypeUint64: {
            uint64_t v = 0;
            if (!ReadU64(f, v)) {
                return false;
            }
            oss << v;
            break;              
        }
        case kGgufTypeInt64: {
            int64_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;            
        }
        case kGgufTypeFloat64: {
            double v = 0.0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;            
        }
        case kGgufTypeArray: {
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
                return false;
            }
            if (len > kMaxArrayElements) {
                return false;
            }
            oss << "[" << len << " x " << GetGgufValueTypeName(elem_type) << "]";
            for (uint64_t i = 0; i < len; ++i) {
                if (!SkipValue(f, elem_type)) {
                    return false;
                }
            }
            break;            
        }
    }
    out = oss.str();
    return true;
}           

uint64_t ParseAlignment(const std::vector<GgufMetaKv>& metadata) {
    uint64_t alignment = kDefaultAlignment;
    for (auto& data : metadata) {
        if (data.key != "general.alignment") {
            continue;
        }
        try {
            size_t use_size = 0;
            alignment = std::stoull(data.value_str, &use_size);
            if (use_size != data.value_str.size()) {
                return 0;
            }
        } catch (const std::exception&) {
            return 0;
        }
    }
    if (alignment == 0 || alignment > kMaxAlignment) {
        return 0;
    }
    if ((alignment & (alignment - 1)) != 0) {
        return 0;
    }
    return alignment;
}   

bool ReturnFailure(std::string& load_error, const std::string& error) {
    load_error = error;
    return false;
}

} // namespace

const char* GetQuantTypeName(uint32_t quant_type) {
    switch (quant_type) {
        case kQuantTypeF32:
            return "F32";
        case kQuantTypeF16:
            return "F16";
        case kQuantTypeQ40:
            return "Q4_0";
        case kQuantTypeQ41:
            return "Q4_1";
        case kQuantTypeQ80:
            return "Q8_0";
        default:
            return "UNKNOWN";   
    }
}

const char* GetGgufValueTypeName(uint32_t data_type) {
    switch (data_type) {
    case kGgufTypeUint8:
        return "uint8";
    case kGgufTypeInt8:
        return "int8";
    case kGgufTypeUint16:
        return "uint16";
    case kGgufTypeInt16:
        return "int16";
    case kGgufTypeUint32:
        return "uint32";
    case kGgufTypeInt32:
        return "int32";
    case kGgufTypeFloat32:
        return "float32";
    case kGgufTypeBool:
        return "bool";
    case kGgufTypeString:
        return "string";
    case kGgufTypeArray:
        return "array";
    case kGgufTypeUint64:
        return "uint64";
    case kGgufTypeInt64:
        return "int64";
    case kGgufTypeFloat64:
        return "float64";
    default:
        return "unknown";
    }
}

bool GgufReader::Load(const std::string& path) {
    version = 0;
    n_tensors = 0;
    n_metadatas = 0;
    data_offset = 0;
    load_error.clear();
    metadatas.clear();
    tensor_infos.clear();

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    //Magic  4byte
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        return ReturnFailure(load_error, "magic is not match");
    }

    //Version uint32
    if (!ReadU32(f, version)) {
        return ReturnFailure(load_error, "fail to read version");
    }
    if (version != 3) {
        return ReturnFailure(load_error, "not support version");
    }

    //n_tensors: uint64  (文件包含的张量总数)​     
    //n_metadatas: uint64  (Metadata 包含的键值对总数)
    if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadatas)) {
        return ReturnFailure(load_error, "n_tensors or n_metadatas is not load");
    }
    if (n_tensors > kMaxTensorEntries) {
        return ReturnFailure(load_error, "n_tensors is too many");
    }
    if (n_metadatas > kMaxMetadataEntries) {
        return ReturnFailure(load_error, "n_metadatas is too many");
    }

    //Matadatas
    /*
    key:    GGUF string (由 uint64 的长度声明 + 字节数据构成)
    ​type:   uint32      (声明 GGUF value 的数据类型)
    ​value:  取决于前面的 type 定义
    */
    metadatas.reserve(n_metadatas);
    std::unordered_set<std::string> key_set;
    for (uint64_t i = 0; i < n_metadatas; i++) {
        GgufMetaKv meta;
        if (!ReadString(f, meta.key)) {
            return ReturnFailure(load_error, "n_tensors is too many");
        }
        if (!key_set.insert(meta.key).second) {
            return ReturnFailure(load_error, "n_tensors is too many");
        }
        if (!ReadU32(f, meta.data_type)) {
            return ReturnFailure(load_error, "n_tensors is too many");
        }
        if (!ReadValueAsString(f, meta.data_type, meta.value_str)) {
            return ReturnFailure(load_error, "n_tensors is too many");
        }
        metadatas.push_back(std::move(meta));
    }

    //Tensorinfos
    /*
    name:   GGUF string
    n_dims: uint32
    dims:   uint64[n_dims]   (每个维度的大小)
    type:   uint32           (GGML 类型，决定存储格式与精度)
    offset: uint64           (相对于 Data Section 起始位置的偏移量)
    */
    tensor_infos.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GgufTensorInfo info;
        if (!ReadString(f, info.name)) {
            
        }
        uint32_t n_dims;
        if (!ReadU32(f, n_dims)) {

        }
        if (n_dims == 0 || n_dims > kMaxTensorDims) {

        }
        info.shape.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; j++) {
            if (!ReadU64(f, info.shape[j])) {

            }
            if (info.shape[j] == 0) {

            }
        }
        if (!ReadU32(f, info.quant_type)) {

        }
        if (!ReadU64(f, info.offset)) {

        }
        tensor_infos.push_back(std::move(info));
    }

    std::unordered_set<std::string> name_set;
    for (auto& info : tensor_infos) {
        if (!name_set.insert(info.name).second) {

        }
    }

    //Padding
    uint64_t alignment = ParseAlignment(metadatas);
    if (alignment == 0) {

    }
    std::streampos current = f.tellg();
    if (current < 0) {

    }
    data_offset = current;
    uint64_t padding = (alignment - (current % alignment)) % alignment;
    data_offset += padding;

    //检查tensor data 正确 
    f.seekg(0, std::ios::end);
    std::streampos end = f.tellg();
    if (end < 0 || static_cast<uint64_t>(end) > data_offset) {

    }
    uint64_t file_size = static_cast<uint64_t>(end);

    for (auto& info : tensor_infos) {
        uint64_t tensor_size = 0;
        if (!GetTensorByteSize(info, tensor_size)) {

        }
        if (tensor_size == 0) {
            continue;
        }

        uint64_t tensor_start = 0;
        if (!CheckAndGetAddOut(data_offset, info.offset, tensor_start)) {

        }
        
        uint64_t tensor_end = 0;
        if (!CheckAndGetAddOut(tensor_start, tensor_size, tensor_end)) {

        }

        if (tensor_end > file_size) {

        }
    }
    return true;
}





void InspectGguf(const GgufReader& reader) {
    std::cout << "=== GGUF Info ===" << std::endl;
    std::cout << "  version:    " << reader.version << std::endl;
    std::cout << "  n_tensors:  " << reader.n_tensors << std::endl;
    std::cout << "  n_metadata: " << reader.n_metadatas << std::endl;
    std::cout << "  data_offset: " << reader.data_offset << std::endl;

    if (!reader.metadatas.empty()) {
        std::cout << std::endl;
        std::cout << "=== Metadata ===" << std::endl;
        for (const auto& meta : reader.metadatas) {
            std::cout << "  " << meta.key << " (" << GetGgufValueTypeName(meta.data_type) << "): " << meta.value_str << std::endl;
        }
    }

    if (!reader.tensor_infos.empty()) {
        std::cout << std::endl;
        std::cout << "=== Tensors ===" << std::endl;
        for (const auto& info : reader.tensor_infos) {
            std::cout << "  " << info.name << std::endl;
            std::cout << "    shape:  [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << info.shape[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "    dtype:  " << GetQuantTypeName(info.quant_type) << " (" << info.quant_type << ")" << std::endl;
            std::cout << "    offset: " << info.offset << std::endl;
        }
    }
}

bool GetGgufTensorByteSize(const GgufTensorInfo& info, uint64_t& out) {
    return GetTensorByteSize(info, out);
}

std::vector<uint8_t> GetGgufSingleTensor(const std::string& path, const GgufTensorInfo& info, uint64_t data_offset) {
    std::vector<uint8_t> result;

    uint64_t tensor_size = 0;
    if (!GetTensorByteSize(info, tensor_size)) {
        return result;
    }
    if (tensor_size == 0) {
        return result;
    }
    
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return result;
    }

    uint64_t tensor_start = 0;
    if (!CheckAndGetAddOut(data_offset, info.offset, tensor_start)) {
        return result;
    }
    if (tensor_start > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return result;
    }

    f.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
    if (!f.good()) {
        return result;
    }

    result.resize(tensor_size);
    f.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(tensor_size));
    if (static_cast<uint64_t>(f.gcount()) != tensor_size) {
        result.clear();
        return result;
    }
    return result;
}

// 函数执行后，找到的meta的key,type之后，f指向了value的位置
bool OpenAndFindMetaDataKey(std::ifstream& f, const std::string& path, const std::string& key, uint32_t& out_type) {
    f.open(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    //跳过 magic， version
    f.seekg(8, std::ios::beg);

    uint64_t n_tensors;
    uint64_t n_metadatas;
    if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadatas)) {
        return false;
    }

    for (uint64_t i = 0; i < n_metadatas; i++) {
        std::string meta_key;
        if (!ReadString(f, meta_key)) {
            return false;
        }
        uint32_t type;
        if (!ReadU32(f, type)) {
            return false;
        }
        if (meta_key == key) {
            out_type = type;
            return true;
        }

        if (!SkipValue(f, type)) {
            return false;
        }
    }
    return false;
}

bool GetGgufMetaDataInt(const std::string& path, const std::string& key, int64_t& out) {
    std::ifstream f;
    uint32_t type;
    if (!OpenAndFindMetaDataKey(f, path, key, type)) {
        return false;
    }

    switch (type) {
        case kGgufTypeUint8: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeInt8: {
            int8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeUint16: {
            uint16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeInt16: {
            int16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeUint32: {
            uint32_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeInt32: {
            int32_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeUint64: {
            uint64_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
        case kGgufTypeInt64: {
            int64_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<int64_t>(v);
            return true;
        }
    }
    return false;
}

bool GetGgufMetaDataFloat(const std::string& path, const std::string& key, double& out) {
    std::ifstream f;
    uint32_t type;
    if (!OpenAndFindMetaDataKey(f, path, key, type)) {
        return false;
    }
    switch (type) {
        case kGgufTypeFloat32: {
            float v = 0.0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = static_cast<double>(v);
            return true;
        }
        case kGgufTypeFloat64: {
            double v = 0.0;
            if (!ReadLe(f, v)) {
                return false;
            }
            out = v;
            return true;
        }
    }
    return false;
}

bool GetGgufMetaDataString(const std::string& path, const std::string& key, std::string& out) {
    std::ifstream f;
    uint32_t type;
    if (!OpenAndFindMetaDataKey(f, path, key, type)) {
        return false;
    }
    if (type != kGgufTypeString) {
        return false;
    }
    return ReadString(f, out);
}

bool GetGgufMetaDataStringArray(const std::string& path, const std::string& key, std::vector<std::string>& out) {
    std::ifstream f;
    uint32_t type;
    if (!OpenAndFindMetaDataKey(f, path, key, type)) {
        return false;
    }
    if (type != kGgufTypeArray) {
        return false;
    }

    uint32_t element_type = 0;
    uint64_t element_size = 0;
    if (!ReadU32(f, element_type) || !ReadU64(f, element_size)) {
        return false;
    }
    if (element_type != kGgufTypeString) {
        return false;
    }
    if (element_size > kMaxTensorEntries) {
        return false;
    }
    out.clear();
    out.reserve(element_size);
    for (uint64_t i = 0; i < element_size; i++) {
        std::string s;
        if (!ReadString(f, s)) {
            return false;
        }
        out.push_back(std::move(s));
    }
    return true;
}

bool GetGgufMetaDataIntArray(const std::string& path, const std::string& key, std::vector<int32_t>& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!OpenAndFindMetaDataKey(f, path, key, type)) {
        return false;
    }
    if (type != kGgufTypeArray) {
        return false;
    }

    uint32_t element_type = 0;
    uint64_t element_size = 0;
    if (!ReadU32(f, element_type) || !ReadU64(f, element_size)) {
        return false;
    }
    if (element_type != kGgufTypeInt32) {
        return false;
    }
    if (element_size > kMaxArrayElements) {
        return false;
    }

    out.clear();
    out.reserve(element_size);

    for (uint64_t i = 0; i < element_size; i++) {
        int32_t v = 0;
        if (!ReadLe(f, v)) {
            return false;
        }
        out.push_back(v);
    }
    return true;
}



}