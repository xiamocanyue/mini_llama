#include "include/tensor.h"

#include <limits>

namespace mini_llama {
namespace {
    
const char* CallerName(const char* caller) {
    return caller == nullptr ? "Tensor" : caller;
}

//{3,4,5}  -> [3,4,5]
std::string ShapeToString(const std::vector<int>& shape) {
    std::string s = "[";
    for (int i = 0; i < shape.size(); i++) {
        if (i > 0) {
            s += ',';
        }
        s += std::to_string(shape[i]);
    }
    s += ']';
    return s;

}
//总数是否会超过类型最大值，并返回总数
size_t CheckTotalNum(const std::vector<int>& shape, const char* caller) {
    size_t total = 1;
    for (int i = 0; i < shape.size(); i++) {
        int dim = shape[i];
        if (dim <= 0) {
            throw std::runtime_error(std::string(CallerName(caller)) + 
                                                ": dimension at axis " + std::to_string(i) +
                                                "must be positive, got " + std::to_string(dim) + 
                                                "in shape " + ShapeToString(shape));                                       
        }
        size_t dim_size = static_cast<size_t>(dim);
        if (total > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::runtime_error(std::string(caller) + "total num overflow in " + ShapeToString(shape));
        }
        total *= dim_size;
    }
    return total;
}

//维度是否越界
void CheckRank(const Tensor& t, int expect, const char* caller) {
    if (t.DimNum() != expect) {
        throw std::runtime_error(std::string(caller) + "expected " +
                                std::to_string(expect) + "D tensor, Got" + 
                                t.ShapeString());
    }
}

// 坐标是否越界, axis 是shape的行号， flat_index 是这一行的第几个
void CheckAxisIndex(const Tensor& t, int axis, int flat_index, const char* caller) {
    int dim = t.shape[axis];
    if (flat_index < 0 || flat_index >= dim) {
        throw std::runtime_error(std::string(caller) + " FlatIndex " + std::to_string(flat_index) +
        "out of range for axis" + std::to_string(axis) + " with size " + std::to_string(dim) + 
        "in tensor" + t.ShapeString());
    }
}
}

Tensor::Tensor(const std::vector<int>& input_shape, float filled) 
            : shape(std::move(input_shape)) {
    size_t total_num = CheckTotalNum(shape, "Tensor Constructor");
    data.resize(total_num, filled);
}

size_t Tensor::GetFlatIndex(const std::vector<int>& index) const {
    if (index.size() != shape.size()) {
        throw std::runtime_error("Tnesor::GetFlatIndex error " + std::to_string(shape.size()) + "indices for tensor "
                                + ShapeString() + ", got " + std::to_string(index.size()));
    }
    for (size_t axis = 0; axis < index.size(); axis++) {
        CheckAxisIndex(*this, static_cast<int>(axis), index[axis], "Tensor::GetFlatIndex");
    }
    size_t stride = 1;
    size_t flat = 0;
    for (int i = static_cast<int>(shape.size() - 1); i >= 0; i--) {
        flat += static_cast<size_t>(index[i]) * stride;
        stride *= static_cast<size_t>(shape[i]);
    }

    return flat;
}

float* Tensor::GetRowPtr(int row) {
    CheckRank(*this, 2, "row_ptr");
    CheckAxisIndex(*this, 0, row, "row_ptr");
    return data.data() + row * shape[1];
}

void Tensor::CheckShape(const std::vector<int>& expect, const char* caller) const {
    if (shape != expect) {
        throw std::runtime_error(std::string(CallerName(caller)) + ": shape can not match expected " + ShapeToString(expect) + ", got" + ShapeString());
    }
}

Tensor Tensor::ReshapeAndCheck(const std::vector<int>& new_shape, const char* caller) const {
    size_t new_total = CheckTotalNum(new_shape, CallerName(caller));
    if (new_total != data.size()) {
        throw std::runtime_error(std::string(CallerName(caller)) + " can not reshape tensor with" + 
                                std::to_string(data.size()) + " element into shape with " + std::to_string(new_total) + " elements");
    }
    Tensor new_tensor = *this;
    new_tensor.shape = new_shape;
    return new_tensor; 

}

float& Tensor::At(const std::vector<int>& index) {
    return data[GetFlatIndex(index)];
}
float Tensor::At(const std::vector<int>& index) const {
    return data[GetFlatIndex(index)];
}

float& Tensor::At1(int i) {
    CheckRank(*this, 1, "AT1");
    CheckAxisIndex(*this, 0, i, "AT1");
    return data[i];
}
float Tensor::At1(int i) const {
    CheckRank(*this, 1, "AT1");
    CheckAxisIndex(*this, 0, i, "AT1");
    return data[i];
}

float& Tensor::At2(int i, int j) {
    CheckRank(*this, 2, "AT2");
    CheckAxisIndex(*this, 0, i, "AT2");
    CheckAxisIndex(*this, 1, j, "AT2");
    return data[i * shape[1] + j];
}
float Tensor::At2(int i, int j) const {
    CheckRank(*this, 2, "AT2");
    CheckAxisIndex(*this, 0, i, "AT2");
    CheckAxisIndex(*this, 1, j, "AT2");
    return data[i * shape[1] + j];
}

float& Tensor::At3(int i, int j, int k) {
    CheckRank(*this, 3, "AT3");
    CheckAxisIndex(*this, 0, i, "AT3");
    CheckAxisIndex(*this, 1, j, "AT3");
    CheckAxisIndex(*this, 2, k, "AT3");
    return data[(i * shape[1] + j) * shape[2] + k];
}
float Tensor::At3(int i, int j, int k) const {
    CheckRank(*this, 3, "AT3");
    CheckAxisIndex(*this, 0, i, "AT3");
    CheckAxisIndex(*this, 1, j, "AT3");
    CheckAxisIndex(*this, 2, k, "AT3");
    return data[(i * shape[1] + j) * shape[2] + k];
}

float& Tensor::At4(int i, int j, int k, int l) {
    CheckRank(*this, 4, "AT4");
    CheckAxisIndex(*this, 0, i, "AT4");
    CheckAxisIndex(*this, 1, j, "AT4");
    CheckAxisIndex(*this, 2, k, "AT4");
    CheckAxisIndex(*this, 3, l, "AT4");
    return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}
float Tensor::At4(int i, int j, int k, int l) const {
    CheckRank(*this, 4, "AT4");
    CheckAxisIndex(*this, 0, i, "AT4");
    CheckAxisIndex(*this, 1, j, "AT4");
    CheckAxisIndex(*this, 2, k, "AT4");
    CheckAxisIndex(*this, 3, l, "AT4");
    return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

 
std::string Tensor::ShapeString() const {
    return ShapeToString(shape);
}

} // namespace mini_llama
