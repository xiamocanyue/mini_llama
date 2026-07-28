#pragma once 

#include <vector>
#include <string>
#include <stdexcept>

namespace mini_llama {

struct Tensor {
    Tensor() = default;
    Tensor(const std::vector<int>& input_shape, float filled = 0.0f);

    size_t GetFlatIndex(const std::vector<int>& index) const;
    float* GetRowPtr(int row);

    float& At(const std::vector<int>& index);
    float At(const std::vector<int>& index) const;

    float& At1(int i);
    float At1(int i) const;
    float& At2(int i, int j);
    float At2(int i, int j) const;
    float& At3(int i, int j, int k);
    float At3(int i, int j, int k) const;
    float& At4(int i, int j, int k, int l);
    float At4(int i, int j, int k, int l) const;

    void CheckShape(const std::vector<int>& expect, const char* caller) const;
    Tensor ReshapeAndCheck(const std::vector<int>& new_shape, const char* caller) const;
    std::string ShapeString() const;

    int DimNum() const {
        return static_cast<int>(shape.size());
    }

    size_t Size() const {
        return data.size();
    }


    std::vector<float> data;
    std::vector<int> shape; 

};

inline Tensor MakeTensor1D(int d0, float filled) {
    return Tensor({d0}, filled);
}
inline Tensor MakeTensor2D(int d0, int d1, float filled) {
    return Tensor({d0, d1}, filled);
}
inline Tensor MakeTensor3D(int d0, int d1, int d2, float filled) {
    return Tensor({d0, d1, d2}, filled);
}
inline Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float filled) {
    return Tensor({d0, d1, d2, d3}, filled);
}

} //namespace mini_llama
