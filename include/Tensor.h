#pragma once
#include <iostream>
#include <armadillo>
#include <vector>
#include <glog/logging.h>

namespace star
{
    template <typename T>
    class Tensor;

    template <>
    class Tensor<float>
    {
    public:
        explicit Tensor() = default;
        Tensor(uint32_t cols, uint32_t rows = 1, uint32_t channels = 1);
        Tensor(Tensor<float> &&tensor);
        // shapes:{channel,rows,cols}
        explicit Tensor(std::vector<uint32_t> &&shapes);
        Tensor<float> &operator=(Tensor &&tensor) noexcept;
        Tensor<float> &operator=(const Tensor &tensor);

        uint32_t rows() const;
        uint32_t cols() const;
        uint32_t channels() const;
        uint32_t size() const;
        std::vector<uint32_t> shapes() const;
        float *raw_ptr();
        float *raw_ptr(uint32_t offset);
        float at(uint32_t channel, uint32_t row, uint32_t col) const;
        float &at(uint32_t channel, uint32_t row, uint32_t col);
        void Show();

        void Fill(std::vector<float> values, bool row_major = true);
        void Fill(float value);
        void Reshape(uint32_t channels, uint32_t rows,
                     uint32_t cols, bool row_major = true);

        void Reshape(const std::vector<uint32_t> &shapes,
                     bool row_major);

        // 先弄成一列，再转置
        void Flatten(bool row_major = true);

        // pads{up,bottom,left,right}
        void Padding(const std::vector<uint32_t> &pads, float padding_value);

        void Ones();

        void Transform(const std::function<float(float)> &filter);

        void Rand();

        std::vector<float> values(bool row_major);

    private:
        std::vector<uint32_t> raw_shapes_;
        arma::fcube data_;
    };

}
