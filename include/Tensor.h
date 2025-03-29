#pragma once
#include <iostream>
#include <armadillo>
#include <vector>
#include <glog/logging.h>
#include <memory>

namespace star
{
    template <typename T>
    class Tensor;

    template <>
    class Tensor<float>
    {
    public:
        explicit Tensor() = default;
        Tensor(uint32_t cols);
        Tensor(uint32_t rows, uint32_t cols);
        Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
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
        bool empty() const;
        float *raw_ptr();
        float *raw_ptr(uint32_t offset);
        float at(uint32_t channel, uint32_t row, uint32_t col) const;
        float &at(uint32_t channel, uint32_t row, uint32_t col);
        float index(uint32_t offset) const;
        float &index(uint32_t offset);
        void set_data(const arma::fcube &data);
        void Show();

        void Fill(std::vector<float> values, bool row_major = true);
        void Fill(float value);
        void Reshape(uint32_t channels, uint32_t rows,
                     uint32_t cols, bool row_major = true);

        void Reshape(const std::vector<uint32_t> &shapes,
                     bool row_major);

        arma::fcube &data();

        /**
         * 返回张量中的数据
         * @return 张量中的数据
         */
        const arma::fcube &data() const;

        /**
         * 返回张量第channel通道中的数据
         * @param channel 需要返回的通道
         * @return 返回的通道
         */
        arma::fmat &slice(uint32_t channel);

        /**
         * 返回张量第channel通道中的数据
         * @param channel 需要返回的通道
         * @return 返回的通道
         */
        const arma::fmat &slice(uint32_t channel) const;

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

    using ftensor = Tensor<float>;
    using sftensor = std::shared_ptr<Tensor<float>>;

}
