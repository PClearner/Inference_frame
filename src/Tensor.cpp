#include "Tensor.h"
#include <glog/logging.h>
#include <numeric>
#include <memory>
#include <functional>

namespace star
{
    Tensor<float>::Tensor(uint32_t cols)
    {
        uint32_t channels = 1;
        uint32_t rows = 1;
        data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    Tensor<float>::Tensor(uint32_t rows, uint32_t cols)
    {
        uint32_t channels = 1;
        data_ = arma::fcube(rows, cols, channels);

        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
    {
        data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    Tensor<float>::Tensor(std::vector<uint32_t> &&shapes)
    {
        CHECK(!shapes.empty() && shapes.size() <= 3);

        uint32_t rows, cols, channels;
        if (shapes.size() == 1)
        {
            cols = shapes[0];
            channels = 1;
            rows = 1;
        }
        else if (shapes.size() == 2)
        {
            rows = shapes[0];
            cols = shapes[1];
            channels = 1;
        }
        else if (shapes.size() == 3)
        {
            channels = shapes[0];
            rows = shapes[1];
            cols = shapes[2];
        }

        this->data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    Tensor<float>::Tensor(Tensor<float> &&tensor)
    {
        if (this != &tensor)
        {
            this->data_ = tensor.data_;
            this->raw_shapes_ = tensor.raw_shapes_;
        }
    }

    uint32_t Tensor<float>::rows() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_rows;
    }

    uint32_t Tensor<float>::cols() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_cols;
    }

    uint32_t Tensor<float>::channels() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_slices;
    }

    float *Tensor<float>::raw_ptr()
    {
        CHECK(!this->data_.empty());
        return this->data_.memptr();
    }

    float *Tensor<float>::raw_ptr(uint32_t offset)
    {
        const uint32_t size = this->size();
        CHECK(!this->data_.empty());
        CHECK_LT(offset, size);
        return this->data_.memptr() + offset;
    }

    uint32_t Tensor<float>::size() const
    {
        CHECK(!this->data_.empty());
        return this->data_.size();
    }

    void Tensor<float>::set_data(const arma::fcube &data)
    {
        CHECK(data.n_rows == this->data_.n_rows)
            << data.n_rows << " != " << this->data_.n_rows;
        CHECK(data.n_cols == this->data_.n_cols)
            << data.n_cols << " != " << this->data_.n_cols;
        CHECK(data.n_slices == this->data_.n_slices)
            << data.n_slices << " != " << this->data_.n_slices;
        this->data_ = data;
    }

    arma::fmat &Tensor<float>::slice(uint32_t channel)
    {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    bool Tensor<float>::empty() const
    {
        return this->data().empty();
    }

    arma::fcube &Tensor<float>::data() { return this->data_; }

    const arma::fcube &Tensor<float>::data() const { return this->data_; }

    const arma::fmat &Tensor<float>::slice(uint32_t channel) const
    {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    std::vector<uint32_t> Tensor<float>::shapes() const
    {
        CHECK(!this->data_.empty());
        return {this->channels(), this->rows(), this->cols()};
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const
    {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col)
    {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    float Tensor<float>::index(uint32_t offset) const
    {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }

    float &Tensor<float>::index(uint32_t offset)
    {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }

    void Tensor<float>::Fill(float value)
    {
        CHECK(!this->data_.empty());
        this->data_.fill(value);
    }

    void Tensor<float>::Fill(std::vector<float> values, bool row_major)
    {
        CHECK(!this->data_.empty());
        const uint32_t total_elems = this->data_.size();
        CHECK_EQ(values.size(), total_elems);

        if (row_major)
        {
            uint32_t row = this->rows();
            uint32_t col = this->cols();
            uint32_t channelsize = row * col;
            uint32_t channels = this->channels();

            for (uint32_t i = 0; i < channels; i++)
            {
                auto &channel = this->data_.slice(i);
                const arma::fmat &data = arma::fmat(values.data() + i * channelsize, col, row);
                channel = data.t();
            }
        }
        else
        {
            std::copy(values.begin(), values.end(), this->data_.memptr());
        }
    }

    std::vector<float> Tensor<float>::values(bool row_major)
    {
        CHECK_EQ(this->data_.empty(), false);
        std::vector<float> values(this->data_.size());

        if (!row_major)
        {
            std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
                      values.begin());
        }
        else
        {
            uint32_t index = 0;
            for (uint32_t c = 0; c < this->data_.n_slices; ++c)
            {
                const arma::fmat &channel = this->data_.slice(c).t();
                std::copy(channel.begin(), channel.end(), values.begin() + index);
                index += channel.size();
            }
            CHECK_EQ(index, values.size());
        }
        return values;
    }

    void Tensor<float>::Reshape(uint32_t channels, uint32_t rows, uint32_t cols, bool row_major)
    {
        // check
        CHECK(!this->data_.empty());
        const uint32_t origin_size = this->size();
        const uint32_t current_size = channels * rows * cols;
        CHECK(current_size == origin_size);
        //
        std::vector<float> values(current_size);
        values = this->values(row_major);
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
        this->data_ = arma::fcube(rows, cols, channels);
        Fill(values, row_major);
    }

    void Tensor<float>::Show()
    {
        for (uint32_t i = 0; i < this->channels(); ++i)
        {
            LOG(INFO) << "Channel: " << i;
            LOG(INFO) << "\n"
                      << this->data_.slice(i);
        }
    }

    void Tensor<float>::Reshape(const std::vector<uint32_t> &shapes,
                                bool row_major)
    {
        CHECK(!this->data_.empty());
        CHECK(!shapes.empty());
        const uint32_t origin_size = this->size();
        const uint32_t current_size =
            std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<uint32_t>());
        CHECK(shapes.size() <= 3);
        CHECK(current_size == origin_size);

        std::vector<float> values;
        if (row_major)
        {
            values = this->values(true);
        }
        if (shapes.size() == 3)
        {
            this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
            this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
        }
        else if (shapes.size() == 2)
        {
            this->data_.reshape(shapes.at(0), shapes.at(1), 1);
            this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
        }
        else
        {
            this->data_.reshape(1, shapes.at(0), 1);
            this->raw_shapes_ = {shapes.at(0)};
        }

        if (row_major)
        {
            this->Fill(values, true);
        }
    }

    void Tensor<float>::Flatten(bool row_major)
    {
        // check
        CHECK(!this->data_.empty());
        //

        uint32_t size = this->size();
        std::vector<float> values(this->size());
        uint32_t rows = this->rows();
        uint32_t cols = this->cols();
        uint32_t channel_size = rows * cols;
        uint32_t channels = this->channels();

        for (uint32_t i = 0; i < channels; i++)
        {
            const arma::fmat &channel_data = this->data_.slice(i).t();
            std::copy(channel_data.begin(), channel_data.end(), values.data() + i * channel_size);
        }

        data_ = arma::fcube(1, size, 1);
        Fill(values);

        this->raw_shapes_ = std::vector<uint32_t>{size};
    }

    Tensor<float> &Tensor<float>::operator=(Tensor<float> &&tensor) noexcept
    {
        if (this != &tensor)
        {
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = tensor.raw_shapes_;
        }
        return *this;
    }

    Tensor<float> &Tensor<float>::operator=(const Tensor &tensor)
    {
        if (this != &tensor)
        {
            this->data_ = tensor.data_;
            this->raw_shapes_ = tensor.raw_shapes_;
        }
        return *this;
    }

    void Tensor<float>::Padding(const std::vector<uint32_t> &pads, float padding_value)
    {
        // check
        CHECK(!this->data_.empty());
        CHECK_EQ(pads.size(), 4);
        //

        uint32_t rows = this->rows() + pads[0] + pads[1];
        uint32_t cols = this->cols() + pads[2] + pads[3];
        uint32_t channel_size = rows * cols;
        uint32_t channels = this->channels();

        arma::fcube tmp = arma::fcube(rows, cols, channels);
        tmp.fill(padding_value);

        for (uint32_t i = 0; i < channels; i++)
        {
            arma::fmat &tmp_data = tmp.slice(i);
            const arma::fmat &channel_data = this->data_.slice(i);
            for (uint32_t r = 0; r < this->cols(); r++)
            {
                uint32_t index = rows * (r + pads[2]) + pads[0];
                std::copy(channel_data.begin() + r * this->rows(), channel_data.begin() + (r + 1) * this->rows(), tmp_data.begin() + index);
            }
        }

        this->data_ = tmp;
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    void Tensor<float>::Ones()
    {
        // check
        CHECK(!this->data_.empty());
        //
        this->data_.fill(1.0f);
    }

    void Tensor<float>::Rand()
    {
        // check
        CHECK(!this->data_.empty());
        //
        this->data_ = arma::randn<arma::fcube>(this->rows(), this->cols(), this->channels());
    }

    void Tensor<float>::Transform(const std::function<float(float)> &filter)
    {
        // check
        CHECK(!this->data_.empty());
        //
        this->data_.transform(filter);
    }

}
