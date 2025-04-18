
#include "layer/abstract/param_layer.hpp"
#include "tensor_util.hpp"

namespace star
{

    ParamLayer::ParamLayer(const std::string &layer_name) : Layer(layer_name) {}

    /**
     * 初始化权重空间
     * @param param_count   卷积核数量
     * @param param_channel 卷积的通道数量
     * @param param_height  卷积的高度
     * @param param_width   卷积的宽度
     */
    void ParamLayer::InitWeightParam(const uint32_t param_count, const uint32_t param_channel,
                                     const uint32_t param_height, const uint32_t param_width)
    {
        this->weights_ = std::vector<sftensor>(param_count);
        for (uint32_t i = 0; i < param_count; i++)
        {
            this->weights_[i] = TensorCreate({param_channel, param_height, param_width});
        }
    }

    /**
     * 初始化偏移参数
     * @param param_count 偏移参数数量
     * @param param_channel 偏移参数通道数量
     * @param param_height 偏移参数高度
     * @param param_width  偏移参数宽度
     */
    void ParamLayer::InitBiasParam(const uint32_t param_count, const uint32_t param_channel,
                                   const uint32_t param_height, const uint32_t param_width)
    {
        this->bias_ = std::vector<sftensor>(param_count);
        for (uint32_t i = 0; i < param_count; i++)
        {
            this->bias_[i] = TensorCreate({param_channel, param_height, param_width});
        }
    }

    /**
     * 返回权重参数
     * @return 权重参数
     */
    const std::vector<std::shared_ptr<Tensor<float>>> &ParamLayer::weights() const
    {
        return this->weights_;
    }

    /**
     * 返回偏移参数
     * @return 偏移参数
     */
    const std::vector<std::shared_ptr<Tensor<float>>> &ParamLayer::bias() const
    {
        return this->bias_;
    }

    /**
     * 设置权重参数
     * @param weights 权重参数
     * 默认每个weight的size相同
     */
    void ParamLayer::set_weights(const std::vector<float> &weights)
    {
        uint32_t weights_size = weights.size();
        uint32_t batchs = this->weights_.size();
        uint32_t w_s = this->weights_[0]->size();
        uint32_t m_size = w_s * batchs;
        CHECK_EQ(m_size, weights_size);
        uint32_t w_start;
        uint32_t w_end;
        for (uint32_t i = 0; i < batchs; i++)
        {
            w_start = i * w_s;
            w_end = (i + 1) * w_s;
            const auto &tmp = std::vector<float>(weights.begin() + w_start, weights.begin() + w_end);
            this->weights_[i]->Fill(tmp);
        }
    }

    /**
     * 设置偏移量参数
     * @param bias 偏移量参数
     */
    void ParamLayer::set_bias(const std::vector<float> &bias)
    {
        uint32_t batchs = this->bias_.size();
        uint32_t bias_size = bias.size();
        uint32_t b_s = this->bias_.size();
        uint32_t m_size = b_s * batchs;
        CHECK_EQ(m_size, bias_size);
        uint32_t b_start;
        uint32_t b_end;
        for (uint32_t i = 0; i < batchs; i++)
        {
            b_start = i * b_s;
            b_end = (i + 1) * b_s;
            const auto &tmp = std::vector<float>(bias.begin() + b_start, bias.begin() + b_end);
            this->bias_[i]->Fill(tmp);
        }
    }

    /**
     * 设置权重参数
     * @param weights 权重参数
     */
    void ParamLayer::set_weights(
        const std::vector<std::shared_ptr<Tensor<float>>> &weights)
    {
        CHECK(weights.size() == weights_.size());
        for (uint32_t i = 0; i < weights.size(); ++i)
        {
            CHECK(this->weights_.at(i) != nullptr);
            CHECK(this->weights_.at(i)->rows() == weights.at(i)->rows());
            CHECK(this->weights_.at(i)->cols() == weights.at(i)->cols());
            CHECK(this->weights_.at(i)->channels() == weights.at(i)->channels());
        }
        this->weights_ = weights;
    }

    /**
     * 设置偏移量参数
     * @param bias 偏移量参数
     */
    void ParamLayer::set_bias(
        const std::vector<std::shared_ptr<Tensor<float>>> &bias)
    {
        if (!this->bias_.empty())
        {
            CHECK(bias.size() == bias_.size());
            for (uint32_t i = 0; i < bias.size(); ++i)
            {
                CHECK(this->bias_.at(i) != nullptr);
                CHECK(this->bias_.at(i)->rows() == bias.at(i)->rows());
                CHECK(this->bias_.at(i)->cols() == bias.at(i)->cols());
                CHECK(this->bias_.at(i)->channels() == bias.at(i)->channels());
            }
            this->bias_ = bias;
        }
    }

}