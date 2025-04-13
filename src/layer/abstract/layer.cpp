#include "layer/abstract/layer.hpp"
#include "status_code.hpp"

namespace star
{
    InferStatus Layer::Forward(
        const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
        std::vector<std::shared_ptr<Tensor<float>>> &outputs)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    /**
     * Layer的执行函数
     * @param current_operator 当前的operator
     * @return 执行的状态
     */
    InferStatus Layer::Forward()
    {
        const auto &runtime_operator = this->runtime_operator_.lock();
        const auto &input_operands = runtime_operator->input_operands_seq;
        std::vector<sftensor> input_datas;
        for (const auto &input_operands : input_operands)
        {
            for (const auto &input_data : input_operands->datas)
            {
                input_datas.push_back(input_data);
            }
        }

        CHECK(runtime_operator->output_operands != nullptr) << "Layer output data is empty";
        auto &output_data = runtime_operator->output_operands->datas;

        CHECK(!input_datas.empty())
            << runtime_operator->name << " Layer input data is empty";
        CHECK(!output_data.empty()) << "Layer output data is empty";

        InferStatus result = runtime_operator->layer->Forward(input_datas, output_data);
        return result;
    }

    /**
     * 返回层的权重
     * @return 返回的权重
     */
    const std::vector<std::shared_ptr<Tensor<float>>> &Layer::weights() const
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    /**
     * 返回层的偏移量
     * @return 返回的偏移量
     */
    const std::vector<std::shared_ptr<Tensor<float>>> &Layer::bias() const
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    /**
     * 设置Layer的权重
     * @param weights 权重
     */
    void Layer::set_weights(
        const std::vector<std::shared_ptr<Tensor<float>>> &weights)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    /**
     * 设置Layer的偏移量
     * @param bias 偏移量
     */
    void Layer::set_bias(
        const std::vector<std::shared_ptr<Tensor<float>>> &bias)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    /**
     * 设置Layer的权重
     * @param weights 权重
     */
    void Layer::set_weights(const std::vector<float> &weights)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    /**
     * 设置Layer的偏移量
     * @param bias 偏移量
     */
    void Layer::set_bias(const std::vector<float> &bias)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    /**
     * 设置层的执行算子
     * @param runtime_operator 该层的执行算子
     */
    void Layer::set_runtime_operator(
        const std::shared_ptr<RuntimeOperator> &runtime_operator)
    {
        CHECK(runtime_operator != nullptr);
        this->runtime_operator_ = runtime_operator;
    }

}
