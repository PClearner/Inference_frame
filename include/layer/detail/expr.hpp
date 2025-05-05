#pragma once
#include "layer/abstract/non_param_layer.hpp"
#include "layer/parse/parse_expression.hpp"

namespace star
{
    class ExpressionLayer : public NonParamLayer
    {
    public:
        explicit ExpressionLayer(std::string statement);

        InferStatus Forward(
            const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

        static ParseParameterAttrStatus GetInstance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &expression_layer);

    private:
        std::string statement_;
        std::unique_ptr<ExpressionParser> parser_;
        std::vector<int32_t> order;
    };
} //