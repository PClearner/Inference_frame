#pragma once
#include "layer/abstract/non_param_layer.hpp"
namespace star
{
    class ReluLayer : public NonParamLayer
    {
    public:
        ReluLayer() : NonParamLayer("Relu") {}
        InferStatus Forward(
            const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

        static ParseParameterAttrStatus GetInstance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &relu_layer);
    };
} //