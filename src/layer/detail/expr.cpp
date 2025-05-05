#include "layer/detail/expr.hpp"
#include <stack>
#include "tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace star
{
    ExpressionLayer::ExpressionLayer(std::string statement)
        : NonParamLayer("Expression"), statement_(std::move(statement))
    {
        parser_ = std::make_unique<ExpressionParser>(statement_);
    }

    InferStatus ExpressionLayer::Forward(
        const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
        std::vector<std::shared_ptr<Tensor<float>>> &outputs)
    {
        if (inputs.empty())
        {
            LOG(ERROR) << "The input tensor array in the expression layer is empty";
            return InferStatus::InferFailedInputEmpty;
        }

        if (outputs.empty())
        {
            LOG(ERROR) << "The output tensor array in the expression layer is empty";
            return InferStatus::InferFailedOutputEmpty;
        }

        CHECK(this->parser_ != nullptr)
            << "The parser in the expression layer is null!";
        this->parser_->Tokenizer(false);
        const auto &expressions = this->parser_->tokens();
        CHECK(!expressions.empty())
            << "The expression parser failed to parse " << statement_;

        for (uint32_t i = 0; i < inputs.size(); ++i)
        {
            const sftensor &input_data = inputs.at(i);
            if (input_data == nullptr || input_data->empty())
            {
                LOG(ERROR) << "The input tensor array in the expression layer has an "
                              "empty tensor "
                           << i << "th";
                return InferStatus::InferFailedInputEmpty;
            }
        }

        const uint32_t batch_size = outputs.size();
        for (uint32_t i = 0; i < batch_size; ++i)
        {
            if (outputs.at(i) == nullptr || outputs.at(i)->empty())
            {
                DLOG(ERROR) << "The output tensor array in the expression layer has an "
                               "empty tensor "
                            << i << "th";
                return InferStatus::InferFailedOutputEmpty;
            }
            outputs.at(i)->Fill(0.f);
        }

        std::unordered_map<int32_t, std::vector<std::shared_ptr<Tensor<float>>>> hash;
        std::stack<std::vector<std::shared_ptr<Tensor<float>>>> op_stack;
        const std::vector<std::shared_ptr<TokenNode>> &token_nodes = this->parser_->Generate();
        for (const auto &token_node : token_nodes)
        {
            if (token_node->num_index >= 0)
            {
                std::vector<std::shared_ptr<Tensor<float>>> input_operand;
                if (hash.find(token_node->num_index) == hash.end())
                {
                    uint32_t start_pos;
                    for (uint32_t p = 0; p < this->order.size(); p++)
                    {
                        if (token_node->num_index == this->order[p])
                        {
                            start_pos = p * batch_size;
                        }
                    }

                    for (uint32_t i = 0; i < batch_size; i++)
                    {
                        input_operand.push_back(inputs.at(i + start_pos));
                    }
                }
                else
                {
                    input_operand = hash.at(token_node->num_index);
                }
                op_stack.push(input_operand);
            }
            else if (token_node->num_index == int(TokenType::TokenMul) ||
                     token_node->num_index == int(TokenType::TokenAdd))
            {
                CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
                std::vector<std::shared_ptr<Tensor<float>>> operand1 = op_stack.top();
                CHECK(operand1.size() == batch_size)
                    << "The first operand doesn't have appropriate number of tensors, "
                       "which need "
                    << batch_size;
                op_stack.pop();
                std::vector<std::shared_ptr<Tensor<float>>> operand2 = op_stack.top();
                CHECK(operand2.size() == batch_size)
                    << "The first operand doesn't have appropriate number of tensors, "
                       "which need "
                    << batch_size;
                op_stack.pop();
                std::vector<std::shared_ptr<Tensor<float>>> output_operands(batch_size);
                if (token_node->num_index == int(TokenType::TokenMul))
                {
                    for (uint32_t j = 0; j < operand1.size(); j++)
                    {
                        output_operands.at(j) = TensorElementAdd(operand1.at(j), operand2.at(j));
                    }
                }
                else
                {
                    for (uint32_t j = 0; j < operand1.size(); j++)
                    {
                        output_operands.at(j) = TensorElementMultiply(operand1.at(j), operand2.at(j));
                    }
                }
                op_stack.push(output_operands);
            }
            else if (token_node->num_index == int(TokenType::TokenSin))
            {
                LOG(INFO) << "sin start";
                std::vector<std::shared_ptr<Tensor<float>>> operand1 = op_stack.top();
                CHECK(operand1.size() == batch_size)
                    << "The first operand doesn't have appropriate number of tensors, "
                       "which need "
                    << batch_size;
                op_stack.pop();
                std::vector<std::shared_ptr<Tensor<float>>> output_operands(batch_size);
                for (uint32_t j = 0; j < batch_size; j++)
                {

                    auto &sft = operand1.at(j);
                    sftensor result = std::make_shared<Tensor<float>>(sft->channels(),
                                                                      sft->rows(), sft->cols());
                    for (uint32_t k = 0; k < sft->channels(); k++)
                    {
                        result->slice(k) = arma::sin(sft->slice(k));
                    }
                    output_operands.at(j) = result;
                }
                op_stack.push(output_operands);
                LOG(INFO) << "sin over";
            }
            else
            {
                LOG(FATAL) << "Unknown operator type: " << token_node->num_index;
            }
        }
        CHECK(op_stack.size() == 1)
            << "The expression has more than one output operand!";
        std::vector<sftensor> output_node = op_stack.top();
        op_stack.pop();
        for (int i = 0; i < batch_size; ++i)
        {
            CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
            CHECK(outputs.at(i)->shapes() == output_node.at(i)->shapes());
            outputs.at(i) = output_node.at(i);
        }
        return InferStatus::InferSuccess;
    }

    ParseParameterAttrStatus ExpressionLayer::GetInstance(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &expression_layer)
    {
        CHECK(op != nullptr) << "Expression operator is nullptr";
        const auto &params = op->params;
        if (params.find("expr") == params.end())
        {
            return ParseParameterAttrStatus::ParameterMissingExpr;
        }

        auto statement_param =
            std::dynamic_pointer_cast<RuntimeParameterString>(params.at("expr"));
        if (statement_param == nullptr)
        {
            LOG(ERROR) << "Can not find the expression parameter";
            return ParseParameterAttrStatus::ParameterMissingExpr;
        }
        if (statement_param->type != RuntimeParameterType::ParameterString)
        {
            LOG(ERROR) << "Can not find the expression parameter";
            return ParseParameterAttrStatus::ParameterMissingExpr;
        }

        expression_layer = std::make_shared<ExpressionLayer>(statement_param->value);

        auto layer = std::dynamic_pointer_cast<ExpressionLayer>(expression_layer);
        std::shared_ptr<RuntimeOperator> runop = layer->runtime_operator_.lock();
        for (const auto &operand : runop->input_operands_seq)
        {
            layer->order.push_back(std::stoi(operand->name));
        }

        return ParseParameterAttrStatus::ParameterAttrParseSuccess;
    }

    LayerRegistererWrapper ExpressionGetInstance("pnnx.Expression",
                                                 ExpressionLayer::GetInstance);
}