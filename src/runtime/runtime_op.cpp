#include "runtime/runtime_op.hpp"

namespace star
{

    RuntimeOperator::~RuntimeOperator()
    {
        for (auto &[_, param] : this->params)
        {
            if (param != nullptr)
            {
                param = nullptr;
            }
        }
    }

    void RuntimeOperatorUtils::InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>> &operators)
    {
        if (operators.empty())
        {
            LOG(ERROR) << "Operators for init input shapes is empty!";
            return;
        }

        for (const auto &op : operators)
        {
            if (op->input_operands.empty())
            {
                continue;
            }
            else
            {
                for (const auto &inoperand : op->input_operands_seq)
                {
                    CHECK(inoperand->type == RuntimeDataType::TypeFloat32)
                        << "The graph only support float32 yet!";
                    CHECK(inoperand->shapes.size() == 2 || inoperand->shapes.size() == 3 || inoperand->shapes.size() == 4) << "Unsupported tensor shape sizes: " << inoperand->shapes.size();
                    CHECK(inoperand->shapes.at(0) >= 0) << "Dynamic batch size is not supported!";
                    if (inoperand->datas.empty())
                    {
                        inoperand->datas.resize(inoperand->shapes.at(0));
                    }
                    else
                    {
                        CHECK(inoperand->datas.size() == inoperand->shapes.at(0));
                    }
                }
            }
        }
    }

    void RuntimeOperatorUtils::InitOperatorOutput(const std::vector<pnnx::Operator *> &pnnx_operators, const std::vector<std::shared_ptr<RuntimeOperator>> &operators)
    {

        CHECK(!pnnx_operators.empty() && !operators.empty());
        CHECK(pnnx_operators.size() == operators.size());
        for (size_t i = 0; i < pnnx_operators.size(); i++)
        {

            const std::vector<pnnx::Operand *> operands = pnnx_operators.at(i)->outputs;
            CHECK(pnnx_operators.at(i)->outputs.size() <= 1) << "Only support one node one output yet!";
            if (pnnx_operators.at(i)->outputs.empty())
            {
                continue;
            }
            CHECK(pnnx_operators.at(i)->outputs.size() == 1) << "Only support one output in the KuiperInfer";
            // 一个节点仅支持一个输出，实际上在pnnx中一个节点拥有两个不同输出的情况也是不存在的

            const auto &pnnx_op = pnnx_operators[i];
            const auto &op = operators[i];

            CHECK(pnnx_op->outputs.size() == 1) << "Only support one output in the KuiperInfer";

            const auto &tt = pnnx_op->outputs.front();

            CHECK(tt != nullptr) << "Operand output is null";
            // 得到需要初始化的输出空间
            // 获取节点的输出张量应有形状
            CHECK(tt->shape.at(0) >= 0) << "Dynamic batch size is not supported!";
            CHECK(tt->shape.size() == 2 || tt->shape.size() == 4 ||
                  tt->shape.size() == 3)
                << "Unsupported shape sizes: " << tt->shape.size();

            // initialize operand datas
            if (op->output_operands == nullptr)
            {
                op->output_operands = std::make_shared<RuntimeOperand>();
                op->output_operands->name = tt->name + "_output";
                op->output_operands->type = RuntimeDataType::TypeFloat32;
                op->output_operands->shapes = tt->shape;
                for (int32_t i = 0; i < op->output_operands->shapes.at(0); i++)
                {
                    sftensor tmp;
                    if (op->output_operands->shapes.size() == 4)
                    {
                        tmp = std::make_shared<Tensor<float>>(op->output_operands->shapes.at(1), op->output_operands->shapes.at(2), op->output_operands->shapes.at(3));
                    }
                    else if (op->output_operands->shapes.size() == 3)
                    {
                        tmp = std::make_shared<Tensor<float>>(op->output_operands->shapes.at(1), op->output_operands->shapes.at(2));
                    }
                    else if (op->output_operands->shapes.size() == 2)
                    {
                        tmp = std::make_shared<Tensor<float>>(op->output_operands->shapes.at(1));
                    }
                    op->output_operands->datas.push_back(tmp);
                }
            }
            else
            {
                CHECK(op->output_operands->datas.size() == op->output_operands->shapes.at(0));
                CHECK(op->output_operands->name == tt->name);
                CHECK(op->output_operands->type == RuntimeDataType::TypeFloat32);
                CHECK(op->output_operands->shapes == tt->shape);
                for (const auto &tmp : op->output_operands->datas)
                {
                    CHECK(tmp != nullptr);
                    CHECK(tmp->shapes().size() == op->output_operands->shapes.size() - 1);

                    if (tt->shape.size() == 4)
                    {
                        if (tmp->shapes().at(0) != tt->shape.at(1) ||
                            tmp->shapes().at(1) != tt->shape.at(2) ||
                            tmp->shapes().at(2) != tt->shape.at(3))
                        {
                            DLOG(WARNING)
                                << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes = std::vector<uint32_t>{
                                (uint32_t)tt->shape.at(1), (uint32_t)tt->shape.at(2),
                                (uint32_t)tt->shape.at(3)};
                            tmp->Reshape(target_shapes);

                            void Reshape(const std::vector<uint32_t> &shapes,
                                         bool row_major);
                        }
                    }
                    else if (tt->shape.size() == 2)
                    {
                        if (tmp->shapes().at(0) != 1 ||
                            tmp->shapes().at(1) != tt->shape.at(1) ||
                            tmp->shapes().at(2) != 1)
                        {
                            DLOG(WARNING)
                                << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes =
                                std::vector<uint32_t>{(uint32_t)tt->shape.at(1)};
                            tmp->Reshape(target_shapes);
                        }
                    }
                    else
                    {
                        // current shape is 3
                        if (tmp->shapes().at(0) != 1 ||
                            tmp->shapes().at(1) != tt->shape.at(1) ||
                            tmp->shapes().at(2) != tt->shape.at(2))
                        {
                            DLOG(WARNING)
                                << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes = std::vector<uint32_t>{
                                (uint32_t)tt->shape.at(1), (uint32_t)tt->shape.at(2)};
                            tmp->Reshape(target_shapes);
                        }
                    }
                }
            }
        }
    }
}