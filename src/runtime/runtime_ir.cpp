#include "runtime/runtime_ir.hpp"
#include "status_code.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace star
{

    RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    {
        set_bin_path(bin_path);
        set_param_path(param_path);
    }

    /**
     * ����Ȩ���ļ�
     * @param bin_path Ȩ���ļ�·��
     */
    void RuntimeGraph::set_bin_path(const std::string &bin_path)
    {
        this->bin_path_ = bin_path;
    }

    /**
     * ���ýṹ�ļ�
     * @param param_path  �ṹ�ļ�·��
     */
    void RuntimeGraph::set_param_path(const std::string &param_path)
    {
        this->param_path_ = param_path;
    }

    /**
     * ���ؽṹ�ļ�
     * @return ���ؽṹ�ļ�
     */
    const std::string &RuntimeGraph::param_path() const
    {
        return this->param_path_;
    }

    /**
     * ����Ȩ���ļ�
     * @return ����Ȩ���ļ�
     */
    const std::string &RuntimeGraph::bin_path() const
    {
        return this->bin_path_;
    }

    /**
     * ����ͼ�ĳ�ʼ��
     * @return �Ƿ��ʼ���ɹ�
     */
    bool RuntimeGraph::Init()
    {
        // CHECK
        if (this->bin_path_.empty() || this->param_path_.empty())
        {
            LOG(ERROR) << "The bin path or param path is empty";
            return false;
        }
        //

        this->graph_ = std::make_unique<pnnx::Graph>();
        int load_result = this->graph_->load(param_path_, bin_path_);
        if (load_result != 0)
        {
            LOG(ERROR) << "Can not find the param path or bin path: " << param_path_
                       << " " << bin_path_;
            return false;
        }

        std::vector<pnnx::Operator *> operators = this->graph_->ops;
        if (operators.empty())
        {
            LOG(ERROR) << "Can not read the layers' define";
            return false;
        }

        this->operators_.clear();
        this->operators_maps_.clear();
        for (size_t i = 0; i < operators.size(); i++)
        {
            if (operators[i] == nullptr)
            {
                LOG(ERROR) << "Meet the empty node";
                continue;
            }
            else
            {
                std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
                op->name = operators[i]->name;
                op->type = operators[i]->type;

                // inputs
                if (!operators[i]->inputs.empty())
                {
                    InitGraphOperatorsInput(operators[i]->inputs, op);
                }

                // outputs
                if (!operators[i]->outputs.empty())
                {
                    InitGraphOperatorsOutput(operators[i]->outputs, op);
                }

                // Attr
                if (!operators[i]->attrs.empty())
                {
                    InitGraphAttrs(operators[i]->attrs, op);
                }

                // paramter
                if (!operators[i]->params.empty())
                {
                    InitGraphParams(operators[i]->params, op);
                }

                this->operators_.push_back(op);
                this->operators_maps_.insert({op->name, op});
            }
        }
        return true;
    }

    const std::vector<std::shared_ptr<RuntimeOperator>> &RuntimeGraph::operators() const
    {
        return this->operators_;
    }

    void RuntimeGraph::InitGraphOperatorsInput(
        const std::vector<pnnx::Operand *> &inputs,
        const std::shared_ptr<RuntimeOperator> &runtime_operator)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            if (inputs[i] == nullptr)
            {
                continue;
            }
            std::shared_ptr<RuntimeOperand> oper = std::make_shared<RuntimeOperand>();
            switch (inputs[i]->type)
            {
            case 1:
            {
                oper->type = RuntimeDataType::kTypeFloat32;
                break;
            }
            case 0:
            {
                oper->type = RuntimeDataType::kTypeUnknown;
                break;
            }
            default:
            {
                LOG(FATAL) << "Unknown input operand type: " << inputs[i]->type;
                break;
            }
            }
            oper->name = inputs[i]->producer->name;

            oper->shapes = inputs[i]->shape;

            runtime_operator->input_operands.insert({oper->name, oper});
            runtime_operator->input_operands_seq.push_back(oper);
        }
    }

    /**
     * ��ʼ��kuiper infer����ͼ�ڵ��е����������
     * @param outputs pnnx�е����������
     * @param runtime_operator ����ͼ�ڵ�
     */
    void RuntimeGraph::InitGraphOperatorsOutput(
        const std::vector<pnnx::Operand *> &outputs,
        const std::shared_ptr<RuntimeOperator> &runtime_operator)
    {
        for (size_t i = 0; i < outputs.size(); i++)
        {
            if (outputs[i] == nullptr)
            {
                continue;
            }

            for (size_t j = 0; j < outputs[i]->consumers.size(); j++)
            {
                runtime_operator->output_names.push_back(outputs[i]->consumers[j]->name);
            }
        }
    }

    /**
     * ��ʼ��kuiper infer����ͼ�еĽڵ�����
     * @param attrs pnnx�еĽڵ�����
     * @param runtime_operator ����ͼ�ڵ�
     */
    void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                                      const std::shared_ptr<RuntimeOperator> &runtime_operator)
    {
        for (const auto &attr : attrs)
        {
            std::shared_ptr<RuntimeAttribute> tmp = std::make_shared<RuntimeAttribute>();
            tmp->weight_data = attr.second.data;
            tmp->shape = attr.second.shape;
            switch (attr.second.type)
            {
            case 1:
            {
                tmp->type = RuntimeDataType::kTypeFloat32;
                break;
            }
            case 0:
            {
                tmp->type = RuntimeDataType::kTypeUnknown;
                break;
            }
            default:
            {
                LOG(FATAL) << "Unknown attr type: " << attr.second.type;
                break;
            }
            }
            runtime_operator->attribute.insert({attr.first, tmp});
        }
    }

    /**
     * ��ʼ��kuiper infer����ͼ�еĽڵ����
     * @param params pnnx�еĲ�������
     * @param runtime_operator ����ͼ�ڵ�
     */
    void RuntimeGraph::InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                       const std::shared_ptr<RuntimeOperator> &runtime_operator)
    {
        for (const auto &[name, param] : params)
        {
            switch (param.type)
            {
            case 1:
            {
                std::shared_ptr<RuntimeParameterBool> tmp = std::make_shared<RuntimeParameterBool>();
                tmp->value = param.b;
                runtime_operator->params.insert({name, tmp});
                break;
            }
            case 2:
            {
                std::shared_ptr<RuntimeParameterInt> tmp = std::make_shared<RuntimeParameterInt>();
                tmp->value = param.i;
                runtime_operator->params.insert({name, tmp});
                break;
            }
            case 3:
            {
                std::shared_ptr<RuntimeParameterFloat> tmp = std::make_shared<RuntimeParameterFloat>();
                tmp->value = param.f;
                runtime_operator->params.insert({name, tmp});
                break;
            }
            case 4:
            {
                std::shared_ptr<RuntimeParameterString> tmp = std::make_shared<RuntimeParameterString>();
                tmp->value = param.i;
                runtime_operator->params.insert({name, tmp});
                break;
            }
            case 5:
            {
                std::shared_ptr<RuntimeParameterIntArray> tmp = std::make_shared<RuntimeParameterIntArray>();
                tmp->value = param.ai;
                runtime_operator->params.insert({name, tmp});
                break;
            }
            case 6:
            {
                std::shared_ptr<RuntimeParameterFloatArray> tmp = std::make_shared<RuntimeParameterFloatArray>();
                tmp->value = param.af;
                runtime_operator->params.insert({name, tmp});
                break;
            }
            case 7:
            {
                std::shared_ptr<RuntimeParameterStringArray> tmp = std::make_shared<RuntimeParameterStringArray>();
                tmp->value = param.as;
                runtime_operator->params.insert({name, tmp});
                break;
            }
            default:
            {
                LOG(FATAL) << "Unknown parameter type: " << param.type;
                break;
            }
            }
        }
    }

}