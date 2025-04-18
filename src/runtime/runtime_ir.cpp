#include "runtime/runtime_ir.hpp"
#include "status_code.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <stack>

namespace star
{

    RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    {
        set_bin_path(bin_path);
        set_param_path(param_path);
    }

    std::shared_ptr<Layer> RuntimeGraph::CreateLayer(
        const std::shared_ptr<RuntimeOperator> &op)
    {
        LOG_IF(FATAL, !op) << "Operator is empty!";
        auto layer = LayerRegisterer::CreateLayer(op);
        LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
        return layer;
    }

    /**
     * 设置权重文件
     * @param bin_path 权重文件路径
     */
    void RuntimeGraph::set_bin_path(const std::string &bin_path)
    {
        this->bin_path_ = bin_path;
    }

    /**
     * 设置结构文件
     * @param param_path  结构文件路径
     */
    void RuntimeGraph::set_param_path(const std::string &param_path)
    {
        this->param_path_ = param_path;
    }

    /**
     * 返回结构文件
     * @return 返回结构文件
     */
    const std::string &RuntimeGraph::param_path() const
    {
        return this->param_path_;
    }

    /**
     * 返回权重文件
     * @return 返回权重文件
     */
    const std::string &RuntimeGraph::bin_path() const
    {
        return this->bin_path_;
    }

    /**
     * 计算图的初始化
     * @return 是否初始化成功
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
        this->graph_state_ = GraphState::NeedBuild;
        return true;
    }

    const std::vector<std::shared_ptr<RuntimeOperator>> &RuntimeGraph::operators() const
    {
        return this->operators_;
    }

    void RuntimeGraph::Build(const std::string &input_name, const std::string &output_name, bool deeporbreath)
    {
        // check
        if (this->graph_state_ == GraphState::Complete)
        {
            LOG(INFO) << "Model has been built already!";
            return;
        }

        if (this->graph_state_ == GraphState::NeedInit)
        {
            bool init_graph = Init();
            LOG_IF(FATAL, !init_graph) << "Init graph failed!";
        }

        CHECK(graph_state_ >= GraphState::NeedBuild)
            << "Graph status error, current state is " << int(graph_state_);
        LOG_IF(FATAL, this->operators_.empty())
            << "Graph operators is empty, may be no init";
        //

        // 获取当前节点的所有后继节点的names，遍历根据next_op_name从operators_maps_中插入所需要的节点
        for (const auto &current_op : this->operators_)
        {
            const auto &outnames = current_op->output_names;
            for (auto outname : outnames)
            {
                auto tmp = this->operators_maps_.find(outname);
                CHECK(tmp != this->operators_maps_.end());
                current_op->output_operators.insert({outname, tmp->second});
            }
        }

        // 初始化节点的输入和输出空间
        // initiaze operands
        RuntimeOperatorUtils::InitOperatorInput(this->operators_);
        RuntimeOperatorUtils::InitOperatorOutput(this->graph_->ops, this->operators_);

        // sort
        // 构建拓扑顺序
        sort(deeporbreath);

        // layer initialize
        for (const auto &op : this->topo_operators_)
        {
            // 除了输入和输出节点，都创建layer
            if (op->type != "pnnx.Input" && op->type != "pnnx.Output")
            {
                std::shared_ptr<Layer> layer = RuntimeGraph::CreateLayer(op);
                CHECK(layer != nullptr)
                    << "Layer " << op->name << " create failed!";
                if (layer)
                {
                    op->layer = layer;
                    layer->set_runtime_operator(op);
                }
            }
        }

        // 收尾工作
        this->graph_state_ = GraphState::Complete;
        this->input_name_ = input_name;
        this->output_name_ = output_name;

        if (this->graph_ != nullptr)
        {
            this->graph_.reset();
            this->graph_ = nullptr;
        }
    }

    void RuntimeGraph::sort(bool deeporbreadth)
    {
        if (deeporbreadth)
        {
            deepsearch();
        }
        else
        {
            breadthsearch();
        }
    }

    void RuntimeGraph::deepsearch()
    {
        std::vector<std::shared_ptr<RuntimeOperator>> result;
        std::stack<std::shared_ptr<RuntimeOperator>> cap;

        this->topo_operators_.clear();
        std::shared_ptr<RuntimeOperator> start;
        for (const auto &s : this->operators_)
        {
            if (s->type == "pnnx.Input" && !s->has_forward)
            {
                start = s;
            }
        }

        cap.push(start);

        while (!cap.empty())
        {
            std::shared_ptr<RuntimeOperator> tmp = cap.top();

            for (const auto &s : tmp->output_names)
            {
                const auto &sp = this->operators_maps_.find(s);
                CHECK(sp != this->operators_maps_.end());
                if (!sp->second->has_forward)
                {
                    cap.push(sp->second);
                    break;
                }
            }

            if (tmp == cap.top())
            {
                tmp->has_forward = true;
                result.push_back(tmp);
                cap.pop();
            }
        }

        for (int i = result.size() - 1; i >= 0; i--)
        {

            this->topo_operators_.push_back(result[i]);
        }
    }

    void RuntimeGraph::breadthsearch()
    {
        std::vector<std::shared_ptr<RuntimeOperator>> result;
        std::queue<std::shared_ptr<RuntimeOperator>> cap;

        this->topo_operators_.clear();
        std::shared_ptr<RuntimeOperator> start;

        std::unordered_map<std::string, size_t> index;
        std::unordered_map<std::string, bool> incap;

        for (const auto &s : this->operators_)
        {
            if (s->type == "pnnx.Input" && !s->has_forward)
            {
                start = s;
            }
        }

        cap.push(start);

        while (!cap.empty())
        {
            std::shared_ptr<RuntimeOperator> tmp = cap.front();

            tmp->has_forward = true;
            result.push_back(tmp);
            index[tmp->name] = result.size() - 1;
            // LOG(INFO) << "[breadthsearch]|tmp:" << tmp->name;
            for (const auto &s : tmp->output_names)
            {
                auto sp = this->operators_maps_.find(s);
                CHECK(sp != this->operators_maps_.end());
                if (!sp->second->has_forward)
                {
                    if (incap.find(sp->second->name) == incap.end())
                    {
                        cap.push(sp->second);
                        incap[sp->second->name] = true;
                    }
                }
                else
                {
                    auto x = tmp;
                    auto y = sp->second;

                    size_t index1 = index[x->name];
                    size_t index2 = index[y->name];
                    CHECK(index1 != index2);

                    if (index1 > index2)
                    {
                        // LOG(INFO) << "[breadthsearch]|sp:" << y->name << " has_forward." << " exchange with tmp:" << x->name;
                        std::shared_ptr<RuntimeOperator> tt = result[index2];
                        result[index2] = result[index1];
                        result[index1] = tt;
                        index[x->name] = index2;
                        index[y->name] = index1;

                        while (1)
                        {
                            size_t min_index = SIZE_MAX;
                            std::string min_name = "";
                            index1 = index[y->name];
                            for (auto on : y->output_names)
                            {
                                if (index.find(on) == index.end())
                                {
                                    continue;
                                }
                                index2 = index[on];
                                CHECK(index1 != index2);

                                if (index1 > index2)
                                {
                                    // LOG(INFO) << "[breadthsearch]|[in while]||sp index: " << index2 << " sp name: " << on << " | ttmp index: " << index1 << " ttmp name: " << y->name;
                                    if (min_index > index2)
                                    {
                                        min_index = index2;
                                        min_name = on;
                                    }
                                }
                            }
                            if (min_name == "")
                            {
                                break;
                            }
                            else
                            {
                                // LOG(INFO) << "[breadthsearch]|[in while]||sp:" << y->name << " has_forward." << " exchange with tmp:" << min_name;
                                index2 = index[min_name];
                                std::shared_ptr<RuntimeOperator> tt = result[index2];
                                result[index2] = result[index1];
                                result[index1] = tt;
                                index[min_name] = index2;
                                index[y->name] = index1;
                                y = this->operators_maps_[min_name];
                            }
                        }
                    }
                    else
                    {
                        // LOG(INFO) << "[breadthsearch]|sp index:" << index2 << " | ttmp index:" << index1;
                        break;
                    }
                }
            }
            cap.pop();
        }
        this->topo_operators_ = result;
    }

    const std::vector<std::shared_ptr<RuntimeOperator>> &RuntimeGraph::get_topo_queues() const
    {
        return topo_operators_;
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
                oper->type = RuntimeDataType::TypeFloat32;
                break;
            }
            case 0:
            {
                oper->type = RuntimeDataType::TypeUnknown;
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
     * 初始化kuiper infer计算图节点中的输出操作数
     * @param outputs pnnx中的输出操作数
     * @param runtime_operator 计算图节点
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
     * 初始化kuiper infer计算图中的节点属性
     * @param attrs pnnx中的节点属性
     * @param runtime_operator 计算图节点
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
                tmp->type = RuntimeDataType::TypeFloat32;
                break;
            }
            case 0:
            {
                tmp->type = RuntimeDataType::TypeUnknown;
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
     * 初始化kuiper infer计算图中的节点参数
     * @param params pnnx中的参数属性
     * @param runtime_operator 计算图节点
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

    RuntimeGraph::GraphState RuntimeGraph::graph_state() const { return this->graph_state_; }
}
