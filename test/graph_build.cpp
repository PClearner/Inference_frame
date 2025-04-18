#include <iostream>
#include "load_data.hpp"
#include "tensor_util.hpp"
#include "runtime/runtime_attr.hpp"
#include "runtime/runtime_operand.hpp"
#include "runtime/runtime_op.hpp"
#include "runtime/runtime_ir.hpp"
// #include <gtest/gtest.h>

using namespace star;
static std::string ShapeStr(const std::vector<int> &shapes)
{
    std::ostringstream ss;
    for (int i = 0; i < shapes.size(); ++i)
    {
        ss << shapes.at(i);
        if (i != shapes.size() - 1)
        {
            ss << " x ";
        }
    }
    return ss.str();
}

std::vector<std::shared_ptr<RuntimeOperator>> topo_operators_1;
std::vector<std::shared_ptr<RuntimeOperator>> topo_operators_2;

void testgraph(std::string param_path, std::string bin_path, std::string inputname, std::string outputname, bool deeporbreadth = true)
{

    std::unique_ptr<RuntimeGraph> graph;
    graph = std::make_unique<RuntimeGraph>(param_path, bin_path);
    int load_result = graph->Init();
    CHECK_EQ(load_result, 1);
    graph->Build(inputname, outputname, deeporbreadth);
    const auto &topo_queues = graph->get_topo_queues();

    int index = 0;

    // const auto &ops = graph->get_operators();
    // for (int i = 0; i < ops.size(); ++i)
    // {
    //     const auto &op = ops.at(i);
    //     std::string op_name = op->name;
    //     LOG(INFO) << op_name;
    //     if (op_name == "linear")
    //     {
    //         for (const auto &attr : op->attribute)
    //         {
    //             LOG(INFO) << "  | linear_" << attr.first << "\n";
    //         }
    //     }
    // }
    // LOG(INFO) << "\n";
    // LOG(INFO) << "operator:";
    // LOG(INFO) << "---------------------------------------------";
    for (const auto &operator_ : topo_queues)
    {
        LOG(INFO) << "Index: " << index << " Type: " << operator_->type
                  << " Name: " << operator_->name;
        index += 1;
    }

    CHECK(int(graph->graph_state()) == 0);
    topo_operators_1 = topo_queues;
}

void testgraph2(std::string param_path, std::string bin_path, std::string inputname, std::string outputname, bool deeporbreadth = true)
{

    std::unique_ptr<RuntimeGraph> graph;
    graph = std::make_unique<RuntimeGraph>(param_path, bin_path);
    int load_result = graph->Init();
    CHECK_EQ(load_result, 1);
    graph->Build(inputname, outputname, deeporbreadth);
    const auto &topo_queues = graph->get_topo_queues();

    int index = 0;

    // const auto &ops = graph->get_operators();
    // for (int i = 0; i < ops.size(); ++i)
    // {
    //     const auto &op = ops.at(i);
    //     std::string op_name = op->name;
    //     LOG(INFO) << op_name;
    //     if (op_name == "linear")
    //     {
    //         for (const auto &attr : op->attribute)
    //         {
    //             LOG(INFO) << "  | linear_" << attr.first << "\n";
    //         }
    //     }
    // }
    // LOG(INFO) << "\n";
    // LOG(INFO) << "operator:";
    // LOG(INFO) << "---------------------------------------------";
    for (const auto &operator_ : topo_queues)
    {
        LOG(INFO) << "Index: " << index << " Type: " << operator_->type
                  << " Name: " << operator_->name;
        index += 1;
    }

    CHECK(int(graph->graph_state()) == 0);
    topo_operators_2 = topo_queues;
    LOG(INFO) << "===================================================================================================================";
    LOG(INFO) << "graph InitOperatorOutput check";
    std::vector<pnnx::Operator *> pnnx_operators;
    std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
    for (int i = 0; i < 4; ++i)
    {
        pnnx::Operator *pnnx_op = new pnnx::Operator;
        pnnx::Operand *pnnx_number = new pnnx::Operand;
        pnnx_number->type = 1;
        pnnx_number->shape = std::vector<int>{8, 3, 32, 32};
        pnnx_op->outputs.push_back(pnnx_number);
        pnnx_operators.push_back(pnnx_op);
        run_ops.push_back(std::make_shared<RuntimeOperator>());
    }

    RuntimeOperatorUtils::InitOperatorOutput(pnnx_operators, run_ops);

    for (const auto &run_op : run_ops)
    {
        const auto &output_datas = run_op->output_operands;
        CHECK_EQ(output_datas->shapes.size(), 4);
        CHECK_EQ(output_datas->datas.size(), 8);
        for (const auto &output_data : output_datas->datas)
        {
            const auto &raw_shapes = output_data->raw_shapes();
            CHECK_EQ(raw_shapes.size(), 3);
            CHECK_EQ(raw_shapes.at(0), 3);
            CHECK_EQ(raw_shapes.at(1), 32);
            CHECK_EQ(raw_shapes.at(2), 32);

            CHECK_EQ(output_data->rows(), 32);
            CHECK_EQ(output_data->cols(), 32);
            CHECK_EQ(output_data->channels(), 3);
            output_data->data().resize(32, 16, 6);
        }
    }
    LOG(INFO) << "===================================================================================================================";
    LOG(INFO) << "graph InitOperatorOutput check";


    LOG(INFO) << "===================================================================================================================";
    LOG(INFO) << "graph state check";
    // CHECK_EQ(int(graph->graph_state()), -2);
    // const bool init_success = graph->Init();
    // CHECK_EQ(init_success, true);
    // CHECK_EQ(int(graph->graph_state()), -1);
    // graph->Build("pnnx_input_0", "pnnx_output_0");
    // CHECK_EQ(int(graph->graph_state()), 0);

    LOG(INFO) << "===================================================================================================================";
    LOG(INFO) << "graph outputtensor check";
    const auto &ops = graph->operators();
    for (const auto &op : ops)
    {
        LOG(INFO) << op->name;
        // 打印op输出空间的张量
        const auto &operand = op->output_operands;
        if (operand == nullptr || operand->datas.empty())
        {
            continue;
        }
        const uint32_t batch_size = operand->datas.size();
        LOG(INFO) << "operand name: " << operand->name;
        LOG(INFO) << "batch: " << batch_size;

        for (uint32_t i = 0; i < batch_size; ++i)
        {
            const auto &data = operand->datas.at(i);
            LOG(INFO) << "channel: " << data->channels()
                      << " height: " << data->rows() << " cols: " << data->cols();
        }
    }
}

int main()
{
    bool deeporbreadth = true;
    LOG(INFO) << "===================================================================================================================";
    std::string param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/test_linear.pnnx.param";
    std::string bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/test_linear.pnnx.bin";
    std::string inputname = "pnnx_input_0";
    std::string outputname = "pnnx_output_0";
    // testgraph(param_path, bin_path, inputname, outputname, deeporbreadth);
    // LOG(INFO) << "===================================================================================================================";

    // param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/resnet18_batch1.param";
    // bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/resnet18_batch1.pnnx.bin";
    // testgraph(param_path, bin_path, inputname, outputname, deeporbreadth);
    // LOG(INFO) << "===================================================================================================================";
    // deeporbreadth = true;
    // testgraph2(param_path, bin_path, inputname, outputname, deeporbreadth);
    // LOG(INFO) << "===================================================================================================================";

    // if (topo_operators_1.size() != topo_operators_2.size())
    // {
    //     LOG(INFO) << "topo_operators_1.size()!=topo_operators_2.size()";
    // }
    // for (size_t i = 0; i < topo_operators_1.size(); i++)
    // {
    //     if (topo_operators_1[i]->name != topo_operators_2[i]->name)
    //     {
    //         LOG(INFO) << "(topo_operators_1[i]->name!=topo_operators_2[i]->name)|i:" << i;
    //         LOG(INFO) << "topo_operators_1[i]->name:" << topo_operators_1[i]->name << " | topo_operators_2[i]->name" << topo_operators_2[i]->name;
    //     }
    // }
    param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/simple_ops.pnnx.param";
    bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/simple_ops.pnnx.bin";
    testgraph2(param_path, bin_path, inputname, outputname, deeporbreadth);
    LOG(INFO) << "===================================================================================================================";

    // param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/simple_ops2.pnnx.param";
    // bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/simple_ops2.pnnx.bin";
    // testgraph(param_path, bin_path, inputname, outputname,deeporbreadth);
    return 0;
}