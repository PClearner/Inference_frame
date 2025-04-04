#include <iostream>
#include "load_data.hpp"
#include "tensor_util.hpp"
#include "runtime/runtime_attr.hpp"
#include "runtime/runtime_operand.hpp"
#include "runtime/runtime_op.hpp"
#include "runtime/runtime_ir.hpp"

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
}

int main()
{
    bool deeporbreadth = false;
    LOG(INFO) << "===================================================================================================================";
    std::string param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/test_linear.pnnx.param";
    std::string bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/test_linear.pnnx.bin";
    std::string inputname = "pnnx_input_0";
    std::string outputname = "pnnx_output_0";
    // testgraph(param_path, bin_path, inputname, outputname, deeporbreadth);
    // LOG(INFO) << "===================================================================================================================";

    param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/resnet18_batch1.param";
    bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/resnet18_batch1.pnnx.bin";
    testgraph(param_path, bin_path, inputname, outputname, deeporbreadth);
    LOG(INFO) << "===================================================================================================================";
    deeporbreadth = true;
    testgraph2(param_path, bin_path, inputname, outputname, deeporbreadth);
    LOG(INFO) << "===================================================================================================================";

    if (topo_operators_1.size() != topo_operators_2.size())
    {
        LOG(INFO) << "topo_operators_1.size()!=topo_operators_2.size()";
    }
    for (size_t i = 0; i < topo_operators_1.size(); i++)
    {
        if (topo_operators_1[i]->name != topo_operators_2[i]->name)
        {
            LOG(INFO) << "(topo_operators_1[i]->name!=topo_operators_2[i]->name)|i:" << i;
            LOG(INFO) << "topo_operators_1[i]->name:" << topo_operators_1[i]->name << " | topo_operators_2[i]->name" << topo_operators_2[i]->name;
        }
    }
    param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/simple_ops.pnnx.param";
    bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/simple_ops.pnnx.bin";
    testgraph(param_path, bin_path, inputname, outputname, deeporbreadth);
    LOG(INFO) << "===================================================================================================================";

    // param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/simple_ops2.pnnx.param";
    // bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/simple_ops2.pnnx.bin";
    // testgraph(param_path, bin_path, inputname, outputname,deeporbreadth);
    return 0;
}