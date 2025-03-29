#include <iostream>
#include "load_data.hpp"
#include "tensor_util.hpp"
#include "runtime/runtime_attr.hpp"
#include "runtime/runtime_operand.hpp"
#include "runtime/runtime_op.hpp"
#include "runtime/runtime_ir.hpp"

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

void testgraph()
{
    using namespace star;
    std::string param_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/test_linear.pnnx.param";
    std::string bin_path = "/root/star/inference/inference_frame/bin/graph/model/model_file/test_linear.pnnx.bin";
    std::unique_ptr<RuntimeGraph> graph;
    graph = std::make_unique<RuntimeGraph>(param_path, bin_path);
    int load_result = graph->Init();
    CHECK_EQ(load_result, 1);
    const auto &ops = graph->get_operators();
    for (int i = 0; i < ops.size(); ++i)
    {
        const auto &op = ops.at(i);
        std::string op_name = op->name;
        LOG(INFO) << op_name;
        if (op_name == "linear")
        {
            for (const auto &attr : op->attribute)
            {
                LOG(INFO) << "  | linear_" << attr.first << "\n";
            }
        }
    }
    LOG(INFO) << "\n";
    LOG(INFO) << "operator:";
    LOG(INFO) << "---------------------------------------------";
    for (int i = 0; i < ops.size(); ++i)
    {
        const auto &op = ops.at(i);
        LOG(INFO) << "OP Name: " << op->name;
        LOG(INFO) << "OP Inputs";
        for (int j = 0; j < op->input_operands_seq.size(); ++j)
        {
            LOG(INFO) << "Input name: " << op->input_operands_seq.at(j)->name
                      << " shape: " << ShapeStr(op->input_operands_seq.at(j)->shapes);
        }

        LOG(INFO) << "OP Output";
        for (int j = 0; j < op->output_names.size(); ++j)
        {
            LOG(INFO) << "Output name: " << op->output_names.at(j);
            //   << " shape: " << ShapeStr(op->output_operators[op->output_names.at(j)]->shapes);
        }
        LOG(INFO) << "---------------------------------------------";
    }
}

int main()
{
    testgraph();
    return 0;
}