
#include "layer/abstract/layer_factory.hpp"
#include "layer/detail/relu.hpp"
#include "layer/abstract/param_layer.hpp"
#include "layer/detail/relu.hpp"
#include "layer/detail/sigmoid.hpp"
#include "tensor_util.hpp"
using namespace star;

static LayerRegisterer::CreateRegistry *RegistryGlobal()
{
    static LayerRegisterer::CreateRegistry *Registry = new LayerRegisterer::CreateRegistry();
    CHECK(Registry != nullptr) << "Global layer register init failed!";
    return Registry;
}

void Registryonly_test()
{
    LayerRegisterer::CreateRegistry *registry1 = RegistryGlobal();
    LayerRegisterer::CreateRegistry *registry2 = RegistryGlobal();

    LayerRegisterer::CreateRegistry *registry3 = RegistryGlobal();
    LayerRegisterer::CreateRegistry *registry4 = RegistryGlobal();
    float *a = new float{3};
    float *b = new float{4};
    CHECK_EQ(registry1, registry2);
}

ParseParameterAttrStatus MyTestCreator(
    const std::shared_ptr<RuntimeOperator> &op,
    std::shared_ptr<Layer> &layer)
{

    layer = std::make_shared<Layer>("test_layer");
    return ParseParameterAttrStatus::ParameterAttrParseSuccess;
}

void layercreate_test()
{
    LayerRegisterer::CreateRegistry registry1 = LayerRegisterer::Registry();
    LayerRegisterer::CreateRegistry registry2 = LayerRegisterer::Registry();
    // CHECK_EQ(registry1, registry2);
    LayerRegisterer::RegisterCreator("test_type", MyTestCreator);
    LayerRegisterer::CreateRegistry registry3 = LayerRegisterer::Registry();
    CHECK_EQ(registry3.size(), 1);
    const auto &tmp = LayerRegisterer::layer_types();
    // LOG(INFO) << tmp.size();
    // for (const auto &x : tmp)
    // {
    //     LOG(INFO) << x;
    // }
    // CHECK_NE(registry3.find("test_type"), registry3.end());
}

void RegisterCreator_test()
{
    // 注册了一个test_type_1算子
    LayerRegisterer::RegisterCreator("test_type_1", MyTestCreator);
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "test_type_1";
    std::shared_ptr<Layer> layer;
    CHECK_EQ(layer, nullptr);
    layer = LayerRegisterer::CreateLayer(op);
    CHECK_NE(layer, nullptr);
}

void Wrapper_test()
{
    LayerRegistererWrapper kReluGetInstance("test_type_2", MyTestCreator);
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "test_type_2";
    std::shared_ptr<Layer> layer;
    CHECK_EQ(layer, nullptr);
    layer = LayerRegisterer::CreateLayer(op);
    CHECK_NE(layer, nullptr);
}

void RELU_test()
{

    LayerRegisterer::RegisterCreator("nn.ReLU", ReluLayer::GetInstance);

    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "nn.ReLU";
    std::shared_ptr<Layer> layer;
    CHECK_EQ(layer, nullptr);
    layer = LayerRegisterer::CreateLayer(op);
    CHECK_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>(3, 4, 4);
    input_tensor->Rand();
    input_tensor->data() -= 0.5f;

    LOG(INFO) << input_tensor->data();

    std::vector<sftensor> inputs(1);
    std::vector<sftensor> outputs(1);
    inputs.at(0) = input_tensor;
    outputs.at(0) = TensorCreate({3, 4, 4});
    layer->Forward(inputs, outputs);

    for (const auto &output : outputs)
    {
        output->Show();
    }
}

void Sigmoid_test()
{

    LayerRegisterer::RegisterCreator("nn.Sigmoid", SigmoidLayer::GetInstance);

    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "nn.Sigmoid";
    std::shared_ptr<Layer> layer;
    CHECK_EQ(layer, nullptr);
    layer = LayerRegisterer::CreateLayer(op);
    CHECK_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>(3, 4, 4);
    input_tensor->Rand();
    input_tensor->data() -= 0.5f;

    LOG(INFO) << input_tensor->data();

    std::vector<sftensor> inputs(1);
    std::vector<sftensor> outputs(1);
    inputs.at(0) = input_tensor;
    outputs.at(0) = TensorCreate({3, 4, 4});
    layer->Forward(inputs, outputs);

    for (const auto &output : outputs)
    {
        output->Show();
    }
}

int main()
{
    Registryonly_test();
    layercreate_test();
    RegisterCreator_test();
    Wrapper_test();
    RELU_test();
    LOG(INFO) << "==============================================================================================";
    Sigmoid_test();

    return 0;
}
