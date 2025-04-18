
#include "layer/abstract/layer_factory.hpp"

namespace star
{
    /**
     * 向注册表注册算子
     * @param layer_type 算子的类型
     * @param creator 需要注册算子的注册表
     */
    void LayerRegisterer::RegisterCreator(const std::string &layer_type,
                                          const Creator &creator)
    {
        CHECK(creator != nullptr);

        auto &Registerer = LayerRegisterer::Registry();
        CHECK_EQ(Registerer.count(layer_type), 0)
            << "Layer type: " << layer_type << " has already registered!";
        Registerer.insert({layer_type, creator});
    }

    /**
     * 通过算子参数op来初始化Layer
     * @param op 保存了初始化Layer信息的算子
     * @return 初始化后的Layer
     */
    std::shared_ptr<Layer> LayerRegisterer::CreateLayer(
        const std::shared_ptr<RuntimeOperator> &op)
    {
        std::string &layer_type = op->type;
        auto &Registerer = LayerRegisterer::Registry();
        LOG_IF(FATAL, Registerer.count(layer_type) <= 0)
            << "Can not find the layer type: " << layer_type;
        std::shared_ptr<Layer> createlayer;
        const auto &creator = Registerer.find(layer_type)->second;
        LOG_IF(FATAL, !creator) << "Layer creator is empty!";
        const auto &status = creator(op, createlayer);
        LOG_IF(FATAL, status != ParseParameterAttrStatus::ParameterAttrParseSuccess)
            << "Create the layer: " << layer_type
            << " failed, error code: " << int(status);
        return createlayer;
    }

    /**
     * 返回算子的注册表
     * @return 算子的注册表
     */
    LayerRegisterer::CreateRegistry &LayerRegisterer::Registry()
    {
        static CreateRegistry *Registerer = new CreateRegistry();
        CHECK(Registry != nullptr) << "Global layer register init failed!";
        return *Registerer;
    }

    /**
     * 返回所有已被注册算子的类型
     * @return 注册算子的类型列表
     */
    std::vector<std::string> LayerRegisterer::layer_types()
    {
        std::vector<std::string> result;
        const auto &Register = LayerRegisterer::Registry();
        for (const auto &i : Register)
        {
            result.push_back(i.first);
        }
        return result;
    }

} //