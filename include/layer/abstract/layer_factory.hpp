#pragma once
#include <map>
#include <memory>
#include <string>
#include "layer/abstract/layer.hpp"
#include "runtime/runtime_op.hpp"

namespace star
{
    class LayerRegisterer
    {
    public:
        typedef ParseParameterAttrStatus (*Creator)(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &layer);

        typedef std::map<std::string, Creator> CreateRegistry;

    public:
        /**
         * 向注册表注册算子
         * @param layer_type 算子的类型
         * @param creator 需要注册算子的注册表
         */
        static void RegisterCreator(const std::string &layer_type,
                                    const Creator &creator);

        /**
         * 通过算子参数op来初始化Layer
         * @param op 保存了初始化Layer信息的算子
         * @return 初始化后的Layer
         */
        static std::shared_ptr<Layer> CreateLayer(
            const std::shared_ptr<RuntimeOperator> &op);

        /**
         * 返回算子的注册表
         * @return 算子的注册表
         */
        static CreateRegistry &Registry();

        /**
         * 返回所有已被注册算子的类型
         * @return 注册算子的类型列表
         */
        static std::vector<std::string> layer_types();
    };

    class LayerRegistererWrapper
    {
    public:
        LayerRegistererWrapper(const std::string &layer_type,
                               const LayerRegisterer::Creator &creator)
        {
            LayerRegisterer::RegisterCreator(layer_type, creator);
        }
    };

} //