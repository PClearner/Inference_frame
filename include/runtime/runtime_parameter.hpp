#pragma once
#include "status_code.hpp"
#include <string>
#include <vector>

/**
 * 计算节点中的参数信息，参数一共可以分为如下的几类
 * 1.int
 * 2.float
 * 3.string
 * 4.bool
 * 5.int array
 * 6.string array
 * 7.float array
 */
namespace star
{
    struct RuntimeParameter
    { /// 计算节点中的参数信息
        virtual ~RuntimeParameter() = default;

        explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::ParameterUnknown) : type(type)
        {
        }
        RuntimeParameterType type = RuntimeParameterType::ParameterUnknown;
    };

    struct RuntimeParameterInt : public RuntimeParameter
    {
        RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::ParameterInt)
        {
        }
        int value = 0;
    };

    struct RuntimeParameterFloat : public RuntimeParameter
    {
        RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::ParameterFloat)
        {
        }
        float value = 0.f;
    };

    struct RuntimeParameterString : public RuntimeParameter
    {
        RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::ParameterString)
        {
        }
        std::string value;
    };

    struct RuntimeParameterIntArray : public RuntimeParameter
    {
        RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::ParameterIntArray)
        {
        }
        std::vector<int> value;
    };

    struct RuntimeParameterFloatArray : public RuntimeParameter
    {
        RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::ParameterFloatArray)
        {
        }
        std::vector<float> value;
    };

    struct RuntimeParameterStringArray : public RuntimeParameter
    {
        RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::ParameterStringArray)
        {
        }
        std::vector<std::string> value;
    };

    struct RuntimeParameterBool : public RuntimeParameter
    {
        RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::ParameterBool)
        {
        }
        bool value = false;
    };
}