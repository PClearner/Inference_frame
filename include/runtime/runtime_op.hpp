#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
// #include "layer/abstract/layer.hpp"
#include "runtime/ir.h"
#include "runtime/runtime_attr.hpp"
#include "runtime/runtime_operand.hpp"
#include "runtime/runtime_parameter.hpp"

namespace star
{
    class Layer;

    /// ����ͼ�еļ���ڵ�
    struct RuntimeOperator
    {
        virtual ~RuntimeOperator() {};

        bool has_forward = false;
        std::string name;             /// ����ڵ������
        std::string type;             /// ����ڵ������
        std::shared_ptr<Layer> layer; /// �ڵ��Ӧ�ļ���Layer

        std::vector<std::string> output_names;           /// �ڵ������ڵ�����
        std::shared_ptr<RuntimeOperand> output_operands; /// �ڵ�����������

        std::map<std::string, std::shared_ptr<RuntimeOperand>>
            input_operands; /// �ڵ�����������
        std::vector<std::shared_ptr<RuntimeOperand>>
            input_operands_seq; /// �ڵ�������������˳������
        std::map<std::string, std::shared_ptr<RuntimeOperator>>
            output_operators; /// ����ڵ�����ֺͽڵ��Ӧ

        std::map<std::string, std::shared_ptr<RuntimeParameter>> params; /// ���ӵĲ�����Ϣ
        std::map<std::string, std::shared_ptr<RuntimeAttribute>>
            attribute; /// ���ӵ�������Ϣ���ں�Ȩ����Ϣ
    };

} // namespace kuiper_infer