#pragma once
#include "runtime/ir.h"
#include "runtime/runtime_operand.hpp"
#include "runtime/runtime_op.hpp"
#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace star
{

    /// ����ͼ�ṹ���ɶ������ڵ�ͽڵ�֮���������ͼ���
    class RuntimeGraph
    {
    public:
        /**
         * ��ʼ������ͼ
         * @param param_path ����ͼ�Ľṹ�ļ�
         * @param bin_path ����ͼ�е�Ȩ���ļ�
         */
        RuntimeGraph(std::string param_path, std::string bin_path);

        /**
         * ����Ȩ���ļ�
         * @param bin_path Ȩ���ļ�·��
         */
        void set_bin_path(const std::string &bin_path);

        /**
         * ���ýṹ�ļ�
         * @param param_path  �ṹ�ļ�·��
         */
        void set_param_path(const std::string &param_path);

        /**
         * ���ؽṹ�ļ�
         * @return ���ؽṹ�ļ�
         */
        const std::string &param_path() const;

        /**
         * ����Ȩ���ļ�
         * @return ����Ȩ���ļ�
         */
        const std::string &bin_path() const;

        /**
         * ����ͼ�ĳ�ʼ��
         * @return �Ƿ��ʼ���ɹ�
         */
        bool Init();

        std::vector<std::shared_ptr<RuntimeOperator>> get_operators() const
        {
            return this->operators_;
        }

        std::map<std::string, std::shared_ptr<RuntimeOperator>> get_operators_maps_() const
        {
            return this->operators_maps_;
        }

        const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

    private:
        /**
         * ��ʼ��kuiper infer����ͼ�ڵ��е����������
         * @param inputs pnnx�е����������
         * @param runtime_operator ����ͼ�ڵ�
         */
        static void InitGraphOperatorsInput(
            const std::vector<pnnx::Operand *> &inputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * ��ʼ��kuiper infer����ͼ�ڵ��е����������
         * @param outputs pnnx�е����������
         * @param runtime_operator ����ͼ�ڵ�
         */
        static void InitGraphOperatorsOutput(
            const std::vector<pnnx::Operand *> &outputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * ��ʼ��kuiper infer����ͼ�еĽڵ�����
         * @param attrs pnnx�еĽڵ�����
         * @param runtime_operator ����ͼ�ڵ�
         */
        static void
        InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                       const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * ��ʼ��kuiper infer����ͼ�еĽڵ����
         * @param params pnnx�еĲ�������
         * @param runtime_operator ����ͼ�ڵ�
         */
        static void
        InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                        const std::shared_ptr<RuntimeOperator> &runtime_operator);

    private:
        std::string input_name_;  /// ����ͼ����ڵ������
        std::string output_name_; /// ����ͼ����ڵ������
        std::string param_path_;  /// ����ͼ�Ľṹ�ļ�
        std::string bin_path_;    /// ����ͼ��Ȩ���ļ�

        std::vector<std::shared_ptr<RuntimeOperator>> operators_;
        std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_maps_;

        std::unique_ptr<pnnx::Graph> graph_; /// pnnx��graph
    };
}