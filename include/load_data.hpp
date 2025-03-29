#pragma once
#include <armadillo>
#include <string>
#include <glog/logging.h>
#include <fstream>
#include <utility>
namespace star
{

    class CSVDataLoader
    {
    public:
        /**
         * ��csv�ļ��г�ʼ������
         * @param file_path csv�ļ���·��
         * @param split_char �ָ�����
         * @return ����csv�ļ��õ�������
         */
        static arma::fmat LoadData(const std::string &file_path, char split_char = ',');

    private:
        /**
         * �õ�csv�ļ��ĳߴ��С��LoadData�и������ﷵ�صĳߴ��С��ʼ�����ص�fmat
         * @param file csv�ļ���·��
         * @param split_char �ָ����
         * @return ����csv�ļ��ĳߴ��С
         */
        static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
    };
}