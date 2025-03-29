#include "Tensor.h"

int main()
{
    star::Tensor<float> tmp(3, 3, 3);
    star::Tensor<float> tmp3 = std::move(tmp);
    star::Tensor<float> tmp4 = std::move(tmp);
    star::Tensor<float> tmp5 = std::move(tmp);
    tmp.Ones();
    tmp.Show();
    std::cout << "=================================" << std::endl;

    std::vector<float> values(tmp.size());
    for (size_t i = 1; i <= values.size(); i++)
    {
        values[i - 1] = i;
    }
    tmp.Fill(values);
    tmp.Show();
    std::cout << "=================================" << std::endl;

    star::Tensor<float> tmp2 = std::move(tmp);
    tmp.Flatten();
    tmp.Show();
    std::cout << "=================================" << std::endl;

    std::vector<uint32_t> pads = {1, 2, 3, 4};
    tmp2.Padding(pads, 6);
    tmp2.Show();
    std::cout << "=================================" << std::endl;

    tmp3.Rand();
    tmp3.Show();
    std::cout << "=================================" << std::endl;

    tmp.Reshape(3, 9, 1);
    tmp.Show();
    std::cout << "=================================" << std::endl;

    return 0;
}