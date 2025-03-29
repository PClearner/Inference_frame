#include "runtime/runtime_attr.hpp"

namespace star
{
    void RuntimeAttribute::ClearWeight()
    {
        if (!this->weight_data.empty())
        {
            std::vector<char> tmp = std::vector<char>();
            this->weight_data.swap(tmp);
        }
    }
}