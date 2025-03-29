#pragma once

#include <glog/logging.h>
#include <vector>
#include "runtime/runtime_datatype.hpp"
#include "status_code.hpp"

namespace star
{

    class RuntimeAttribute
    {
    public:
        std::vector<char> weight_data;
        std::vector<int> shape;

        RuntimeDataType type = RuntimeDataType::kTypeUnknown;

        template <class T>
        std::vector<T> get(bool need_clear_weight = true);

        void ClearWeight();
    };

    template <class T>
    std::vector<T> RuntimeAttribute::get(bool need_clear_weight)
    {
        CHECK(!weight_data.empty());
        CHECK(type != RuntimeDataType::kTypeUnknown);
        std::vector<T> weight;
        switch (type)
        {

        case RuntimeDataType::kTypeFloat32:
        {
            const bool is_float = std::is_same<T, float>::value;
            CHECK_EQ(is_float, true);
            uint16_t float_size = sizeof(float);
            CHECK(weight_data.size() % float_size == 0);
            for (uint32_t index = 0; index < weight_data.size() / float_size; index++)
            {
                float w = *((float *)weight_data.data() + index);
                weight.push_back(w);
            }
            break;
        }
        default:
        {
            LOG(FATAL) << "Unknown weight data type: " << int(type);
        }
        }

        if (need_clear_weight)
        {
            this->ClearWeight();
        }
    }

}
