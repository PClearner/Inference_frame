#pragma once

namespace star
{
    enum class RuntimeDataType
    {
        kTypeUnknown = 0,
        kTypeFloat32 = 1,
        kTypeFloat64 = 2,
        kTypeFloat16 = 3,
        kTypeInt32 = 4,
        kTypeInt64 = 5,
        kTypeInt16 = 6,
        kTypeInt8 = 7,
        kTypeUInt8 = 8,
    };
}
