#pragma once

namespace star
{

    enum class RuntimeParameterType
    {
        ParameterUnknown = 0,
        ParameterBool = 1,
        ParameterInt = 2,

        ParameterFloat = 3,
        ParameterString = 4,
        ParameterIntArray = 5,
        ParameterFloatArray = 6,
        ParameterStringArray = 7,
    };

    enum class InferStatus
    {
        InferUnknown = -1,
        InferSuccess = 0,

        InferFailedInputEmpty = 1,
        InferFailedWeightParameterError = 2,
        InferFailedBiasParameterError = 3,
        InferFailedStrideParameterError = 4,
        InferFailedDimensionParameterError = 5,
        InferFailedInputOutSizeMatchError = 6,

        InferFailedOutputSizeError = 7,
        InferFailedShapeParameterError = 9,
        InferFailedChannelParameterError = 10,
        InferFailedOutputEmpty = 11,

    };

    enum class ParseParameterAttrStatus
    {
        ParameterMissingUnknown = -1,
        ParameterMissingStride = 1,
        ParameterMissingPadding = 2,
        ParameterMissingKernel = 3,
        ParameterMissingUseBias = 4,
        ParameterMissingInChannel = 5,
        ParameterMissingOutChannel = 6,

        ParameterMissingEps = 7,
        ParameterMissingNumFeatures = 8,
        ParameterMissingDim = 9,
        ParameterMissingExpr = 10,
        ParameterMissingOutHW = 11,
        ParameterMissingShape = 12,
        ParameterMissingGroups = 13,
        ParameterMissingScale = 14,
        ParameterMissingResizeMode = 15,
        ParameterMissingDilation = 16,
        ParameterMissingPaddingMode = 16,

        AttrMissingBias = 21,
        AttrMissingWeight = 22,
        AttrMissingRunningMean = 23,
        AttrMissingRunningVar = 24,
        AttrMissingOutFeatures = 25,
        AttrMissingYoloStrides = 26,
        AttrMissingYoloAnchorGrides = 27,
        AttrMissingYoloGrides = 28,

        ParameterAttrParseSuccess = 0
    };
} //