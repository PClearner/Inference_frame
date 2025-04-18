#include "layer/detail/maxpooling.hpp"

namespace star
{
    MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w,
                                     uint32_t pooling_size_h, uint32_t pooling_size_w,
                                     uint32_t stride_h, uint32_t stride_w) : NonParamLayer("MaxPooling"),
                                                                             padding_h_(padding_h),
                                                                             padding_w_(padding_w),
                                                                             pooling_size_h_(pooling_size_h),
                                                                             pooling_size_w_(pooling_size_w),
                                                                             stride_h_(stride_h),
                                                                             stride_w_(stride_w) {}

    InferStatus MaxPoolingLayer::Forward(
        const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
        std::vector<std::shared_ptr<Tensor<float>>> &outputs)
    {

        if (inputs.empty())
        {
            LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
            return InferStatus::InferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size())
        {
            LOG(ERROR)
                << "The input and output tensor array size of the max pooling layer "
                   "do not match";
            return InferStatus::InferFailedInputOutSizeMatchError;
        }

        const uint32_t batch = inputs.size();
        const uint32_t pooling_h = this->pooling_size_h_;
        const uint32_t pooling_w = this->pooling_size_w_;
        if (!stride_h_ || !stride_w_)
        {
            LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                          "greater than 0";
            return InferStatus::InferFailedStrideParameterError;
        }

        for (uint32_t i = 0; i < batch; ++i)
        {
            const std::shared_ptr<ftensor> &input_data = inputs.at(i);
            if (input_data == nullptr || input_data->empty())
            {
                LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                              "empty tensor "
                           << i << "th";
                return InferStatus::InferFailedInputEmpty;
            }
            else
            {
                uint32_t input_h = input_data->rows();
                uint32_t input_w = input_data->cols();
                uint32_t output_h = uint32_t(std::floor(
                    (int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
                uint32_t output_w = uint32_t(std::floor(
                    (int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1));
                if (!output_w || !output_h)
                {
                    LOG(ERROR) << "The output size of tensor " << i << "th"
                               << " in the max pooling layer is less than zero";
                    return InferStatus::InferFailedOutputSizeError;
                }
                else
                {
                    const std::shared_ptr<ftensor> &output_data = outputs.at(i);
                    if (output_data != nullptr && !output_data->empty())
                    {
                        if (output_data->rows() != output_h ||
                            output_data->cols() != output_w)
                        {
                            LOG(ERROR) << "The output tensor array in the max pooling layer "
                                          "has an incorrectly sized tensor "
                                       << i << "th";
                            return InferStatus::InferFailedOutputSizeError;
                        }
                    }
                }
            }
        }

        for (uint32_t i = 0; i < batch; ++i)
        {
            const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
            CHECK(input_data == nullptr || !input_data->empty())
                << "The input tensor array in the max pooling layer has an "
                   "empty tensor "
                << i << "th";

            const uint32_t input_h = input_data->rows();
            const uint32_t input_w = input_data->cols();
            const uint32_t input_padded_h = input_data->rows() + 2 * padding_h_;
            const uint32_t input_padded_w = input_data->cols() + 2 * padding_w_;

            const uint32_t input_c = input_data->channels();

            const uint32_t output_h = uint32_t(
                std::floor((int(input_padded_h) - int(pooling_h)) / stride_h_ + 1));
            const uint32_t output_w = uint32_t(
                std::floor((int(input_padded_w) - int(pooling_w)) / stride_w_ + 1));

            std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
            if (output_data == nullptr || output_data->empty())
            {
                output_data =
                    std::make_shared<Tensor<float>>(input_c, output_h, output_w);
                outputs.at(i) = output_data;
            }

            CHECK(output_data->rows() == output_h && output_data->cols() == output_w &&
                  output_data->channels() == input_c)
                << "The output tensor array in the max pooling layer "
                   "has an incorrectly sized tensor "
                << i << "th";

            const auto &pooling_h = this->pooling_size_h_;
            const auto &pooling_w = this->pooling_size_w_;
            const auto &stride_h = this->stride_h_;
            const auto &stride_w = this->stride_w_;
            const auto &batch = inputs.size();

            const auto &input = input_data;
            auto &output = output_data;

            const uint32_t output_h = uint32_t(
                std::floor((int(input_padded_h) - int(pooling_h)) / stride_h_ + 1));
            const uint32_t output_w = uint32_t(
                std::floor((int(input_padded_w) - int(pooling_w)) / stride_w_ + 1));

            for (uint32_t c = 0; c < input->channels(); c++)
            {
                arma::fmat &input_matrix = input->slice(c);
                arma::fmat &output_matrix = output->slice(c);
                for (uint32_t r = 0; r < input_padded_w - pooling_w + 1; r += stride_w)
                {
                    int out_col = int(r / stride_w);
                    float *output_col_ptr = output_matrix.colptr(out_col);
                    for (uint32_t c = 0; c < input_padded_h - pooling_w + 1; c += stride_h)
                    {
                        int out_row = int(c / stride_h);
                        float max = std::numeric_limits<float>::lowest();
                        for (uint32_t w = 0; w < pooling_w; w++)
                        {
                            float *input_col_ptr = input_matrix.colptr(r + w - this->padding_w_);

                            for (uint32_t h = 0; h < pooling_h; h++)
                            {
                                float current_max;
                                if (c + h > this->padding_h_ && r + w > this->padding_w_ &&
                                    c + h < input_padded_h - this->padding_h_ && r + w < input_padded_w - this->padding_w_)
                                {
                                    current_max = *(input_col_ptr + c + h - this->padding_h_);
                                }
                                else
                                {
                                    current_max = std::numeric_limits<float>::lowest();
                                }
                                max = max > current_max ? max : current_max;
                            }
                        }
                        *(output_col_ptr + out_row) = max;
                    }
                }
            }
        }
    }

    ParseParameterAttrStatus MaxPoolingLayer::GetInstance(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &max_layer)
    {
        const auto params = op->params;

        if (params.find("padding") == params.end())
        {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::ParameterMissingPadding;
        }

        auto padding = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));

        if (!padding)
        {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::ParameterMissingPadding;
        }

        if (params.find("stride") == params.end())
        {
            LOG(ERROR) << "Can not find the stride parameter";
            return ParseParameterAttrStatus::ParameterMissingStride;
        }
        auto stride = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
        if (!stride)
        {
            LOG(ERROR) << "Can not find the stride parameter";
            return ParseParameterAttrStatus::ParameterMissingStride;
        }

        if (params.find("kernel_size") == params.end())
        {
            LOG(ERROR) << "Can not find the kernel size parameter";
            return ParseParameterAttrStatus::ParameterMissingKernel;
        }
        auto kernel_size = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("kernel_size"));
        if (!kernel_size)
        {
            LOG(ERROR) << "Can not find the kernel size parameter";
            return ParseParameterAttrStatus::ParameterMissingKernel;
        }

        const auto &padding_value = padding->value;
        const auto &stride_value = stride->value;
        const auto &kernel_value = kernel_size->value;

        const uint32_t dims = 2;
        if (padding_value.size() != dims)
        {
            LOG(ERROR) << "Can not find the right padding parameter";
            return ParseParameterAttrStatus::ParameterMissingPadding;
        }

        if (stride_value.size() != dims)
        {
            LOG(ERROR) << "Can not find the right stride parameter";
            return ParseParameterAttrStatus::ParameterMissingStride;
        }

        if (kernel_value.size() != dims)
        {
            LOG(ERROR) << "Can not find the right kernel size parameter";
            return ParseParameterAttrStatus::ParameterMissingKernel;
        }

        max_layer = std::make_shared<MaxPoolingLayer>(padding_value.at(0), padding_value.at(1),
                                                      kernel_value.at(0), kernel_value.at(1),
                                                      stride_value.at(0), stride_value.at(1));

        return ParseParameterAttrStatus::ParameterAttrParseSuccess;
    }

    LayerRegistererWrapper MaxPoolingwrapper("nn.MaxPooling", MaxPoolingLayer::GetInstance);

}