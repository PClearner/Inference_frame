#include "layer/detail/convolution.hpp"

namespace star
{
    ConvolutionLayer::ConvolutionLayer(uint32_t output_channel, uint32_t in_channel,
                                       uint32_t kernel_h, uint32_t kernel_w,
                                       uint32_t padding_h, uint32_t padding_w,
                                       uint32_t stride_h, uint32_t stride_w,
                                       uint32_t groups, bool use_bias) : ParamLayer("Convolution"),
                                                                         use_bias_(use_bias),
                                                                         groups_(groups),
                                                                         padding_h_(padding_h),
                                                                         padding_w_(padding_w),
                                                                         stride_h_(stride_h),
                                                                         stride_w_(stride_w)
    {
        if (groups != 1)
        {
            in_channel /= groups;
        }
        this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
        if (use_bias_)
        {
            this->InitBiasParam(output_channel, 1, 1, 1);
        }
    }

    ParseParameterAttrStatus ConvolutionLayer::GetInstance(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &conv_layer)
    {

        CHECK(op != nullptr) << "Convolution operator is nullptr";
        const std::map<std::string, std::shared_ptr<RuntimeParameter>> &params =
            op->params;

        if (params.find("dilation") == params.end())
        {
            LOG(ERROR) << "Can not find the dilation parameter";
            return ParseParameterAttrStatus::ParameterMissingDilation;
        }

        auto dilation_param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
            params.at("dilation"));

        if (dilation_param == nullptr || dilation_param->value.size() != 2)
        {
            LOG(ERROR) << "Can not find the dilation parameter";
            return ParseParameterAttrStatus::ParameterMissingDilation;
        }

        CHECK(dilation_param->value.at(0) != 1 || dilation_param->value.at(1))
            << "Only support dilation value equals to one!";

        if (params.find("in_channels") == params.end())
        {
            LOG(ERROR) << "Can not find the in channel parameter";
            return ParseParameterAttrStatus::ParameterMissingInChannel;
        }
        auto in_channel =
            std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("in_channels"));
        if (!in_channel)
        {
            LOG(ERROR) << "Can not find the in channel parameter";
            return ParseParameterAttrStatus::ParameterMissingInChannel;
        }

        if (params.find("out_channels") == params.end())
        {
            LOG(ERROR) << "Can not find the out channel parameter";
            return ParseParameterAttrStatus::ParameterMissingOutChannel;
        }

        auto out_channel =
            std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("out_channels"));
        if (!out_channel)
        {
            LOG(ERROR) << "Can not find the out channel parameter";
            return ParseParameterAttrStatus::ParameterMissingOutChannel;
        }

        if (params.find("padding") == params.end())
        {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::ParameterMissingPadding;
        }

        auto padding =
            std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
        if (!padding)
        {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::ParameterMissingPadding;
        }

        if (params.find("bias") == params.end())
        {
            LOG(ERROR) << "Can not find the bias parameter";
            return ParseParameterAttrStatus::ParameterMissingUseBias;
        }
        auto use_bias =
            std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
        if (!use_bias)
        {
            LOG(ERROR) << "Can not find the bias parameter";
            return ParseParameterAttrStatus::ParameterMissingUseBias;
        }

        if (params.find("stride") == params.end())
        {
            LOG(ERROR) << "Can not find the stride parameter";
            return ParseParameterAttrStatus::ParameterMissingStride;
        }
        auto stride =
            std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
        if (!stride)
        {
            LOG(ERROR) << "Can not find the stride parameter";
            return ParseParameterAttrStatus::ParameterMissingStride;
        }

        if (params.find("kernel_size") == params.end())
        {
            LOG(ERROR) << "Can not find the kernel parameter";
            return ParseParameterAttrStatus::ParameterMissingKernel;
        }
        auto kernel = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
            params.at("kernel_size"));
        if (!kernel)
        {
            LOG(ERROR) << "Can not find the kernel parameter";
            return ParseParameterAttrStatus::ParameterMissingKernel;
        }

        if (params.find("padding_mode") != params.end())
        {
            auto padding_mode = std::dynamic_pointer_cast<RuntimeParameterString>(
                params.at("padding_mode"));
            if (padding_mode == nullptr)
            {
                LOG(ERROR) << "Can not find the padding parameter";
                return ParseParameterAttrStatus::ParameterMissingPaddingMode;
            }
            else
            {
                const std::string &padding_mode_str = padding_mode->value;
                if (padding_mode_str != "zeros")
                {
                    LOG(ERROR) << "Padding mode unsupported: " << padding_mode_str;
                    return ParseParameterAttrStatus::ParameterMissingPaddingMode;
                }
            }
        }
        else
        {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParameterAttrStatus::ParameterMissingPaddingMode;
        }

        auto groups =
            std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("groups"));
        if (!groups)
        {
            LOG(ERROR) << "Can not find the groups parameter";
            return ParseParameterAttrStatus::ParameterMissingGroups;
        }

        const uint32_t dims = 2;
        const std::vector<int> &kernels = kernel->value;
        const std::vector<int> &paddings = padding->value;
        const std::vector<int> &strides = stride->value;
        if (paddings.size() != dims)
        {
            LOG(ERROR) << "Can not find the right padding parameter";
            return ParseParameterAttrStatus::ParameterMissingPadding;
        }

        if (strides.size() != dims)
        {
            LOG(ERROR) << "Can not find the right stride parameter";
            return ParseParameterAttrStatus::ParameterMissingStride;
        }

        if (kernels.size() != dims)
        {
            LOG(ERROR) << "Can not find the right kernel size parameter";
            return ParseParameterAttrStatus::ParameterMissingKernel;
        }

        conv_layer = std::make_shared<ConvolutionLayer>(out_channel->value, in_channel->value, kernels.at(0), kernels.at(1),
                                                        paddings.at(0), paddings.at(1), strides.at(0), strides.at(1),
                                                        groups->value, use_bias->value);

        const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &attrs = op->attribute;
        if (use_bias->value)
        {
            if (attrs.find("bias") == attrs.end())
            {
                LOG(ERROR) << "Can not find the bias attribute";
                return ParseParameterAttrStatus::AttrMissingBias;
            }
            const auto &bias = attrs.at("bias");
            const std::vector<int> &bias_shape = bias->shape;
            if (bias_shape.empty() || bias_shape.at(0) != out_channel->value)
            {
                LOG(ERROR) << "The attribute of bias shape is wrong";
                return ParseParameterAttrStatus::AttrMissingBias;
            }

            const std::vector<float> &bias_values = bias->get<float>();
            conv_layer->set_bias(bias_values);
        }

        if (attrs.find("weight") == attrs.end())
        {
            LOG(ERROR) << "Can not find the weight attribute";
            return ParseParameterAttrStatus::AttrMissingWeight;
        }

        const auto &weight = attrs.at("weight");
        const std::vector<int> &weight_shape = weight->shape;
        if (weight_shape.empty())
        {
            LOG(ERROR) << "The attribute of weight shape is wrong";
            return ParseParameterAttrStatus::AttrMissingWeight;
        }

        const std::vector<float> &weight_values = weight->get<float>();
        conv_layer->set_weights(weight_values);

        auto conv_layer_derived =
            std::dynamic_pointer_cast<ConvolutionLayer>(conv_layer);
        CHECK(conv_layer_derived != nullptr);
        conv_layer_derived->InitIm2ColWeight();
        return ParseParameterAttrStatus::ParameterAttrParseSuccess;
    }

    InferStatus ConvolutionLayer::Forward(
        const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
        std::vector<std::shared_ptr<Tensor<float>>> &outputs)
    {

        if (inputs.empty())
        {
            LOG(ERROR) << "The input tensor array in the convolution layer is empty";
            return InferStatus::InferFailedInputEmpty;
        }

        if (inputs.size() != outputs.size())
        {
            LOG(ERROR) << "The input and output tensor array size of the convolution "
                          "layer do not match";
            return InferStatus::InferFailedInputOutSizeMatchError;
        }

        if (weights_.empty())
        {
            LOG(ERROR) << "The number of kernel matrix in the convolution layer should "
                          "be greater than zero";
            return InferStatus::InferFailedWeightParameterError;
        }

        if (this->use_bias_ && this->bias_.size() != this->weights_.size())
        {
            LOG(ERROR) << "The number of kernel matrix and bias matrix do not match";
            return InferStatus::InferFailedBiasParameterError;
        }

        if (!stride_h_ || !stride_w_)
        {
            LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                          "greater than 0";
            return InferStatus::InferFailedStrideParameterError;
        }

        const uint32_t kernel_count = this->weights_.size();
        const uint32_t kernel_h = this->weights_.at(0)->rows();
        const uint32_t kernel_w = this->weights_.at(0)->cols();
        const uint32_t kernel_c = this->weights_.at(0)->channels();
        const uint32_t row_len = kernel_h * kernel_w;
        CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
            << "The size of kernel matrix in the convolution layer should be greater "
               "than zero";

        for (uint32_t k = 0; k < kernel_count; ++k)
        {
            const std::shared_ptr<Tensor<float>> &kernel = this->weights_.at(k);
            CHECK(kernel->rows() == kernel_h);
            CHECK(kernel->cols() == kernel_w);
            CHECK(kernel->channels() == kernel_c);
        }
        const uint32_t kernel_count_group = kernel_count / groups_;
        const uint32_t batch_size = inputs.size();

        if (kernel_matrix_arr_.empty())
        {
            this->InitIm2ColWeight();
        }
        else
        {
            if (groups_ == 1)
            {
                CHECK(kernel_matrix_arr_.size() == kernel_count_group)
                    << "The number of kernel matrix and kernel_count_group do not match";
            }
            else
            {
                CHECK(kernel_matrix_arr_.size() == kernel_count)
                    << "The number of kernel matrix and kernel_count do not match";
            }
        }

        for (uint32_t i = 0; i < batch_size; ++i)
        {
            const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
            CHECK(input != nullptr && !input->empty())
                << "The input tensor array in the convolution layer has an empty  "
                   "tensor "
                << i << " th";

            const uint32_t input_c = input->channels();
            const uint32_t input_padded_h = input->rows() + 2 * padding_h_;
            const uint32_t input_padded_w = input->cols() + 2 * padding_w_;

            const uint32_t output_h =
                std::floor((int(input_padded_h) - int(kernel_h)) / stride_h_ + 1);
            const uint32_t output_w =
                std::floor((int(input_padded_w) - int(kernel_w)) / stride_w_ + 1);
            CHECK(output_h > 0 && output_w > 0)
                << "The size of the output tensor should be greater than zero " << i
                << " th";

            if (groups_ != 1)
            {
                CHECK(kernel_count % groups_ == 0);
                CHECK(input_c % groups_ == 0);
            }

            uint32_t col_len = output_h * output_w;
            CHECK(col_len > 0) << "Output_h x output_w for the convolution layer "
                                  "should be greater than zero "
                               << i << " th";

            uint32_t input_c_group = input_c / groups_;
            CHECK(input_c_group == kernel_c) << "The number of channel for the kernel "
                                                "matrix and input tensor do not match";

            for (uint32_t g = 0; g < this->groups_; g++)
            {
                const auto &input_matrix = Im2Col(input, kernel_w, kernel_h, input->cols(),
                                                  input->rows(), input_c_group, g, row_len, col_len);

                std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
                if (output_tensor == nullptr || output_tensor->empty())
                {
                    output_tensor =
                        std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
                    outputs.at(i) = output_tensor;
                }

                CHECK(output_tensor->rows() == output_h &&
                      output_tensor->cols() == output_w &&
                      output_tensor->channels() == kernel_count)
                    << "The output tensor array in the convolution layer has an "
                       "incorrectly sized tensor "
                    << i << "th";

                for (uint32_t k = 0; k < kernel_count_group; k++)
                {
                    const auto &kernel_arr = this->kernel_matrix_arr_.at(g * kernel_count_group + k);
                    ConvGemmBias(input_matrix, output_tensor, g, k, kernel_count_group, kernel_arr, output_w, output_h);
                }
            }
        }

        return InferStatus::InferSuccess;
    }

    /**
     * 初始化kernel的im2col排布
     */
    void ConvolutionLayer::InitIm2ColWeight()
    {
        const uint32_t kernel_count = this->weights_.size();
        CHECK(kernel_count > 0) << "kernel count must greater than zero";
        const uint32_t kernel_h = this->weights_.at(0)->rows();
        const uint32_t kernel_w = this->weights_.at(0)->cols();
        const uint32_t kernel_c = this->weights_.at(0)->channels();
        const uint32_t row_len = kernel_h * kernel_w;
        CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
            << "The size of kernel matrix should be greater than zero";

        std::vector<arma::frowvec> kernel_matrix_arr;
        for (uint32_t k = 0; k < kernel_count; ++k)
        {
            const std::shared_ptr<Tensor<float>> &kernel = this->weights_.at(k);
            CHECK(kernel->rows() == kernel_h);
            CHECK(kernel->cols() == kernel_w);
            CHECK(kernel->channels() == kernel_c);

            arma::frowvec kernel_matrix_c(kernel->size());
            for (uint32_t ic = 0; ic < kernel->channels(); ic++)
            {
                memcpy(kernel_matrix_c.memptr() + ic * row_len, kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
            }
            kernel_matrix_arr.push_back(kernel_matrix_c);
        }

        this->kernel_matrix_arr_ = std::move(kernel_matrix_arr);
    }

    void ConvolutionLayer::ConvGemmBias(const arma::fmat &input_matrix, sftensor output_tensor,
                                        uint32_t group, uint32_t kernel_index,
                                        uint32_t kernel_count_group, const arma::frowvec &kernel,
                                        uint32_t output_w, uint32_t output_h) const
    {
        arma::fmat output(output_tensor->matrix_raw_ptr(kernel_index + group * kernel_count_group),
                          output_h, output_w, false, true);

        CHECK(output.size() == output_h * output_w)
            << "Output_h x output_w for the convolution layer "
               "should be output tensor size";
        if (!this->bias_.empty() && this->use_bias_)
        {
            std::shared_ptr<Tensor<float>> bias;
            bias = this->bias_.at(kernel_index);
            if (bias != nullptr && !bias->empty())
            {
                float bias_value = bias->index(0);
                output = kernel * input_matrix + bias_value;
            }
            else
            {
                LOG(FATAL) << "Bias tensor is empty or nullptr";
            }
        }
        else
        {
            output = kernel * input_matrix;
        }
    }

    arma::fmat ConvolutionLayer::Im2Col(sftensor input, uint32_t kernel_w, uint32_t kernel_h,
                                        uint32_t input_w, uint32_t input_h, uint32_t input_c_group,
                                        uint32_t group, uint32_t row_len, uint32_t col_len) const
    {
        arma::fmat input_matrix(input_c_group * row_len, col_len);
        const uint32_t input_padded_h = input_h + 2 * padding_h_;
        const uint32_t input_padded_w = input_w + 2 * padding_w_;
        const float padding_value = 0.f;

        for (uint32_t ic = 0; ic < input_c_group; ic++)
        {

            float *input_slice = input->matrix_raw_ptr(ic + group * input_c_group);
            uint32_t channel_row = ic * row_len;
            uint32_t current_col = 0;
            for (uint32_t c = 0; c < input_padded_w - kernel_w + 1; c += this->stride_w_)
            {
                for (uint32_t r = 0; r < input_padded_h - kernel_h + 1; r += this->stride_h_)
                {
                    auto input_col_ptr = input_matrix.colptr(current_col) + channel_row;
                    current_col++;
                    for (uint32_t w = 0; w < kernel_w; w++)
                    {
                        const uint32_t region_w = input_h * (c + w - padding_w_);
                        for (uint32_t h = 0; h < kernel_h; h++)
                        {
                            if (w + c >= this->padding_w_ && w + c < input_w + this->padding_w_ &&
                                h + r >= this->padding_h_ && h + r < input_h + this->padding_h_)
                            {
                                *(input_col_ptr) = *(input_slice + region_w + r + h - padding_h_);
                            }
                            else
                            {
                                *(input_col_ptr) = padding_value;
                            }
                            input_col_ptr += 1;
                        }
                    }
                }
            }
        }

        return input_matrix;
    }
}