#include "tensor_util.hpp"

namespace star
{

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                                uint32_t cols)
    {
        return std::make_shared<Tensor<float>>(channels, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(
        const std::vector<uint32_t> &shapes)
    {
        CHECK(shapes.size() == 3);
        return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
    }

    std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor &tensor1,
                                                   const sftensor &tensor2)
    {
        CHECK(tensor1 != nullptr && tensor2 != nullptr);
        if (tensor1->shapes() == tensor2->shapes())
        {
            return {tensor1, tensor2};
        }
        else
        {
            CHECK(tensor1->channels() == tensor2->channels());
            if (tensor1->rows() == 1 && tensor1->cols() == 1)
            {
                sftensor new_tensor = TensorCreate(tensor1->channels(), tensor2->rows(), tensor2->cols());
                CHECK(tensor1->size() == tensor1->channels());
                for (uint32_t c = 0; c < tensor2->channels(); c++)
                {
                    new_tensor->slice(c).fill(tensor1->at(c, 0, 0));
                }
                return {new_tensor, tensor2};
            }
            else if (tensor2->rows() == 1 && tensor2->cols() == 1)
            {
                sftensor new_tensor = TensorCreate(tensor2->channels(), tensor1->rows(), tensor1->cols());
                CHECK(tensor2->size() == tensor2->channels());
                for (uint32_t c = 0; c < tensor1->channels(); c++)
                {
                    new_tensor->slice(c).fill(tensor2->at(c, 0, 0));
                }
                return {tensor1, new_tensor};
            }
            else
            {
                LOG(FATAL) << "Broadcast shape is not adapting!";
                return {tensor1, tensor2};
            }
        }
    }

    std::shared_ptr<Tensor<float>> TensorPadding(
        const std::shared_ptr<Tensor<float>> &tensor,
        const std::vector<uint32_t> &pads, float padding_value)
    {
        CHECK(tensor != nullptr && !tensor->empty());
        CHECK(pads.size() == 4);
        sftensor new_tensor = TensorClone(tensor);
        new_tensor->Padding(pads, padding_value);
        return new_tensor;
    }

    bool TensorIsSame(const std::shared_ptr<Tensor<float>> &a,
                      const std::shared_ptr<Tensor<float>> &b,
                      float threshold)
    {
        CHECK(a != nullptr);
        CHECK(b != nullptr);

        CHECK(a->shapes() == b->shapes());

        bool is_same = arma::approx_equal(a->data(), b->data(), "absdiff", threshold);
        return is_same;
    }

    std::shared_ptr<Tensor<float>> TensorElementAdd(
        const std::shared_ptr<Tensor<float>> &tensor1,
        const std::shared_ptr<Tensor<float>> &tensor2)
    {
        CHECK(tensor1 != nullptr && tensor2 != nullptr);
        sftensor new_tensor;
        if (tensor1->shapes() == tensor2->shapes())
        {
            new_tensor = TensorClone(tensor1);
            new_tensor->set_data(tensor1->data() + tensor2->data());
            return new_tensor;
        }
        else
        {
            const auto &[Btensor1, Btensor2] = TensorBroadcast(tensor1, tensor2);
            CHECK(Btensor1->shapes() == Btensor2->shapes());
            new_tensor = TensorClone(Btensor1);
            new_tensor->set_data(Btensor1->data() + Btensor2->data());
            return new_tensor;
        }
    }

    void TensorElementAdd(const std::shared_ptr<Tensor<float>> &tensor1,
                          const std::shared_ptr<Tensor<float>> &tensor2,
                          const std::shared_ptr<Tensor<float>> &output_tensor)
    {
        CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);

        if (tensor1->shapes() == tensor2->shapes())
        {
            CHECK(output_tensor->shapes() == tensor1->shapes());
            output_tensor->set_data(tensor1->data() + tensor2->data());
        }
        else
        {
            const auto &[Btensor1, Btensor2] = TensorBroadcast(tensor1, tensor2);
            CHECK(output_tensor->shapes() == Btensor1->shapes() && output_tensor->shapes() == Btensor2->shapes());
            output_tensor->set_data(Btensor1->data() + Btensor2->data());
        }
    }

    void TensorElementMultiply(const std::shared_ptr<Tensor<float>> &tensor1,
                               const std::shared_ptr<Tensor<float>> &tensor2,
                               const std::shared_ptr<Tensor<float>> &output_tensor)
    {
        CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
        if (tensor1->shapes() == tensor2->shapes())
        {
            CHECK(output_tensor->shapes() == tensor1->shapes());
            output_tensor->set_data(tensor1->data() % tensor2->data());
        }
        else
        {
            const auto &[Btensor1, Btensor2] = TensorBroadcast(tensor1, tensor2);
            CHECK(output_tensor->shapes() == Btensor1->shapes() && output_tensor->shapes() == Btensor2->shapes());
            output_tensor->set_data(Btensor1->data() % Btensor2->data());
        }
    }

    std::shared_ptr<Tensor<float>> TensorElementMultiply(
        const std::shared_ptr<Tensor<float>> &tensor1,
        const std::shared_ptr<Tensor<float>> &tensor2)
    {
        CHECK(tensor1 != nullptr && tensor2 != nullptr);
        sftensor new_tensor;
        if (tensor1->shapes() == tensor2->shapes())
        {
            new_tensor = TensorClone(tensor1);
            new_tensor->set_data(tensor1->data() % tensor2->data());
            return new_tensor;
        }
        else
        {
            const auto &[Btensor1, Btensor2] = TensorBroadcast(tensor1, tensor2);
            CHECK(Btensor1->shapes() == Btensor2->shapes());
            new_tensor = TensorClone(Btensor1);
            new_tensor->set_data(Btensor1->data() % Btensor2->data());
            return new_tensor;
        }
    }

    std::shared_ptr<Tensor<float>> TensorClone(
        std::shared_ptr<Tensor<float>> tensor)
    {
        sftensor tmp = std::move(tensor);
        return tmp;
    }

    // std::shared_ptr<Tensor<float>> TensorClone(
    //     std::shared_ptr<Tensor<float>> tensor)
    // {
    //       return std::make_shared<Tensor<float>>(*tensor);
    // }
}
