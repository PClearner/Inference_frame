#include "layer/detail/expr.hpp"
using namespace star;

void tokenizer()
{

    const std::string &str = "add(@0,mul(@1,@2))";
    ExpressionParser parser(str);
    parser.Tokenizer();
    const auto &tokens = parser.tokens();
    CHECK_EQ(tokens.empty(), false);

    const auto &token_strs = parser.token_strs();
    CHECK_EQ(token_strs.at(0), "add");
    CHECK_EQ(int(tokens.at(0).token_type), int(TokenType::TokenAdd));

    CHECK_EQ(token_strs.at(1), "(");
    CHECK_EQ(int(tokens.at(1).token_type), int(TokenType::TokenLeftBracket));

    CHECK_EQ(token_strs.at(2), "@0");
    CHECK_EQ(int(tokens.at(2).token_type), int(TokenType::TokenInputNumber));

    CHECK_EQ(token_strs.at(3), ",");
    CHECK_EQ(int(tokens.at(3).token_type), int(TokenType::TokenComma));

    CHECK_EQ(token_strs.at(4), "mul");
    CHECK_EQ(int(tokens.at(4).token_type), int(TokenType::TokenMul));

    CHECK_EQ(token_strs.at(5), "(");
    CHECK_EQ(int(tokens.at(5).token_type), int(TokenType::TokenLeftBracket));

    CHECK_EQ(token_strs.at(6), "@1");
    CHECK_EQ(int(tokens.at(6).token_type), int(TokenType::TokenInputNumber));

    CHECK_EQ(token_strs.at(7), ",");
    CHECK_EQ(int(tokens.at(7).token_type), int(TokenType::TokenComma));

    CHECK_EQ(token_strs.at(8), "@2");
    CHECK_EQ(int(tokens.at(8).token_type), int(TokenType::TokenInputNumber));

    CHECK_EQ(token_strs.at(9), ")");
    CHECK_EQ(int(tokens.at(9).token_type), int(TokenType::TokenRightBracket));

    CHECK_EQ(token_strs.at(10), ")");
    CHECK_EQ(int(tokens.at(10).token_type), int(TokenType::TokenRightBracket));
}

void treetest()
{
    std::string str = "add(@0,@1)";
    ExpressionParser parser1(str);
    parser1.Tokenizer();
    int index = 0; // 从0位置开始构建语法树
    // 抽象语法树:
    //
    //    add
    //    /  \
  //  @0    @1

    std::string token_s1 = "";
    for (const auto &i : parser1.token_strs())
    {
        token_s1 += i + " ";
    }
    LOG(INFO) << token_s1;
    const auto &node1 = parser1.Generatebyrecursion(index);
    CHECK_EQ(int(node1->num_index), int(TokenType::TokenAdd));
    CHECK_EQ(node1->left->num_index, 0);
    CHECK_EQ(node1->right->num_index, 1);

    str = "add(mul(@0,@1),@2)";
    ExpressionParser parser2(str);
    parser2.Tokenizer();
    index = 0;
    // 从0位置开始构建语法树
    // 抽象语法树:
    //
    //       add
    //       /  \
    //     mul   @2
    //    /   \
    //  @0    @1
    std::string token_s2 = "";
    for (const auto &i : parser2.token_strs())
    {
        token_s2 += i + " ";
    }
    LOG(INFO) << token_s2;
    const auto &node2 = parser2.Generatebyrecursion(index);
    CHECK_EQ(int(node2->num_index), int(TokenType::TokenAdd));
    CHECK_EQ(int(node2->left->num_index), int(TokenType::TokenMul));
    CHECK_EQ(node2->left->left->num_index, 0);
    CHECK_EQ(node2->left->right->num_index, 1);
    CHECK_EQ(node2->right->num_index, 2);
}

void polish()
{
    const std::string &str = "add(mul(@0,@1),@2)";
    ExpressionParser parser(str);
    parser.Tokenizer();
    // 抽象语法树:
    //
    //       add
    //       /  \
    //     mul   @2
    //    /   \
    //  @0    @1

    const auto &vec = parser.Generate();
    for (const auto &item : vec)
    {
        if (item->num_index == -5)
        {
            LOG(INFO) << "Mul";
        }
        else if (item->num_index == -6)
        {
            LOG(INFO) << "Add";
        }
        else
        {
            LOG(INFO) << item->num_index;
        }
    }
}

void complex()
{
    const std::string &str = "mul(@2,add(@0,@1))";
    ExpressionLayer layer(str);
    LOG(INFO) << "layer initiatize over";
    std::shared_ptr<Tensor<float>> input1 =
        std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(2.f);
    std::shared_ptr<Tensor<float>> input2 =
        std::make_shared<Tensor<float>>(3, 224, 224);
    input2->Fill(3.f);

    std::shared_ptr<Tensor<float>> input3 =
        std::make_shared<Tensor<float>>(3, 224, 224);
    input3->Fill(4.f);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);

    layer.get_order().push_back(0);
    layer.get_order().push_back(1);
    layer.get_order().push_back(2);

    std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
    outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
    LOG(INFO) << "layer forward start";
    const auto status = layer.Forward(inputs, outputs);
    LOG(INFO) << "layer forward over";
    CHECK_EQ(int(status), int(InferStatus::InferSuccess));
    CHECK_EQ(outputs.size(), 1);
    std::shared_ptr<Tensor<float>> output2 =
        std::make_shared<Tensor<float>>(3, 224, 224);
    output2->Fill(20.f);
    std::shared_ptr<Tensor<float>> output1 = outputs.front();

    // LOG(INFO) << "output1->data()" << output1->data();
    CHECK(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

void tokenizer_sin()
{
    const std::string &str = "add(sin(@0),@1)";
    ExpressionParser parser(str);
    parser.Tokenizer();
    const auto &tokens = parser.tokens();
    CHECK_EQ(tokens.empty(), false);

    const auto &token_strs = parser.token_strs();

    std::string token_s2 = "";
    for (const auto &i : parser.token_strs())
    {
        token_s2 += i + " ";
    }
    LOG(INFO) << token_s2;

    CHECK_EQ(token_strs.at(0), "add");
    CHECK_EQ(int(tokens.at(0).token_type), int(TokenType::TokenAdd));

    CHECK_EQ(token_strs.at(1), "(");
    CHECK_EQ(int(tokens.at(1).token_type), int(TokenType::TokenLeftBracket));

    CHECK_EQ(token_strs.at(2), "sin");
    CHECK_EQ(int(tokens.at(2).token_type), int(TokenType::TokenSin));

    CHECK_EQ(token_strs.at(3), "(");
    CHECK_EQ(int(tokens.at(3).token_type), int(TokenType::TokenLeftBracket));

    CHECK_EQ(token_strs.at(4), "@0");
    CHECK_EQ(int(tokens.at(4).token_type), int(TokenType::TokenInputNumber));

    CHECK_EQ(token_strs.at(5), ")");
    CHECK_EQ(int(tokens.at(5).token_type), int(TokenType::TokenRightBracket));

    CHECK_EQ(token_strs.at(6), ",");
    CHECK_EQ(int(tokens.at(6).token_type), int(TokenType::TokenComma));

    CHECK_EQ(token_strs.at(7), "@1");
    CHECK_EQ(int(tokens.at(7).token_type), int(TokenType::TokenInputNumber));

    CHECK_EQ(token_strs.at(8), ")");
    CHECK_EQ(int(tokens.at(8).token_type), int(TokenType::TokenRightBracket));
}

void generate_sin()
{
    const std::string &str = "add(sin(@0),@1)";

    int index = 0;
    /**
          add
          /   \
        sin    @1
         |
        @0
     */
    ExpressionParser parser(str);
    parser.Tokenizer(true);
    const auto &node = parser.Generatebyrecursion(index);
    CHECK_EQ(node->num_index, int(TokenType::TokenAdd));
    CHECK_EQ(node->left->num_index, int(TokenType::TokenSin));
    CHECK_EQ(node->left->left->num_index, 0);
    CHECK_EQ(node->right->num_index, 1);
}

void complex2()
{
    const std::string &str = "mul(@1,sin(@0))";
    ExpressionLayer layer(str);
    std::shared_ptr<Tensor<float>> input1 =
        std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(2.f); // @0
    std::shared_ptr<Tensor<float>> input2 =
        std::make_shared<Tensor<float>>(3, 224, 224);
    input2->Fill(3.f); //@1

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);

    layer.get_order().push_back(0);
    layer.get_order().push_back(1);

    std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
    outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
    LOG(INFO) << "layer forward start";
    const auto status = layer.Forward(inputs, outputs);
    LOG(INFO) << "layer forward over";
    CHECK_EQ(int(status), int(InferStatus::InferSuccess));
    CHECK_EQ(outputs.size(), 1);

    float val = 2.f;
    float res = std::sin(val) * 3.f;
    std::shared_ptr<Tensor<float>> output2 =
        std::make_shared<Tensor<float>>(3, 224, 224);
    output2->Fill(res);
    std::shared_ptr<Tensor<float>> output1 = outputs.front();
    CHECK(arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

int main()
{
    LOG(INFO) << "TOKENIZER========================================================================================";
    // tokenizer();
    LOG(INFO) << "TESTTREE========================================================================================";
    treetest();
    LOG(INFO) << "POLISH========================================================================================";
    polish();
    LOG(INFO) << "COMPLEX========================================================================================";
    complex();
    LOG(INFO) << "TOKENIZER_SIN========================================================================================";
    tokenizer_sin();
    LOG(INFO) << "GENERATE_SIN========================================================================================";
    generate_sin();
    LOG(INFO) << "COMPLEX2========================================================================================";
    complex2();
    return 0;
}