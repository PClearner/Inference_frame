#include "layer/parse/parse_expression.hpp"
#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>
#include <stack>

namespace star
{
    void ExpressionParser::Tokenizer(bool retokenize)
    {
        // check

        if (!retokenize && !this->tokens_.empty())
        {
            return;
        }

        CHECK(!statement_.empty()) << "The input statement is empty!";
        statement_.erase(std::remove_if(statement_.begin(), statement_.end(),
                                        [](char c)
                                        { return std::isspace(c); }),
                         statement_.end());
        CHECK(!statement_.empty()) << "The input statement is empty!";

        for (int32_t i = 0; i < statement_.size();)
        {
            char c = statement_.at(i);
            if (c == 'a')
            {
                CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'd')
                    << "Parse add token failed, illegal character: "
                    << statement_.at(i + 1);
                CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'd')
                    << "Parse add token failed, illegal character: "
                    << statement_.at(i + 2);
                Token token(TokenType::TokenAdd, i, i + 3);
                tokens_.push_back(token);
                std::string token_operation =
                    std::string(statement_.begin() + i, statement_.begin() + i + 3);
                token_strs_.push_back(token_operation);
                i = i + 3;
            }
            else if (c == 's')
            {
                CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'i')
                    << "Parse add token failed, illegal character: "
                    << statement_.at(i + 1);
                CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'n')
                    << "Parse add token failed, illegal character: "
                    << statement_.at(i + 2);
                Token input(TokenType::TokenSin, i, i + 3);
                this->tokens_.push_back(input);
                std::string token_left_bracket = std::string(statement_.begin() + i, statement_.begin() + i + 3);
                token_strs_.push_back(token_left_bracket);
                i = i + 3;
            }
            else if (c == 'm')
            {
                CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u')
                    << "Parse multiply token failed, illegal character: "
                    << statement_.at(i + 1);
                CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'l')
                    << "Parse multiply token failed, illegal character: "
                    << statement_.at(i + 2);
                Token token(TokenType::TokenMul, i, i + 3);
                tokens_.push_back(token);
                std::string token_operation =
                    std::string(statement_.begin() + i, statement_.begin() + i + 3);
                token_strs_.push_back(token_operation);
                i = i + 3;
            }
            else if (c == '@')
            {
                CHECK(i + 1 < statement_.size() && std::isdigit(statement_.at(i + 1)))
                    << "Parse number token failed, illegal character: "
                    << statement_.at(i + 1);
                int32_t j = i + 1;
                for (; j < statement_.size(); ++j)
                {
                    if (!std::isdigit(statement_.at(j)))
                    {
                        break;
                    }
                }
                Token token(TokenType::TokenInputNumber, i, j);
                CHECK(token.start_pos < token.end_pos);
                tokens_.push_back(token);
                std::string token_input_number =
                    std::string(statement_.begin() + i, statement_.begin() + j);
                token_strs_.push_back(token_input_number);
                i = j;
            }
            else if (c == ',')
            {
                Token token(TokenType::TokenComma, i, i + 1);
                tokens_.push_back(token);
                std::string token_comma =
                    std::string(statement_.begin() + i, statement_.begin() + i + 1);
                token_strs_.push_back(token_comma);
                i += 1;
            }
            else if (c == '(')
            {
                Token token(TokenType::TokenLeftBracket, i, i + 1);
                tokens_.push_back(token);
                std::string token_left_bracket =
                    std::string(statement_.begin() + i, statement_.begin() + i + 1);
                token_strs_.push_back(token_left_bracket);
                i += 1;
            }
            else if (c == ')')
            {
                Token token(TokenType::TokenRightBracket, i, i + 1);
                tokens_.push_back(token);
                std::string token_right_bracket =
                    std::string(statement_.begin() + i, statement_.begin() + i + 1);
                token_strs_.push_back(token_right_bracket);
                i += 1;
            }
            else
            {
                LOG(FATAL) << "Unknown  illegal character: " << c;
            }
        }
    }

    const std::vector<Token> &ExpressionParser::tokens() const
    {
        return this->tokens_;
    }

    const std::vector<std::string> &ExpressionParser::token_strs() const
    {
        return this->token_strs_;
    }

    std::shared_ptr<TokenNode> ExpressionParser::Generatebyrecursion(int32_t &index)
    {
        CHECK(index < this->tokens_.size());
        const auto current_token = this->tokens_.at(index);
        CHECK(current_token.token_type == TokenType::TokenInputNumber ||
              current_token.token_type == TokenType::TokenAdd ||
              current_token.token_type == TokenType::TokenMul ||
              current_token.token_type == TokenType::TokenSin);
        if (current_token.token_type == TokenType::TokenInputNumber)
        {
            uint32_t start_pos = current_token.start_pos + 1;
            uint32_t end_pos = current_token.end_pos;
            CHECK(end_pos > start_pos || end_pos <= this->statement_.length())
                << "Current token has a wrong length";
            const std::string &str_number =
                std::string(this->statement_.begin() + start_pos,
                            this->statement_.begin() + end_pos);
            return std::make_shared<TokenNode>(std::stoi(str_number), nullptr, nullptr);
        }
        else if (current_token.token_type == TokenType::TokenMul ||
                 current_token.token_type == TokenType::TokenAdd)
        {
            std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
            current_node->num_index = int(current_token.token_type);

            index += 1;
            CHECK(index < this->tokens_.size()) << "Missing left bracket!";
            CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);

            index += 1;
            CHECK(index < this->tokens_.size()) << "Missing correspond left token!";
            const auto left_token = this->tokens_.at(index);

            if (left_token.token_type == TokenType::TokenInputNumber ||
                left_token.token_type == TokenType::TokenAdd ||
                left_token.token_type == TokenType::TokenMul ||
                left_token.token_type == TokenType::TokenSin)
            {
                current_node->left = Generatebyrecursion(index);
            }
            else
            {
                LOG(FATAL) << "Unknown token type: " << int(left_token.token_type);
            }

            index += 1;
            CHECK(index < this->tokens_.size()) << "Missing comma!";
            CHECK(this->tokens_.at(index).token_type == TokenType::TokenComma);

            index += 1;
            CHECK(index < this->tokens_.size()) << "Missing correspond right token!";
            const auto right_token = this->tokens_.at(index);
            if (right_token.token_type == TokenType::TokenInputNumber ||
                right_token.token_type == TokenType::TokenAdd ||
                right_token.token_type == TokenType::TokenMul ||
                right_token.token_type == TokenType::TokenSin)
            {
                current_node->right = Generatebyrecursion(index);
            }
            else
            {
                LOG(FATAL) << "Unknown token type: " << int(right_token.token_type);
            }

            index += 1;
            CHECK(index < this->tokens_.size()) << "Missing right bracket!";
            CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
            return current_node;
        }
        else if (current_token.token_type == TokenType::TokenSin)
        {
            std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
            current_node->num_index = int(current_token.token_type);

            index += 1;
            CHECK(index < this->tokens_.size()) << "Missing left bracket!";
            CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);

            index += 1;
            CHECK(index < this->tokens_.size()) << "Missing correspond left token!";
            const auto left_token = this->tokens_.at(index);

            if (left_token.token_type == TokenType::TokenInputNumber ||
                left_token.token_type == TokenType::TokenAdd ||
                left_token.token_type == TokenType::TokenMul ||
                left_token.token_type == TokenType::TokenSin)
            {
                current_node->left = Generatebyrecursion(index);
            }
            else
            {
                LOG(FATAL) << "Unknown token type: " << int(left_token.token_type);
            }

            index += 1;
            CHECK(index < this->tokens_.size()) << "Missing right bracket!";
            CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
            return current_node;
        }
        else
        {
            LOG(FATAL) << "Unknown token type: " << int(current_token.token_type);
        }
    }

    std::shared_ptr<TokenNode> ExpressionParser::Generatebystack()
    {

        std::stack<std::shared_ptr<TokenNode>> store;
        std::stack<std::shared_ptr<TokenNode>> compute;
        std::shared_ptr<TokenNode> result;
        // std::shared_ptr<TokenNode> lastcompute;

        uint16_t operandinput = 0;

        for (const auto &token : this->tokens_)
        {
            if (token.token_type == TokenType::TokenAdd || token.token_type == TokenType::TokenMul)
            {
                operandinput = 0;
                std::shared_ptr<TokenNode> tmp = std::make_shared<TokenNode>();
                tmp->num_index = int(token.token_type);
                store.push(tmp);
                if (compute.empty())
                {
                    compute.push(tmp);
                    result = tmp;
                }
                else
                {
                    const auto &lastcompute = compute.top();
                    if (lastcompute->left != nullptr)
                    {
                        CHECK(lastcompute->right == nullptr) << "TokenNode is full";
                        lastcompute->right = tmp;
                    }
                    else
                    {
                        lastcompute->left = tmp;
                    }
                    compute.push(tmp);
                }
            }
            else if (token.token_type == TokenType::TokenInputNumber)
            {
                std::shared_ptr<TokenNode> tmp = std::make_shared<TokenNode>();
                const std::string &str_number = std::string(this->statement_.begin() + token.start_pos,
                                                            this->statement_.begin() + token.end_pos);
                tmp->num_index = std::stoi(str_number);
                operandinput++;
                auto &lastcompute = compute.top();
                if (lastcompute->num_index == int(TokenType::TokenAdd) ||
                    lastcompute->num_index == int(TokenType::TokenMul))
                {
                    if (operandinput == 2)
                    {
                        lastcompute->right = tmp;

                        while (1)
                        {
                            if (store.empty())
                            {
                                break;
                            }
                            store.pop();
                            store.pop();
                            compute.pop();
                            auto &precompute = compute.top();
                            auto &node = store.top();
                            if (node->num_index >= 0)
                            {
                                // CHECK(precompute->right == nullptr && precompute->left == node)
                                //     << "error in [operandinput == 2][node->num_index > 0] ";
                                precompute->right = lastcompute;
                                lastcompute = precompute;
                            }
                            else if (node->num_index == int(TokenType::TokenAdd) ||
                                     node->num_index == int(TokenType::TokenMul))
                            {
                                precompute->left = lastcompute;
                                std::shared_ptr<TokenNode> tmp2 = std::make_shared<TokenNode>();
                                tmp2->num_index = int(TokenType::Tokenoccupy);
                                store.push(tmp2);
                                break;
                            }
                            else if (node->num_index == int(TokenType::TokenSin))
                            {
                                precompute->left = lastcompute;
                                lastcompute = precompute;
                                std::shared_ptr<TokenNode> tmp2 = std::make_shared<TokenNode>();
                                tmp2->num_index = int(TokenType::Tokenoccupy);
                                store.push(tmp2);
                            }
                            else if (node->num_index == int(TokenType::Tokenoccupy))
                            {
                            }
                            else
                            {
                                CHECK(false) << "error in operandinput==2";
                            }
                        }
                        operandinput = 1;
                    }
                    else if (operandinput == 1)
                    {
                        CHECK(lastcompute->left == nullptr) << "TokenNode input error";
                        lastcompute->left = tmp;
                        store.push(tmp);
                    }
                }
                else if (lastcompute->num_index == int(TokenType::TokenSin))
                {
                    lastcompute->right = tmp;
                    std::shared_ptr<TokenNode> tmp2 = std::make_shared<TokenNode>();
                    tmp2->num_index = int(TokenType::Tokenoccupy);
                    store.push(tmp2);
                    while (1)
                    {
                        if (store.empty())
                        {
                            break;
                        }
                        store.pop();
                        store.pop();
                        compute.pop();
                        auto &precompute = compute.top();
                        auto &node = store.top();
                        if (node->num_index >= 0)
                        {
                            // CHECK(precompute->right == nullptr && precompute->left == node)
                            //     << "error in [operandinput == 2][node->num_index > 0] ";
                            precompute->right = lastcompute;
                            lastcompute = precompute;
                        }
                        else if (node->num_index == int(TokenType::TokenAdd) ||
                                 node->num_index == int(TokenType::TokenMul))
                        {
                            precompute->left = lastcompute;
                            std::shared_ptr<TokenNode> tmp2 = std::make_shared<TokenNode>();
                            tmp2->num_index = int(TokenType::Tokenoccupy);
                            store.push(tmp2);
                            break;
                        }
                        else if (node->num_index == int(TokenType::TokenSin))
                        {
                            precompute->left = lastcompute;
                            lastcompute = precompute;
                            std::shared_ptr<TokenNode> tmp2 = std::make_shared<TokenNode>();
                            tmp2->num_index = int(TokenType::Tokenoccupy);
                            store.push(tmp2);
                        }
                        else if (node->num_index == int(TokenType::Tokenoccupy))
                        {
                        }
                        else
                        {
                            CHECK(false) << "error in ";
                        }
                    }
                }
                else
                {
                    CHECK(true) << "lastcompute is error in TokenType::TokenSin";
                }
            }
            else
            {
            }
        }
        return result;
    }

    void ReversePolish(const std::shared_ptr<TokenNode> &root_node,
                       std::vector<std::shared_ptr<TokenNode>> &reverse_polish, bool method)
    {
        if (method)
        {
            ReversePolishbyrecurision(root_node, reverse_polish);
        }
        else
        {
            ReversePolishbystack(root_node, reverse_polish);
        }
    }

    void ReversePolishbystack(const std::shared_ptr<TokenNode> &root_node,
                              std::vector<std::shared_ptr<TokenNode>> &reverse_polish)
    {
        std::stack<std::shared_ptr<TokenNode>> backlist;
        backlist.push(root_node);
        std::shared_ptr<TokenNode> head = backlist.top();

        while (!backlist.empty())
        {
            while (head->left != nullptr)
            {
                backlist.push(head->left);
                head = head->left;
            }
            if (head->right != nullptr)
            {
                backlist.push(head->right);
                head = head->right;
            }
            else
            {
                while (1)
                {
                    std::shared_ptr<TokenNode> record = head;
                    reverse_polish.push_back(head);
                    if (backlist.empty())
                    {
                        break;
                    }
                    backlist.pop();
                    head = backlist.top();
                    if (head->right != nullptr && record != head->right)
                    {
                        head = head->right;
                        backlist.push(head);
                        break;
                    }
                }
            }
        }
    }

    void ReversePolishbyrecurision(const std::shared_ptr<TokenNode> &root_node,
                                   std::vector<std::shared_ptr<TokenNode>> &reverse_polish)
    {
        if (root_node != nullptr)
        {
            ReversePolish(root_node->left, reverse_polish);
            ReversePolish(root_node->right, reverse_polish);
            reverse_polish.push_back(root_node);
        }
    }

    std::vector<std::shared_ptr<TokenNode>> ExpressionParser::Generate(bool method)
    {
        if (this->tokens_.empty())
        {
            this->Tokenizer(true);
        }
        std::shared_ptr<TokenNode> root;
        if (method)
        {
            int index = 0;
            root = Generatebyrecursion(index);
            CHECK(index == tokens_.size() - 1);
        }
        else
        {
            root = Generatebystack();
        }
        CHECK(root != nullptr);

        // 转逆波兰式,之后转移到expression中
        std::vector<std::shared_ptr<TokenNode>> reverse_polish;
        ReversePolish(root, reverse_polish, method);
        return reverse_polish;
    }

    TokenNode::TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left,
                         std::shared_ptr<TokenNode> right)
        : num_index(num_index), left(left), right(right) {}
}
