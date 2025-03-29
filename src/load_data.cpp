#include "load_data.hpp"
#include <fstream>
#include <glog/logging.h>
namespace star
{

    arma::fmat CSVDataLoader::LoadData(const std::string &file_path, char split_char)
    {
        arma::fmat data;

        if (file_path.empty())
        {
            LOG(ERROR) << "CSV file path is empty: " << file_path;
            return data;
        }

        std::ifstream in(file_path);
        if (!in.is_open() || !in.good())
        {
            LOG(ERROR) << "File open failed: " << file_path;
            return data;
        }

        std::string line_str;
        std::stringstream filestream;

        const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
        data.zeros(rows, cols);

        size_t row = 0;
        while (in.good())
        {
            std::getline(in, line_str);

            std::string token;
            filestream.clear();
            filestream.str(line_str);

            size_t col = 0;
            while (filestream.good())
            {
                std::getline(filestream, token, split_char);

                try
                {
                    data.at(row, col) = std::stof(token);
                }
                catch (std::exception &e)
                {
                    DLOG(ERROR) << "Parse CSV File meet error: " << e.what()
                                << " row:" << row << " col:" << col;
                }
                col++;
                CHECK_LE(col, cols) << "There are excessive elements on the column";
            }
            row++;
            CHECK(row <= rows) << "There are excessive elements on the row";
        }

        return data;
    }

    std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream &file, char split_char)
    {
        size_t rows = 0;
        size_t cols = 0;
        file.clear();
        std::string sline;
        std::stringstream linestream;

        const std::ifstream::pos_type position = file.tellg();

        while (file.good())
        {
            std::getline(file, sline);
            if (sline.empty())
            {
                break;
            }
            linestream.clear();
            linestream.str(sline);
            std::string token;

            size_t linecols = 0;
            while (linestream.good())
            {
                std::getline(linestream, token, split_char);
                linecols++;
            }

            if (linecols > cols)
            {
                cols = linecols;
            }
            rows++;
        }

        file.clear();
        file.seekg(position);
        return {rows, cols};
    }

}
