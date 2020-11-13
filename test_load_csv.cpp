#include <iostream>
#include "CSV_Reader.h"

int main(int argc, char** argv) {

    std::cout << "importing sugar creek data \n";
    //Call CSVReader constructor
    CSVReader reader("sugar_creek_input.csv");
    //Get the data from CSV File
    std::vector<std::vector<std::string> > data_str = reader.getData();

    int nrow = data_str.size();
    int ncol = data_str[0].size();
    std::vector<std::vector<double> > data_out;
    data_out.resize(ncol);

    for (int i = 1; i < nrow; ++i) {

        for (int j = 0; j < ncol; ++j){
            double temp = strtof(data_str[i][j].c_str(),NULL);
            data_out[j].push_back(temp);
        }
    }

    std::vector<double> LWDOWN = data_out[1];
    // LWDOWN;
    // PSFC
    // Q2D
    // RAINRATE
    // SWDOWN
    // T2D
    // U2D
    // V2D
    // lat
    // lon
    // area_sqkm

    for(int i=0; i < LWDOWN.size(); i++)
    std::cout << LWDOWN.at(i) << ' ';    
    // std::cout << "\n";

    // for (int i = 1; i < ncol; ++i) {
    //     for (int j = 0; j < 10; ++j){
    //       std::cout << data_out[i][j] << " ";
    //     }
    //     std::cout << "\n";
    // }
  
  return 0;
}
