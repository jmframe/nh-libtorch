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

    std::cout << "size of data_out";
    std::cout << data_out.size();

    // LWDOWN;
    std::cout <<"\n";
    std::cout << "LWDOWN";
    std::vector<double> LWDOWN = data_out[0];
    // PSFC
    std::cout <<"\n";
    std::cout << "PSFC";
    std::vector<double> PSFC = data_out[1];
    // Q2D
    std::cout <<"\n";
    std::cout << "Q2D";
    std::vector<double> Q2D = data_out[2];
    // RAINRATE
    std::cout <<"\n";
    std::cout <<"RAINRATE";
    std::vector<double> RAINRATE = data_out[3];
    // SWDOWN
    std::cout <<"\n";
    std::cout << "SWDOWN";
    std::vector<double> SWDOWN = data_out[4];
    // T2D
    std::cout <<"\n";
    std::cout <<"T2D";
    std::vector<double> T2D = data_out[5];
    // U2D
    std::cout <<"\n";
    std::cout << "U2D";
    std::vector<double> U2D = data_out[6];
    // V2D
    std::cout <<"\n";
    std::cout << "V2D";
    std::vector<double> V2D = data_out[7];
    // lat
    std::cout <<"\n";
    std::cout << "lat";
    std::vector<double> lat = data_out[8];
    // lon
    std::cout <<"\n";
    std::cout << "lon";
    std::vector<double> lon = data_out[9];
    // area_sqkm
    std::cout <<"\n";
    std::cout << "area_sqkm";
    std::vector<double> area_sqkm = data_out[10];
    std::cout <<"\n";

    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << LWDOWN.at(i) << ' '; 
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << PSFC.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << Q2D.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << RAINRATE.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << SWDOWN.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << T2D.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << U2D.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << V2D.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << lat.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << lon.at(i) << ' ';    
    std::cout <<"\n";
    for(int i=0; i < 10; i++)
    std::cout << area_sqkm.at(i) << ' ';    

    std::cout <<"\n";
  return 0;
}
