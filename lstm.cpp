#include <iostream>
#include "CSV_Reader.h"
#include <torch/script.h>
#include <unordered_map>
#include <iterator>

typedef std::unordered_map< std::string, std::unordered_map< std::string, double> > ScaleParams;

ScaleParams read_scale_params()
{
    //Read the mean and standard deviation and put them into a map
    ScaleParams params;
    CSVReader reader("input_scaling.csv");
    auto data = reader.getData();
    std::vector<std::string> header = data[0];
    auto row = data.begin();
    std::advance(row, 1);
    for(; row != data.end(); ++row)
    {
	for(int i = 0; i < (*row).size(); i++)
	{
    	    params[ (*row)[0] ]["mean"] = strtof( (*row)[1].c_str(), NULL); 
  	    params[ (*row)[0] ]["std_dev"] = strtof( (*row)[2].c_str(), NULL);
	}
    }

    return params;
}


torch::Tensor* read_input(std::string path, int& nrows, int& ncols)
{
    // Import forcing data and save in a vector of vectors. 
    // Columns are in order for trained NeuralHydrology code
    // LWDOWN, PSFC, Q2D, RAINRATE, SWDOWN, T2D, U2D, V2D, lat, lon, area_sqkm
    CSVReader reader(path); 
    std::vector<std::vector<std::string> > data_str = reader.getData();
    nrows = data_str.size();
    ncols = data_str[0].size();
    std::vector<torch::Tensor> out;
    out.resize(nrows);
    ScaleParams scale = read_scale_params();

    auto header = data_str[0];
    double temp, mean, std_dev;
    std::vector< double > data(ncols);
    torch::Tensor* tensors = new torch::Tensor[nrows];

    for (int i = 1; i < nrows ; ++i) {
	//init an empty tensor
	//out.push_back( torch::zeros( {1, ncols}, torch::dtype(torch::kFloat64) ));
	tensors[i] =  torch::zeros( {1, ncols}, torch::dtype(torch::kFloat64) );
        for (int j = 0; j < ncols; ++j){
            temp = strtof(data_str[i][j].c_str(),NULL);
            mean = scale[ header[j] ]["mean"];
	    std_dev = scale[ header[j] ]["std_dev"];
	    data[j] = (temp - mean) / std_dev;
	    tensors[i][0][j] =  (temp - mean) / std_dev;
        }
        //std::cout<<"HERE\n";
        //std::cout<<data<<"\n";
	//torch::Tensor t = torch::from_blob(data.data(), {1,ncols} , torch::dtype(torch::kFloat64));
        //std::cout<<t<<"\n";	
        //out.push_back( std::move( t ) );
        //std::cout<<tensors[i]<<"\n";
    }
    
    return tensors;
}
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr <<"Please provide model name." << std::endl;
    return 1;
  }
  bool useGPU = false;
  if (argc == 3 && std::string(argv[2]) == "-g")
  {
	useGPU = true;
  }

  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);i
  torch::Device device( useGPU ? torch::kCUDA : torch::kCPU, 0);
  
  torch::manual_seed(0);

  torch::jit::script::Module model;

  try {

    model = torch::jit::load(std::string(argv[1]));
    std::cout << "load model successfully \n"; 
    // Explicitly load model onto CPU, you can use kGPU if you are on Linux
    // and have libtorch version with CUSA support (and a GPU)
    model.to( device );
  
    // Set to `eval` model (just like Python)
    model.eval();
  
//    // Get current value of `init_func`
//    auto init_func = model.attr("init_func");
//  
//    // Set initialization function to ones
//    model.setattr("init_func", "ones");
  
    // Within this scope/thread, don't use gradients (again, like in Python)
    torch::NoGradGuard no_grad_;

    std::cout << "importing sugar creek data \n";
    int nrows, ncols;

    //std::vector<torch::Tensor> input_data = read_input("data/sugar_creek_input.csv", nrows, ncols);
    torch::Tensor* input_data = read_input("data/sugar_creek_input.csv", nrows, ncols);
    std::cout<<"Preparing input tensor with "<<nrows<<", "<<ncols<<"\n";   
    // turn array into torch tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    //torch::Tensor input_tensor = torch::from_blob(input_arr, {nrows, ncols}, torch::dtype(torch::kFloat64)).to(device);
    torch::Tensor h_t = torch::zeros({1, 1, 64}, options);
    torch::Tensor c_t = torch::zeros({1, 1, 64}, options);
    torch::Tensor output = torch::zeros({1}, options);
    std::cout<<"Created tensors, preparing for forward pass\n";
    // Input to the model is a vector of "IValues" (tensors)
//    std::vector<torch::jit::IValue> input = {input_tensor, h_t, c_t};
    // `model.forward` does what you think it does; it returns an IValue
    // which we convert back to a Tensor
    for (int i = 1; i < nrows; ++i){
        std::vector<torch::jit::IValue> inputs;
	//Why do this when we already have input_tensor above?
        //torch::Tensor input_row = input_data[i].to(device);
        inputs.push_back(input_data[i].to(device));
        inputs.push_back(h_t);
        inputs.push_back(c_t);
        auto output_tuple = model.forward(inputs);
        torch::Tensor output = output_tuple.toTuple()->elements()[0].toTensor();
        torch::Tensor h_t = output_tuple.toTuple()->elements()[1].toTensor();
        torch::Tensor c_t = output_tuple.toTuple()->elements()[2].toTensor();
        std::cout << output;

    }
  
    delete[] input_data;
 
  }
  catch (const c10::Error& e) {
    std::cerr << "An error occured: " << e.what() << std::endl;
    return 1;
  }
  
  // We can list the model attributes in C++ like so:
  for (const auto& attr : model.named_attributes())
    std::cout << attr.name << std::endl;

  // Get a list of methods in the model class
  for (const auto& method : model.get_methods())
    std::cout << method.name() << "\n";

   return 0;
}
