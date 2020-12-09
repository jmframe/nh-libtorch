#include <iostream>
#include "CSV_Reader.h"
#include <torch/script.h>
#include <unordered_map>
#include <iterator>

typedef std::unordered_map< std::string, std::unordered_map< std::string, double> > ScaleParams;

ScaleParams read_scale_params(std::string path)
{
    //Read the mean and standard deviation and put them into a map
    ScaleParams params;
    CSVReader reader(path);
    auto data = reader.getData();
    std::vector<std::string> header = data[0];
    //Advance the iterator to the first data row (skip the header)
    auto row = data.begin();
    std::advance(row, 1);
    //Loop form first row to end of data
    for(; row != data.end(); ++row)
    {
	for(int i = 0; i < (*row).size(); i++)
	{   //row has var, mean, std_dev
 	    //read into map keyed on var name, param name
    	    params[ (*row)[0] ]["mean"] = strtof( (*row)[1].c_str(), NULL); 
  	    params[ (*row)[0] ]["std_dev"] = strtof( (*row)[2].c_str(), NULL);
	}
    }

    return params;
}


std::vector<torch::Tensor> read_input(std::string path, std::string scale_path, int& nrows, int& ncols)
{
    // Import forcing data and save in a vector of vectors. 
    // Columns are in order for trained NeuralHydrology code
    // LWDOWN, PSFC, Q2D, RAINRATE, SWDOWN, T2D, U2D, V2D, lat, lon, area_sqkm
    // NOTE this order is important!!!
    
    CSVReader reader(path); 
    std::vector<std::vector<std::string> > data_str = reader.getData();
    nrows = data_str.size();
    ncols = data_str[0].size();
    std::vector<torch::Tensor> out;
    //Create nrows default (undefined) tensors
    out.resize(nrows);
    //get the scaling parameters
    ScaleParams scale = read_scale_params(scale_path);
    //get the header row
    auto header = data_str[0];
    //variables to parse into
    double temp, mean, std_dev;
    torch::Tensor t;

    for (int i = 1; i < nrows ; ++i) {
	//init an empty tensor
	t =  torch::zeros( {1, ncols}, torch::dtype(torch::kFloat64) );
        for (int j = 0; j < ncols; ++j){
	    //data value as double
            temp = strtof(data_str[i][j].c_str(),NULL);
            mean = scale[ header[j] ]["mean"];
	    std_dev = scale[ header[j] ]["std_dev"];
	    t[0][j] =  (temp - mean) / std_dev;
        }
	out[i] = t;
	//you can push data to a vector and crate the tensor from the blob, but have to clone it since the data would be scoped to the function
	//so just create the empty tensor and fill it in, so the vector out owns the tensor pointer AND its data...
	// torch::from_blob(data.data(), {1,ncols}, torch::dtype(torch::kFloat64)).clone();
    }
    
    return out;
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

    std::vector<torch::Tensor> input_data = read_input("data/sugar_creek_input.csv", "input_scaling.csv", nrows, ncols);

    // create the rest of the input tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    torch::Tensor h_t = torch::zeros({1, 1, 64}, options);
    torch::Tensor c_t = torch::zeros({1, 1, 64}, options);
    torch::Tensor output = torch::zeros({1}, options);
    
    // Input to the model is a vector of "IValues" (tensors)
    //    std::vector<torch::jit::IValue> input = {input_tensor, h_t, c_t};
    // `model.forward` does what you think it does; it returns an IValue
    // which we convert back to a Tensor
    // Loop over each input
    for (int i = 1; i < nrows; ++i){
        std::vector<torch::jit::IValue> inputs;
        // Create the model input for one time step
	inputs.push_back(input_data[i].to(device));
        inputs.push_back(h_t);
        inputs.push_back(c_t);
	// Run the model
        auto output_tuple = model.forward(inputs);
	//Get the outputs
        torch::Tensor output = output_tuple.toTuple()->elements()[0].toTensor();
        torch::Tensor h_t = output_tuple.toTuple()->elements()[1].toTensor();
        torch::Tensor c_t = output_tuple.toTuple()->elements()[2].toTensor();
        std::cout << output;

    }
  
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
