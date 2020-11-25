#include <iostream>
#include "CSV_Reader.h"
#include <torch/script.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr <<"Please provide model name." << std::endl;
    return 1;
  }

  torch::manual_seed(0);

  torch::jit::script::Module model;

  try {

    model = torch::jit::load(std::string(argv[1]));
  
    // Explicitly load model onto CPU, you can use kGPU if you are on Linux
    // and have libtorch version with CUSA support (and a GPU)
    model.to(torch::kCPU);
  
    // Set to `eval` model (just like Python)
    model.eval();
  
    // Get current value of `init_func`
    auto init_func = model.attr("init_func");
  
    // Set initialization function to ones
    model.setattr("init_func", "ones");
  
    // Within this scope/thread, don't use gradients (again, like in Python)
    torch::NoGradGuard no_grad_;

    // Import forcing data and save in a vector of vectors. 
    // Columns are in order for trained NeuralHydrology code
    // LWDOWN, PSFC, Q2D, RAINRATE, SWDOWN, T2D, U2D, V2D, lat, lon, area_sqkm
    std::cout << "importing sugar creek data \n";
    CSVReader reader("../sugar_creek_input.csv");
    std::vector<std::vector<std::string> > data_str = reader.getData();
    int nrow = data_str.size();
    int ncol = data_str[0].size();
    double input_arr[300][ncol];
    std::vector<std::vector<double> > data_col;
    data_col.resize(ncol);
    std::vector<std::vector<double> > data_row;
    data_row.resize(nrow);
    for (int i = 1; i < 301; ++i) {
        for (int j = 0; j < ncol; ++j){
            double temp = strtof(data_str[i][j].c_str(),NULL);
            temp = (temp - mean) / std
            data_col[j].push_back(temp);
            input_arr[i][j] = temp;
        }
    }
    for (int i = 1; i < ncol; ++i) {
        for (int j = 0; j < nrow; ++j){
            double temp = strtof(data_str[j][i].c_str(),NULL);
            data_row[j].push_back(temp);
        }
    }

    // turn array into torch tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor input_tensor = torch::from_blob(input_arr, {300, ncol}, options);
    torch::Tensor h_t = torch::zeros({ 1, 1, 100 }, torch::dtype(torch::kFloat))
    torch::Tensor c_t = torch::zeros({ 1, 1, 100 }, torch::dtype(torch::kFloat))

    // Input to the model is a vector of "IValues" (tensors)
    std::vector<torch::jit::IValue> input = input_tensor
    std::vector<torch::jit::IValue> h_t = h_t
    std::vector<torch::jit::IValue> c_t = c_t
    
    // `model.forward` does what you think it does; it returns an IValue
    // which we convert back to a Tensor
    auto output, h_t, c_t = model.forward(input, h_t, c_t, Weights).toTensor();
    //.forward(input)
  
    // Extract size of output (of the first and only batch) and preallocate
    // a vector with that size
    auto output_size = output.sizes()[1];
    auto output_vector = std::vector<float>(output_size);
  
    // Fill result vector with tensor items using `Tensor::item`
    for (int i=0; i < output_size; i++) {
      output_vector[i] = output[0][i].item<float>();
    }
  
    // Print the vector here
    for (float x : output_vector)
      std::cout << x << ", ";
    std::cout << std::endl;
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

  // Run a model class member function with parameters other than `forward`:
  // (Note that `model.forward` is just a shortcut for 
  // `model.run_method("forward", ..)`)
  // auto result = model.run_method("forward", ..);


  return 0;
}
