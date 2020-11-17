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
    std::cout << "importing sugar creek data \n";
    CSVReader reader("sugar_creek_input.csv");
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

     // Input to the model is a vector of "IValues" (tensors)
     std::vector<torch::jit::IValue> input = {
     // Corresponds to `input`
     torch::zeros({ 1,20 }, torch::dtype(torch::kFloat)),
     // Corresponds to the `future` parameter
     torch::full({ 1 }, 10, torch::dtype(torch::kInt))
     };

    // From example
    // https://github.com/driazati/torchscript-examples/...
    // blob/5ca0a02a835dcf75c66336fbc2280351b13fdc40/values/ivalues.cpp#L15-L21
    torch::ivalue::UnorderedMap input_dict;
    input_dict.reserve(11);
    input_dict.insert({std::string("LWDOWN"), torch::data_out[0]});
    input_dict.insert({std::string("PSFC"), data_out[1]});
    input_dict.insert({std::string("Q2D"), data_out[2]});
    input_dict.insert({std::string("RAINRATE"), data_out[3]});
    input_dict.insert({std::string("SWDOWN"), data_out[4]});
    input_dict.insert({std::string("T2D"), data_out[5]});
    input_dict.insert({std::string("U2D"), data_out[6]});
    input_dict.insert({std::string("V2D"), data_out[7]});
    input_dict.insert({std::string("lat"), data_out[8]});
    input_dict.insert({std::string("lon"), data_out[9]});
    input_dict.insert({std::string("area_sqkm"), data_out[10]});
    
      
    // `model.forward` does what you think it does; it returns an IValue
    // which we convert back to a Tensor
    auto output = model
      .forward(input_dict)
      .toTensor();
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
//  auto result = model.run_method("forward", ..);

  return 0;
}
