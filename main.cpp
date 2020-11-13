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

    //Call CSVReader constructor
    cout << "importing sugar creek data";
    CSVReader reader("sugar_creek_input.csv")
    //Get the data from CSV File
    std::vector<std::vector<std::string> > data_list = reader.getData();

    // PASTE IN NEW CODE FROM TEST.CPP
    // expect a c array. call vector.data(), will give first address.


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
    input_dict.reserve(11);
    input_dict.insert({std::string("LWDOWN"), LWDOWN});
    input_dict.insert({std::string("PSFC"), PSFC});
    input_dict.insert({std::string("Q2D"), Q2D});
    input_dict.insert({std::string("RAINRATE"), RAINRATE});
    input_dict.insert({std::string("SWDOWN"), SWDOWN});
    input_dict.insert({std::string("T2D"), T2D});
    input_dict.insert({std::string("U2D"), U2D});
    input_dict.insert({std::string("V2D"), V2D});
    input_dict.insert({std::string("lat"), lat});
    input_dict.insert({std::string("lon"), lon});
    input_dict.insert({std::string("area_sqkm"), area_sqkm});
    
      
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
