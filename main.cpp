#include <iostream>

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
  
    // Input to the model is a vector of "IValues" (tensors)
    std::vector<torch::jit::IValue> input = {
      // Corresponds to `input`
      torch::zeros({ 1,20 }, torch::dtype(torch::kFloat)),
      // Corresponds to the `future` parameter
      torch::full({ 1 }, 10, torch::dtype(torch::kInt))
    };
  
    // `model.forward` does what you think it does; it returns an IValue
    // which we convert back to a Tensor
    auto output = model
      .forward(input)
      .toTensor();
  
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
