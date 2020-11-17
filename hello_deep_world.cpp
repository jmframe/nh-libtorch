#include<iostream>
#include<torch/torch.h>

// build a neural network similar to how you would do it with Pytorch

struct Model : torch::nn::Module {

    // Constructor
    Model() {     
        // construct and register your layers
        in = register_module("in",torch::nn::Linear(8,64));
        h = register_module("h",torch::nn::Linear(64,64));
        out = register_module("out",torch::nn::Linear(64,1));
    }

    // the forward operation (how data will flow from layer to layer)
    torch::Tensor forward(torch::Tensor X){
        // let's pass relu
        X = torch::relu(in->forward(X));
        X = torch::relu(h->forward(X));
        X = torch::sigmoid(out->forward(X));

        // return the output
        return X;
    }
    
    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};

};

int main(){
    Model model;
    auto in = torch::rand({8,});
    auto out = model.forward(in);
    std::cout << in << std::endl;
    std::cout << out << std::endl;
}





