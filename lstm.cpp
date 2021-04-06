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
    // WRONG:   LWDOWN, PSFC, Q2D, RAINRATE, SWDOWN, T2D, U2D, V2D, lat, lon, area_sqkm
    // WRIGHT: RAINRATE, Q2D, T2D, LWDOWN, SWDOWN, PSFC, U2D, V2D, area_sqkm, lat, lon
    // NOTE this order is important!!!
   
    std::vector<double> meanz;
    std::vector<double> stdz;
//    meanz = {0.0000467, 0.006817294, 283.25012, 302.84995, 182.7724, 94364.266, 1.0540344, 0.7839034, 474.2365, 39.949427, -96.920133};
//    stdz = { 0.00022482131, 0.004297631, 10.967861, 65.58242, 253.21057, 8891.513, 3.040779, 3.262279, 510.592316, 4.182431, 16.651526};
    //nwmv3_normalarea_scaler
//    meanz = {0.0000422, 0.006754941, 283.35245, 298.0179, 191.4093, 92224.42, 1.0639832, 0.6383235, 409.477622, 38.771945, -96.486558};
//    stdz = {0.000213777, 0.00443235, 11.103534, 68.286644, 261.9189, 8617.37, 3.1770709, 3.433336, 542.945063, 6.012443, 18.294944};
    //nwmv3_nosnow_normalarea_672_scaler
    meanz = {0.00004726, 0.00836329, 287.22906494, 324.31698608, 194.36569214, 97301.06250000, 0.64094031, 0.66864729, 309.55147533, 36.2614258, -93.24146675};
    stdz = {0.00025686, 0.00457982, 10.20672703, 65.08100128, 265.89950562, 3514.87890625, 2.92154479, 3.53210354, 222.60185322, 4.97177035, 15.40204469};

 
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
    std::cout << header << std::endl;

    //variables to parse into
    double temp, mean, std_dev;
    torch::Tensor t;

    for (int i = 1; i < nrows; ++i) {
	    //init an empty tensor
	    t =  torch::zeros( {1, ncols}, torch::dtype(torch::kFloat64) );
        
      for (int j = 0; j < ncols; ++j){

	      //data value as double
        temp = strtof(data_str[i][j].c_str(),NULL);

        //Multiplu precip to match units
        if (j == 0)
          if (i>16033)
            temp = temp*1000;

        //mean = scale[ header[j] ]["mean"];
	      //std_dev = scale[ header[j] ]["std_dev"];
	      mean = meanz[j];
        std_dev = stdz[j];
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
  
    // Within this scope/thread, don't use gradients (again, like in Python)
    torch::NoGradGuard no_grad;

    std::cout << "importing sugar creek data \n";
    int nrows, ncols;

//    std::vector<torch::Tensor> input_data = read_input("data/sugar_creek_IL_input_all3.csv", "data/input_scaling.csv", nrows, ncols);
    std::vector<torch::Tensor> input_data = read_input("/glade/scratch/jframe/data/sugar_creek_nc/forcing_with_warmup/cat-87-nodate.csv", "data/nosnow_normalarea_672/input_scaling.csv", nrows, ncols);
    std::cout << "sugar creek data has been imported \n" << std::endl;

    // create the rest of the input tensors
    std::cout << "initializing LSTM states \n" << std::endl;
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    torch::Tensor h_t = torch::zeros({1, 1, 64}, options);
    torch::Tensor c_t = torch::zeros({1, 1, 64}, options);
    torch::Tensor output = torch::zeros({1}, options);
    
    std::vector<double> outz;

    // Input to the model is a vector of "IValues" (tensors)
    //    std::vector<torch::jit::IValue> input = {input_tensor, h_t, c_t};
    // `model.forward` does what you think it does; it returns an IValue
    // which we convert back to a Tensor
    // Loop over each input
    std::cout << "Starting time loop \n";
    for (int i = 1; i < nrows; ++i){

        std::vector<torch::jit::IValue> inputs;

        // Create the model input for one time step
	      inputs.push_back(input_data[i].to(device));
        inputs.push_back(h_t);
        inputs.push_back(c_t);

	      // Run the model
        auto output_tuple = model.forward(inputs);

	      //Get the outputs
        torch::Tensor output = output_tuple.toTuple()->elements()[0].toTensor() * 0.052799540 + 0.022887329;
        torch::Tensor h_t2 = output_tuple.toTuple()->elements()[1].toTensor();
        torch::Tensor c_t2 = output_tuple.toTuple()->elements()[2].toTensor();
       
        // Update the states 
       // for (int j = 0; j < 64; ++j){
       //   h_t[0][0][j] = h_t2[0][0][j];
       //   c_t[0][0][j] = c_t2[0][0][j];
       // }
        h_t = h_t2;
        c_t = c_t2;

        // Print the output
        if (i>16033)
            std::cout << output[0][0][0]*15.617*35.315 << std::endl;
        
    }
//    std::cout << outz << std::endl;
  
  }

  catch (const c10::Error& e) {
    std::cerr << "An error occured: " << e.what() << std::endl;
    return 1;
  }
  

   return 0;
}
