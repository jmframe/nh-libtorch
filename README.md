# Run a prediction from a trained NeuralHydrology model through the C++ Pytorch API

## Please update this readme frequently. I am just adding a bunch of notes right now, but will replace with code documentation as the project develops.

#### I worked through Gary's blog to get a simple pytorch model running with the C++ API. https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-1-why-libtorch/

#### NeuralHydrology python code is in the directory ./nh

#### Got rid of the cfg functionality, and just added a dictionary in the cudalstm, basemodel and head files, although I'm sure it is not needed in all three. Probably just one.

#### The cudalstm has been traced, and is in the nh folder. I saved the text output as a text file called "jit_traced_cudalstm.py", this has the re-written python code which has been converted to binary in the "cudalstm.ptc" file.

#### The next step, and one of the most important steps is loading in the data to be used in the forward pass for prediction. I currently have the data saved in a pickle file, but the Pytorch C++ API will not be able to load this in, so I need to convert it to either a binary file, or a json file. From what I have read on-line (https://discuss.pytorch.org/t/serialization-in-c-frontend-api/30200) the binary file option is prefered, because json is slow with large tensors. So, I just need to convert the model input data, and the model weights, into binary files, from their pickle files outputted from NeuralHydrology.

#### Finally got the forcing data to inport with C++, not have to pass to pytorch.
