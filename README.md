# Run a prediction from a trained NeuralHydrology model through the C++ Pytorch API

## Please update this readme frequently. I am just adding a bunch of notes right now, but will replace with code documentation as the project develops.

#### I worked through Gary's blog to get a simple pytorch model running with the C++ API. https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-1-why-libtorch/

#### NeuralHydrology python code is in the directory ./nh

#### Got rid of the cfg functionality, and just added a dictionary in the cudalstm, basemodel and head files, although I'm sure it is not needed in all three. Probably just one.
