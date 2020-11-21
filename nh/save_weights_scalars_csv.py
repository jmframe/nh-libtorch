import pickle
import torch
import numpy as np
import pandas as pd

with open('sugar_creek_scaler.p','rb') as fp:
    scalar = pickle.load(fp)
trained_model_weights = torch.load('sugar_creek_trained.pt') 
print('scalars')

for i in scalar:
    print(i)
    weights = scalar[i]
    print(pd.DataFrame(weights))

#print('\n')
#print('model weights')
#for i in trained_model_weights:
#    print(i)
#    weights = trained_model_weights[i].detach().numpy()
#    pd.DataFrame(weights).to_csv(i+'.csv')
