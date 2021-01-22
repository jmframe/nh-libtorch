import pickle
import numpy as np
import pandas as pd

model_name = 'nwmv3_nosnow_normalarea_672_scaler'

dyns = ['RAINRATE', 'Q2D', 'T2D', 'LWDOWN', 'SWDOWN', 'PSFC', 'U2D', 'V2D']
atts = ['area_sqkm', 'lat', 'lon']
namez = dyns + atts
ngen_names = ['Precip_rate', 'SPFH_2maboveground_kg_per_kg','TMP_2maboveground_K','DLWRF_surface_W_per_meters_squared',
              'DSWRF_surface_W_per_meters_squared', 'PRES_surface_Pa', 'UGRD_10maboveground_meters_per_second',
              'VGRD_10maboveground_meters_per_second', 'Area_Square_km', 'Latitude', 'Longitude', 'obs']

with open(model_name+'.p', 'rb') as f:
    weight_dict = pickle.load(f)
dmeans = list(np.array([weight_dict['xarray_means'][i].values for i in dyns]))
smeans = list(np.array([weight_dict['attribute_means'][i] for i in atts]))
meanz = dmeans + smeans
meanz.append(weight_dict['xarray_means']['obs'].values)
dstds = list(np.array([weight_dict['xarray_stds'][i].values for i in dyns]))
sstds = list(np.array([weight_dict['attribute_stds'][i] for i in atts]))
stdz = dstds + sstds
stdz.append(weight_dict['xarray_stds']['obs'].values)

df = pd.DataFrame({'variable':ngen_names, 'mean': meanz, 'std_dev':stdz})

df.to_csv(model_name+'.csv', index=False)
