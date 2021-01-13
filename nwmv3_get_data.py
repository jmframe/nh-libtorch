import pickle
import numpy as np
import pandas as pd

def dynamic_data(basin: str) -> pd.DataFrame:
    data_dir = '/glade/scratch/jframe/neuralhydrology/data/'
    # Load dynamic inputs
    with open(data_dir+"full_period.pickle", 'rb') as f:
        start_end_date = pickle.load(f)
    start_end_date = start_end_date[basin]
    start_date = start_end_date['start_dates']
    end_date = start_end_date['end_dates']

    with open(data_dir+'basin_ave_forcing/'+basin+'.pickle', 'rb') as f:
        basin_forcing = pickle.load(f)

    #basin_forcing = basin_forcing.set_index('TIME')
    basin_forcing = basin_forcing.rename_axis('date')

    with open(data_dir+'obs_q/'+basin+'.csv', 'r') as f:
        basin_q = pd.read_csv(f)
    basin_q = basin_q.set_index('POSIXct')
    basin_q = basin_q.rename_axis('date')

    df = basin_forcing.join(basin_q['obs'])

    # replace invalid discharge values by NaNs
    qobs_cols = [col for col in df.columns if 'obs' in col.lower()]
    for col in qobs_cols:
        df.loc[df[col] < 0, col] = np.nan
    df.loc[np.isnan(df['obs']),'obs'] = np.nan
    
    return df

def static_data() -> pd.DataFrame:
    data_dir = '/glade/scratch/jframe/data/nwmv3/'
    
    # Load attributes
    hi = pd.read_csv(data_dir+'meta/domainMeta_HI.csv')
    pr = pd.read_csv(data_dir+'meta/domainMeta_PR.csv')
    nosnol = pd.read_csv(data_dir+'meta/nosnowy_large_basin.csv')
    nosnos = pd.read_csv(data_dir+'meta/nosnowy_small_basin.csv')
    snol1 = pd.read_csv(data_dir+'meta/snowy_large_basin_1.csv')
    snol2 = pd.read_csv(data_dir+'meta/snowy_large_basin_2.csv')
    snos = pd.read_csv(data_dir+'meta/snowy_small_basin.csv')
    df = pd.concat([hi, pr, nosnol, nosnos, snol1, snol2, snos])
    df = df.set_index('site_no')

    # For some reason these areas aren't in the tables. I looked them up online.
    df.loc['50038100','area_sqkm'] = 525.768
    df.loc['50051800','area_sqkm'] = 106.42261

    return df
