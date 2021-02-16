import torch
from torch import nn

class LSTM(nn.Module):
    #--------------------------------------------------------------------------------------------------
    # This is the LSTM model. Based on the simple "CudaLSTM" in NeuralHydrolog
    # Onlt meant for forward predictions, this is not for training. Do training in NeuralHydrology
    #--------------------------------------------------------------------------------------------------
    def __init__(self, input_size=11, hidden_layer_size=64, output_size=1, batch_size=1, seq_length=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.seq_length=seq_length
        self.batch_size = batch_size # We shouldn't neeed to do a higher batch size.
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.head = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_layer, h_t, c_t):
        h_t = h_t.float()
        c_t = c_t.float()
        input_layer = input_layer.float()
        input_view = input_layer.view(self.seq_length, self.batch_size, self.input_size)
        output, (h_t, c_t) = self.lstm(input_view, (h_t,c_t))
        prediction = self.head(output)
        return prediction, h_t, c_t

    #----------------------------------------------
    # Required, static attributes of the model
    #----------------------------------------------
    _att_map = {
        'model_name':         'LSTM for EMELI',
        'version':            '1.0',
        'author_name':        'Jonathan Martin Frame',
        'grid_type':          'none',
        'time_step_type':     '1H',
        'step_method':        'none',
        'time_units':         'none' }

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    # LWDOWN,PSFC,Q2D,RAINRATE,SWDOWN,T2D,U2D,V2D
    #---------------------------------------------
    _input_var_names = [
        'land_surface_radiation~incoming~longwave__energy_flux',
        'land_surface_air__pressure',
        'atmosphere_air_water~vapor__relative_saturation',
        'atmosphere_water__liquid_equivalent_precipitation_rate',
        'land_surface_radiation~incoming~shortwave__energy_flux',
        'land_surface_air__temperature',
        'land_surface_wind__x_component_of_velocity',
        'land_surface_wind__y_component_of_velocity']

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = ['land_surface_water__runoff_volume_flux']

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    #------------------------------------------------------
     _var_name_map = { 
         'land_surface_radiation~incoming~longwave__energy_flux':'LWDOWN',
         'land_surface_air__pressure':'PSFC',
         'atmosphere_air_water~vapor__relative_saturation':'Q2D',
         'atmosphere_water__liquid_equivalent_precipitation_rate':'RAINRATE',
         'land_surface_radiation~incoming~shortwave__energy_flux':'SWDOWN',
         'land_surface_air__temperature':'T2D',
         'land_surface_wind__x_component_of_velocity':'U2D',
         'land_surface_wind__y_component_of_velocity':'V2D'}

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the units of each model variable.
    #------------------------------------------------------
    _var_units_map = {
        'land_surface_water__runoff_volume_flux':'mm'
        #--------------------------------------------------
         'land_surface_radiation~incoming~longwave__energy_flux':'W m-2',
         'land_surface_air__pressure':'Pa',
         'atmosphere_air_water~vapor__relative_saturation':'kg kg-1',
         'atmosphere_water__liquid_equivalent_precipitation_rate':'kg m-2',
         'land_surface_radiation~incoming~shortwave__energy_flux':'W m-2',
         'land_surface_air__temperature':'K',
         'land_surface_wind__x_component_of_velocity':'m s-1',
         'land_surface_wind__y_component_of_velocity':'m s-1'}

    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    def get_attribute(self, att_name):
    
        try:
            return self._att_map[ att_name.lower() ]
        except:
            print('###################################################')
            print(' ERROR: Could not find attribute: ' + att_name)
            print('###################################################')
            print()

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names

    def get_output_var_names(self):
 
        return self._output_var_names

    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, long_var_name):

        return str( self.get_value( long_var_name ).dtype )

    #-------------------------------------------------------------------
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time( self ):
    
        return 0.0

    #-------------------------------------------------------------------
    def get_end_time( self ):

        return (self.n_steps * self.dt)


    #-------------------------------------------------------------------
    def get_current_time( self ):

        return self.time

    #-------------------------------------------------------------------
    def get_time_step( self ):

        return self.dt

    #-------------------------------------------------------------------
    def get_time_units( self ):

        return self.get_attribute( 'time_units' ) 

    #-------------------------------------------------------------------
    def initialize( self, sfg_file=None ):

        self.timer_start = time.time()

        #---------------------------------------------------------------
        #---------------------------------------------------------------
        nldas = data_tools.load_hourly_nldas_forcings(file_path, lat, lon, area_sqkm)

        nldas['U2D'] = nldas['Wind']/2
        nldas['V2D'] = nldas['Wind']/2
        nldas = nldas.loc['2015-01-01':'2015-11-30', ['RAINRATE', 'Q2D', 'T2D', 'LWDOWN',  'SWDOWN',  'PSFC',  'U2D', 'V2D', 'area_sqkm', 'lat', 'lon']]
        with open(self.forcing,'r') as f:
            df = pd.read_csv(f)
        df = df.rename(columns={'precip_rate':'RAINRATE', 'SPFH_2maboveground':'Q2D', 'TMP_2maboveground':'T2D', 
                           'DLWRF_surface':'LWDOWN',  'DSWRF_surface':'SWDOWN',  'PRES_surface':'PSFC',
                           'UGRD_10maboveground':'U2D', 'VGRD_10maboveground':'V2D'})
        df['area_sqkm'] = [area_sqkm for i in range(df.shape[0])]
        df['lat'] = [lat for i in range(df.shape[0])]
        df['lon'] = [lon for i in range(df.shape[0])]
        df = df.drop(['time'], axis=1) #cat-87.csv has no observation data
        df = df.loc[:,['RAINRATE', 'Q2D', 'T2D', 'LWDOWN',  'SWDOWN',  'PSFC',  'U2D', 'V2D', 'area_sqkm', 'lat', 'lon']]
        output_factor = df['area_sqkm'][0] * 35.315 # from m3/s to ft3/s
        # The precipitation rate units for the training set were obviously different than these forcings. Guessing it is a m -> mm conversion.
        df['RAINRATE'] = df['RAINRATE']*1000
        df = pd.concat([nldas.iloc[(nldas.shape[0]-nwarm):,:], df])
        with open(data_dir+'obs_q_02146562.csv', 'r') as f:
            obs = pd.read_csv(f)
        obs = pd.Series(data=list(obs['90100_00060']), index=pd.to_datetime(obs.datetime)).resample('60T').mean()
        obs = obs.loc['2015-12-01':'2015-12-30']

        input_tensor = torch.tensor(df.values)
        warmup_tensor = torch.tensor(nldas.values)
        
        with open(self.scaler_file, 'rb') as fb:
            scalers = pickle.load(fb)
        
        p_dict = torch.load(self.trained_model_file, map_location=torch.device('cpu'))
        m_dict = model.state_dict()
        lstm_weights = {x:p_dict[x] for x in m_dict.keys()}
        head_weights = {}
        head_weights['head.weight'] = p_dict.pop('head.net.0.weight')
        head_weights['head.bias'] = p_dict.pop('head.net.0.bias')
        model.load_state_dict(lstm_weights)
        head.load_state_dict(head_weights)
        
        obs_std = np.array(scalers['xarray_stds'].to_array())[-1]
        obs_mean = np.array(scalers['xarray_means'].to_array())[-1]
        xm = np.array(scalers['xarray_means'].to_array())[:-1]
        xs = np.array(scalers['xarray_stds'].to_array())[:-1]
        xm2 = np.array([xm[x] for x in list([3,2,5,0,4,1,6,7])]) # scalar means ordered to match forcing
        xs2 = np.array([xs[x] for x in list([3,2,5,0,4,1,6,7])]) # scalar stds ordered to match forcing
        att_means = np.append( np.array(scalers['attribute_means'].values)[0],np.array(scalers['attribute_means'].values)[1:3])
        att_stds = np.append(np.array(scalers['attribute_stds'].values)[0], np.array(scalers['attribute_stds'].values)[1:3])
        scaler_mean = np.append(xm2, att_means)
        scaler_std = np.append(xs2, att_stds)
        input_tensor = (input_tensor-scaler_mean)/scaler_std
        warmup_tensor = (input_tensor-scaler_mean)/scaler_std
    
    def do_warmup():
        h_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()
        c_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()
        for t in range(istart, warmup_tensor.shape[0]):
            with torch.no_grad():
                input_layer = warmup_tensor[t-seq_length:t, :]
                output, (h_t, c_t) = model(input_layer, h_t, c_t)
                h_t = h_t.transpose(0,1)
                c_t = c_t.transpose(0,1)
                if t == istart-1:
                    h_t_np = h_t[0,0,:].numpy()
                    h_t_df = pd.DataFrame(h_t_np)
                    h_t_df.to_csv(self.h_t_init_file)
                    c_t_np = c_t[0,0,:].numpy()
                    c_t_df = pd.DataFrame(c_t_np)
                    c_t_df.to_csv(self.c_t_init_file)

    def read_initial_states():
        h_t = np.genfromtxt(self.h_t_init_file, skip_header=1, delimiter=",")[:,1]
        h_t = torch.tensor(h_t).view(1,1,-1)
        c_t = np.genfromtxt(self.c_t_init_file, skip_header=1, delimiter=",")[:,1]
        c_t = torch.tensor(c_t).view(1,1,-1)
