from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.tracking import SingleAxisTracker
from pvlib.modelchain import ModelChain
from pvlib.forecast import GFS

import numpy as np
from datetime import datetime
import math
import pandas as pd



def get_irradiation_date(s_effect):
    
    #possible input: give dt_obj as input

    #solar_irradiation_data=pd.read_csv('/Users/can/Documents/GitHub/EnergySimulation/data/2784951_42.38_-71.13_2018.csv')
    solar_irradiation_data=pd.read_csv('/home/erhan/energy_cdc_github/energy/weather_data/2784951_42.38_-71.13_2018.csv')
    solar_irradiation_data=solar_irradiation_data.iloc[:,[5,6,7]]
    solar_irradiation_data.columns=np.char.lower(solar_irradiation_data.iloc[1,:].values.astype('<U5'))
    solar_irradiation_data=solar_irradiation_data.drop([0,1])
    solar_irradiation_data['ghi'] = solar_irradiation_data['ghi'].astype(float)
    solar_irradiation_data['dhi'] = solar_irradiation_data['dhi'].astype(float)
    solar_irradiation_data['dni'] = solar_irradiation_data['dni'].astype(float)
    
    
    update_time_periods=pd.date_range("2018-01-01", "2019-01-01", freq="5min")
    update_time_periods=update_time_periods[:len(update_time_periods)-1]
    solar_irradiation_data.index=update_time_periods
    
    if s_effect==1:
        dt_obj = "01/01/2018"
    else:
        #dt_obj = "01/01/2018"
        dt_obj = "08/07/2018"
    dt_obj=datetime.strptime(dt_obj, '%m/%d/%Y')
    start_index=np.where(dt_obj.date()==solar_irradiation_data.index.date)[0][0]
    #solar_irradiation_data=solar_irradiation_data.iloc[start_index:,:].values
    solar_irradiation_data=solar_irradiation_data.iloc[start_index:,:]
    
    #this part create randomness for each house in solar data
    zeros=(solar_irradiation_data==0)
    #tmp=np.random.normal(0,20,solar_irradiation_data.shape)
    tmp=np.ones(solar_irradiation_data.shape)
    tmp[zeros]=0
    solar_irradiation_data=solar_irradiation_data+tmp
    solar_irradiation_data=abs(solar_irradiation_data)
    #solar_irradiation_data[solar_irradiation_data<0]=0
    ind=np.arange(0,288,3)
    solar_irradiation_data=solar_irradiation_data.iloc[ind,:]
    
    return solar_irradiation_data


class pv:
    def __init__(self,time_res=15,s_effect=1):
        self.time_r=time_res
        #initialize those by appliances.
        self.efficiency=0.2
        self.m_square=10
        
                
        latitude, longitude, tz = 40.7, -74.0, 'America/New_York'
        #latitude, longitude, tz = -27.5, 153.0, 'Australia/Queensland'
        
        start = pd.Timestamp(datetime.now(), tz=tz)
        
        end = start + pd.Timedelta(hours=24)
        
        irrad_vars = ['ghi', 'dni', 'dhi']
        
        
        sandia_modules = retrieve_sam('sandiamod')

        cec_inverters = retrieve_sam('cecinverter')

        module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

        inverter = cec_inverters['SMA_America__SC630CP_US__with_ABB_EcoDry_Ultra_transformer_']

        temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        # model a big tracker for more fun
        # system = SingleAxisTracker(module_parameters=module, inverter_parameters=inverter, temperature_model_parameters=temperature_model_parameters, modules_per_string=15, strings_per_inverter=300)



        # fx_model = GFS()

        # fx_data = fx_model.get_processed_data(latitude, longitude, start, end)
        
        # new_fx_data=get_irradiation_date(s_effect)
        # new_fx_data=new_fx_data.iloc[:int(24*60/time_res),]
        # mc = ModelChain(system, fx_model.location)
        # mc.run_model(new_fx_data)
        
        # #we can add randomness here maybe?
        # self.stored_energy=mc.results.total_irrad['poa_global']*self.m_square*self.efficiency
        # self.stored_energy=self.stored_energy/1000
        # self.stored_energy=self.stored_energy*(60*self.time_r/3600)
        # self.stored_energy= self.stored_energy.values
        
        self.stored_energy = np.zeros(int(24*60/self.time_r))
        
        
        
        
        
        
        
        
    
