import numpy as np
import pandas as pd
import random
from datetime import datetime



def get_temperature(s_effect):
    #temperature=pd.read_csv('/Users/can/Desktop/energy/code/solar_data/boston/2784951_42.38_-71.13_2018.csv')
    temperature=pd.read_csv('/home/erhan/energy_cdc_github/energy/weather_data/2784951_42.38_-71.13_2018.csv')
    temperature=temperature.iloc[:,[8]]
    temperature.columns=np.array(["temperature"])




    temperature=temperature.drop([0,1])
    temperature['temperature'] = temperature['temperature'].astype(float)



    update_time_periods=pd.date_range("2018-01-01", "2019-01-01", freq="5min")
    update_time_periods=update_time_periods[:len(update_time_periods)-1]
    temperature.index=update_time_periods


    adjust_time_resolution=int(15/5)
    #print(adjust_time_resolution)
    time_tmp=np.arange(0,len(update_time_periods)-1,adjust_time_resolution)
    temperature=temperature.iloc[time_tmp,]




    if s_effect==1:
        dt_obj = "01/01/2018"
    else:
        dt_obj = "08/07/2018"
    dt_obj_date=datetime.strptime(dt_obj, '%m/%d/%Y')
    start_index=np.where(dt_obj_date.date()==temperature.index.date)[0][0]




    outside_temp=temperature.iloc[start_index:start_index+96,0].values
    return outside_temp

def initialize_appliance_property(home,s_effect=1,rng=None):
    #s_effect =1 during winter, s_effect= -1 during summer
    home.hvac.s_effect=s_effect
    
    
    
    """
                            EWH
    --------------------------------------------------------                        
    TODO
        change tap water temperature from season to season.
        
    1.https://www.energy.gov/energysaver/sizing-new-water-heater
      capacity is 270 kg on average.
    2.efficiency is randomly selected.
    3.water amount indicates available hot water. Initially full
    4.tap_water_temp is 4 celcius during winter 10 celcius during summer.
        s_effect=1 during summer, -1 during winter.
        Note that s_effect is hvac parameter!
    5.des_water_temp takes random value between 40 and 42.
    """
    
    #ewh properties
    home.ewh.nominal_power=4#Kw
    home.ewh.capacity=270#kg
    home.ewh.efficiency=0.95
    
    #home specific prooerties
    home.water_amount=0.0#
    #home.water_amount=home.ewh.capacity/1.5
    home.tap_water_temp=4
    #home.des_water_temp=random.randint(40, 42)
    home.des_water_temp=rng.integers(low=40, high=42)
    
    
    
    
    """
                            EV
    --------------------------------------------------------  
    1.https://ev-database.org/cheatsheet/useable-battery-capacity-electric-car
      average capacity is 60kwH
    
    2.https://insideevs.com/news/343162/how-many-amps-does-your-home-charging-station-really-need/
      Most electric and plug-in hybrid vehicles available today can only accept 
      a maximum of 16 to 32-amps, while charging on a level 2, 240-volt charging station.
    
      nominal power depends on ev_current, is calculated in load_curve function of ev.
    3.ev_battery indicates how much charge exists in battery currently.
    """
    #ev properties
    home.ev.capacity=60#KWh
    home.ev.capacity=home.ev.capacity*3600#kwh to kw conversion
    
    #home specific property.
    home.ev_current=24#Amper
    home.ev_battery=0.0#KwH (Initially the battery is fully charged)
    #home.ev_battery=home.ev.capacity/1.5#KwH (Initially the battery is fully charged)
    
    ##hvac
    """
                            HVAC
    -------------------------------------------------------- 
    #TODO
        introduce different insulation levels.
        introduce different home size option.
        add randomness to temperature of each house.
    1. Need to initialize init temperature.
    2. Get outside temperatures for next 24hours. (1*96 array)
    3. gamma1 governs heat exchange between inside outside.
    4. gamma2 governs heat change because of HVAC.
    5. nominal power denotes the maximum capacity of heater.
    6. s_effect control heating season.
        s_effect=1 during summer, -1 during winter.
    """
    
    
    #hvac properties
    if s_effect==1:
        home.hvac.nominal_power=3#Kw
    elif s_effect==-1:
        home.hvac.nominal_power=2#Kw
    home.hvac.efficiency=0.9
    
    #home specific property.
    #home.gamma1=np.random.normal(0.10,0.001)
    #home.gamma2=np.random.normal(0.0000032,0.0000001)
    home.gamma1=rng.normal(0.10,0.001)
    home.gamma2=rng.normal(0.0000032,0.0000001)
    home.init_temp=23.0#random.randint(22, 24)#degrees initial room temperature.
    
    home.outside_temp=get_temperature(s_effect)
    

    
    ##refrigerator
    """#nominal power consumption is fixed across all houses"""
    
    
    ##oven
    """#nominal power consumption is fixed across all houses"""
    
    
    ##wm
    """#nominal power consumption is fixed across all houses"""
    
    
    ##dryer
    """#nominal power consumption is fixed across all houses"""
    
    
        
    """
                            PV
    --------------------------------------------------------  
    """
    #home.pv.efficiency=0.2
    #home.pv.m_square=10.0
    home.init_energy=0.0
    
    
    

    
    return home
    #raise NotImplemented

def old_initialize_appliance_property(home,s_effect=1,rng=None):
    #s_effect =1 during winter, s_effect= -1 during summer
    home.hvac.s_effect=s_effect
    
    
    
    """
                            EWH
    --------------------------------------------------------                        
    TODO
        change tap water temperature from season to season.
        
    1.https://www.energy.gov/energysaver/sizing-new-water-heater
      capacity is 270 kg on average.
    2.efficiency is randomly selected.
    3.water amount indicates available hot water. Initially full
    4.tap_water_temp is 4 celcius during winter 10 celcius during summer.
        s_effect=1 during summer, -1 during winter.
        Note that s_effect is hvac parameter!
    5.des_water_temp takes random value between 40 and 42.
    """
    
    #ewh properties
    home.ewh.nominal_power=4#Kw
    home.ewh.capacity=270#kg
    home.ewh.efficiency=0.95
    
    #home specific prooerties
    home.water_amount=0.0#
    #home.water_amount=home.ewh.capacity/1.5
    home.tap_water_temp=4
    home.des_water_temp=random.randint(40, 42)
    
    
    
    
    """
                            EV
    --------------------------------------------------------  
    1.https://ev-database.org/cheatsheet/useable-battery-capacity-electric-car
      average capacity is 60kwH
    
    2.https://insideevs.com/news/343162/how-many-amps-does-your-home-charging-station-really-need/
      Most electric and plug-in hybrid vehicles available today can only accept 
      a maximum of 16 to 32-amps, while charging on a level 2, 240-volt charging station.
    
      nominal power depends on ev_current, is calculated in load_curve function of ev.
    3.ev_battery indicates how much charge exists in battery currently.
    """
    #ev properties
    home.ev.capacity=60#KWh
    home.ev.capacity=home.ev.capacity*3600#kwh to kw conversion
    
    #home specific property.
    home.ev_current=24#Amper
    home.ev_battery=0.0#KwH (Initially the battery is fully charged)
    #home.ev_battery=home.ev.capacity/1.5#KwH (Initially the battery is fully charged)
    
    ##hvac
    """
                            HVAC
    -------------------------------------------------------- 
    #TODO
        introduce different insulation levels.
        introduce different home size option.
        add randomness to temperature of each house.
    1. Need to initialize init temperature.
    2. Get outside temperatures for next 24hours. (1*96 array)
    3. gamma1 governs heat exchange between inside outside.
    4. gamma2 governs heat change because of HVAC.
    5. nominal power denotes the maximum capacity of heater.
    6. s_effect control heating season.
        s_effect=1 during summer, -1 during winter.
    """
    
    
    #hvac properties
    if s_effect==1:
        home.hvac.nominal_power=3#Kw
    elif s_effect==-1:
        home.hvac.nominal_power=2#Kw
    home.hvac.efficiency=0.9
    
    #home specific property.
    home.gamma1=np.random.normal(0.10,0.001)
    home.gamma2=np.random.normal(0.0000032,0.0000001)
    home.init_temp=23.0#random.randint(22, 24)#degrees initial room temperature.
    
    home.outside_temp=get_temperature(s_effect)
    

    
    ##refrigerator
    """#nominal power consumption is fixed across all houses"""
    
    
    ##oven
    """#nominal power consumption is fixed across all houses"""
    
    
    ##wm
    """#nominal power consumption is fixed across all houses"""
    
    
    ##dryer
    """#nominal power consumption is fixed across all houses"""
    
    
        
    """
                            PV
    --------------------------------------------------------  
    """
    #home.pv.efficiency=0.2
    #home.pv.m_square=10.0
    home.init_energy=0.0
    
    
    

    
    return home
    #raise NotImplemented
