import numpy as np
import pandas as pd
import math
import copy
from gurobipy import *



class hvac:
    
    def __init__(self,time_res=15):
        
        """
        capacity depends on ev type.
        nominal_power depends on heating season
        TODO
            nominal power will be initialized by outside function.
        """
        self.time_r=time_res
        self.nominal_power=None
        self.efficiency=None
        
    def load_curve(self,set_temperature,deadband,T_out,init_temp,gamma1,gamma2):
        #eff_capacity=self.nominal_power*self.efficiency
        horizon=len(set_temperature)
        self.T_out=T_out
        
        
        m = Model("desirableHVAC")
        s_HVAC_neg=m.addVars(horizon,lb=0)
        s_HVAC_pos=m.addVars(horizon,lb=0)
        P_HVAC=m.addVars(horizon,lb=0,ub=self.nominal_power)
        T_in=m.addVars(horizon+1)
        
        
        neg_dev=quicksum(s_HVAC_neg[i]*1 for i in range(horizon))
        pos_dev=quicksum(s_HVAC_pos[i]*1 for i in range(horizon))
        m.setObjective(neg_dev+pos_dev, GRB.MINIMIZE)
        
        const_temp_change=m.addConstrs((T_in[i+1]==T_in[i]+gamma1*(T_out[i]-T_in[i])+gamma2*P_HVAC[i]*self.efficiency*(1000*60*self.time_r)*self.s_effect
                                             for i in range(horizon)),name='c_hvac_temp_chng')


        const_temp_cntrl_low=m.addConstrs((T_in[i+1]>=set_temperature[i]-deadband-s_HVAC_neg[i]
                                             for i in range(horizon)),name='c_temp_low')

        const_temp_cntrl_up=m.addConstrs((T_in[i+1]<=set_temperature[i]+deadband+s_HVAC_pos[i]
                                             for i in range(horizon)),name='c_temp_up')

        const_init_temp=m.addConstr(T_in[0]==init_temp,name='c_wm_balance')

        m.write('/Users/can/Desktop/energy/code/desirableHVAC.lp')
        
        
        
        m.Params.Method=3
        m.optimize()

        desirable_HVAC_load=np.zeros(horizon)
        next_room_temperatures=np.zeros(horizon)
        neg_devs=np.zeros(horizon)
        pos_devs=np.zeros(horizon)

        for i in range(horizon):
            #desirable_HVAC_load[i]=P_HVAC[i].X/(self.time_r*60) #15 min * 60 seconds
            desirable_HVAC_load[i]=P_HVAC[i].X
            
        for i in range(horizon):
            neg_devs[i]=s_HVAC_neg[i].X
            
        for i in range(horizon):
            pos_devs[i]=s_HVAC_pos[i].X
            
        for i in range(horizon):
            next_room_temperatures[i]=T_in[i+1].X
        #print(next_room_temperatures)   
        return desirable_HVAC_load,neg_devs,pos_devs
        
        #raise NotImplemented
