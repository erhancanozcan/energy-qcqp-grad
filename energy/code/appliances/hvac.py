import numpy as np
import pandas as pd
import math
import copy
from gurobipy import *
#import matplotlib.pyplot as plt



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
        room_temp=init_temp
        self.T_out=T_out
        
        pred=[None]*(horizon)
        s_HVAC_neg=[None]*(horizon)
        s_HVAC_pos=[None]*(horizon)
        P_HVAC=[None]*(horizon)
        
        
        for i in range(horizon):
            
            if self.s_effect == 1:
                if room_temp < set_temperature[i]- deadband :
                    
                    pred[i]= room_temp+gamma1*(T_out[i]-room_temp)+gamma2*self.nominal_power*self.efficiency*(1000*60*self.time_r)*self.s_effect
                    load=self.nominal_power
                else:
                    pred[i]= room_temp+gamma1*(T_out[i]-room_temp)
                    load=0.0
                    
                s_neg=abs(pred[i]-(set_temperature[i]- deadband)) if pred[i]<(set_temperature[i]- deadband) else 0.0
                s_pos=abs(pred[i]-(set_temperature[i]+ deadband)) if pred[i]>(set_temperature[i]+ deadband) else 0.0
            elif self.s_effect == -1:
                if room_temp > set_temperature[i]+ deadband :
                    pred[i]= room_temp+gamma1*(T_out[i]-room_temp)+gamma2*self.nominal_power*self.efficiency*(1000*60*self.time_r)*self.s_effect
                    load=self.nominal_power
                else:
                    pred[i]= room_temp+gamma1*(T_out[i]-room_temp)
                    load=0.0
                s_neg=abs(pred[i]-(set_temperature[i]- deadband)) if pred[i]<(set_temperature[i]- deadband) else 0.0
                s_pos=abs(pred[i]-(set_temperature[i]+ deadband)) if pred[i]>(set_temperature[i]+ deadband) else 0.0
            else:
                raise Exception("season effect must be either winter (1) or summer(-1)")
                
            
            s_HVAC_neg[i]=s_neg
            s_HVAC_pos[i]=s_pos
            room_temp=pred[i]
            P_HVAC[i]=load
            
        
        """
        Temperature and Power related plots
        """
        # fig, ax = plt.subplots()
        # ax.plot(pred,label="Inside Temperature")
        # ax.set_ylabel("Inside Temperature (Celcius)")
        # ax.plot(set_temperature,label="Set Temperature")
        # lower=set_temperature-deadband
        # upper=set_temperature+deadband
        # ax.fill_between(np.arange(horizon), lower, upper, color='orange', alpha=.2)
        # ax.legend(loc="upper right")
        
        # fig, ax = plt.subplots()
        # ax.step(np.arange(horizon),P_HVAC,label="Power Consumption")
        # ax.set_ylabel("Kw")
        
        return P_HVAC,s_HVAC_neg,s_HVAC_pos
            
            
                
                
                
                
        
        
        
        
        
        
        # m = Model("desirableHVAC")
        # s_HVAC_neg=m.addVars(horizon,lb=0)
        # s_HVAC_pos=m.addVars(horizon,lb=0)
        # P_HVAC=m.addVars(horizon,lb=0,ub=self.nominal_power)
        # T_in=m.addVars(horizon+1)
        
        
        # neg_dev=quicksum(s_HVAC_neg[i]*1 for i in range(horizon))
        # pos_dev=quicksum(s_HVAC_pos[i]*1 for i in range(horizon))
        # m.setObjective(neg_dev+pos_dev, GRB.MINIMIZE)
        
        # const_temp_change=m.addConstrs((T_in[i+1]==T_in[i]+gamma1*(T_out[i]-T_in[i])+gamma2*P_HVAC[i]*self.efficiency*(1000*60*self.time_r)*self.s_effect
        #                                      for i in range(horizon)),name='c_hvac_temp_chng')


        # const_temp_cntrl_low=m.addConstrs((T_in[i+1]>=set_temperature[i]-deadband-s_HVAC_neg[i]
        #                                      for i in range(horizon)),name='c_temp_low')

        # const_temp_cntrl_up=m.addConstrs((T_in[i+1]<=set_temperature[i]+deadband+s_HVAC_pos[i]
        #                                      for i in range(horizon)),name='c_temp_up')

        # const_init_temp=m.addConstr(T_in[0]==init_temp,name='c_wm_balance')

        # m.write('/Users/can/Desktop/energy/code/desirableHVAC.lp')
        
        
        
        # m.Params.Method=3
        # m.optimize()

        # desirable_HVAC_load=np.zeros(horizon)
        # next_room_temperatures=np.zeros(horizon)
        # neg_devs=np.zeros(horizon)
        # pos_devs=np.zeros(horizon)

        # for i in range(horizon):
        #     #desirable_HVAC_load[i]=P_HVAC[i].X/(self.time_r*60) #15 min * 60 seconds
        #     desirable_HVAC_load[i]=P_HVAC[i].X
            
        # for i in range(horizon):
        #     neg_devs[i]=s_HVAC_neg[i].X
            
        # for i in range(horizon):
        #     pos_devs[i]=s_HVAC_pos[i].X
            
        # for i in range(horizon):
        #     next_room_temperatures[i]=T_in[i+1].X
        # #print(next_room_temperatures)   
        #return desirable_HVAC_load,neg_devs,pos_devs
        
        #raise NotImplemented
