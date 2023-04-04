import numpy as np
import pandas as pd
import math
import copy



class ev:
    
    def __init__(self,time_res=15):
        
        """
        capacity depends on ev type.
        nominal_power depends on ev type.
        TODO
            capacity will be initialized by outside function.
        """
        self.time_r=time_res
        self.nominal_power=None#(240V*24I)/1000 = Kw. 
        self.capacity=None
        
    def load_curve(self,demand,current):
        self.nominal_power=(240*current)/1000
        average_energy=0.346#KwH per mile
        #average_energy=average_energy*3600 #kwh to kw conversion.
        #average_energy=0.346*(3600/(self.time_r*60))#KwH per mile
        
        non_chargable_periods=copy.deepcopy(demand)
        cum_demand=np.cumsum(demand)
        non_chargable_periods=demand!=0
        cum_demand[non_chargable_periods]=0



        cumulative_energy=cum_demand*average_energy
        
        power=np.zeros(len(demand))

        for i in range(len(cumulative_energy)):
            energy_i=cumulative_energy[i]
            if energy_i>0:
                energy_i=min(energy_i*(3600/(self.time_r*60)),self.nominal_power)
                power[i]=energy_i
                cumulative_energy=cumulative_energy-energy_i/(3600/(self.time_r*60))
        
        #please see that 
        #np.sum(demand)*0.346 equals to np.sum(power)/4
        #return power*(3600/(60*self.time_r))
        #return power
        #raise NotImplemented

        estimated_demand=copy.deepcopy(power)
        
        summ=0
        for i in range(len(estimated_demand)):
            summ+=estimated_demand[i]
            if summ>0 and estimated_demand[i]==0:
                estimated_demand[i]=summ
                summ=0
            else:
                estimated_demand[i]=0
        estimated_demand=estimated_demand/(average_energy*(3600/(self.time_r*60)))
        
        
        indices=np.where(estimated_demand>0)[0]
        miles_in_15_minutes=demand[np.where(demand>0)[0][0]]
        for ind in indices:
            #divided by six because we assume that each demand is 6 miles in ev, which is a poor
            #estimate
            #miles_in_15_minutes=6
            end=ind+int(np.round(estimated_demand[ind]/miles_in_15_minutes))
            estimated_demand[ind:end]=miles_in_15_minutes
        
        return estimated_demand,power
        



    


