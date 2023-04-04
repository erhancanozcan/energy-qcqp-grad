import numpy as np
import pandas as pd
import math
import copy

    
def adjuster_v2(adjust):
    
    #adjust=copy_steps_tmp
    tmp=adjust>1
    cntrl=np.sum(tmp)
    
    if cntrl>0:
        #need to schedule in the next period.
        index_list=np.where(tmp)[0]
        remainder=adjust[index_list[0]]-1
        adjust[index_list[0]]=1
        if(index_list[0]+1)!=len(adjust):
            adjust[index_list[0]+1]=adjust[index_list[0]+1]+remainder
        return adjuster_v2(adjust)
    else:
        #print(adjust)
        return adjust
    #return adjust


class ewh:
    
    def __init__(self,time_res=15):
        
        """
        nominal_power depends on ewh type.
        capacity depends on ewh type.
        efficiency depends on ewh type.
        TODO
            those will be initialized by outside function.
        """
        self.time_r=time_res
        self.nominal_power=None
        self.capacity=None
        self.efficiency=None
        
    def load_curve(self,demand,tap_water_temp,desired_temp):
        water_heat_capacity=4186#J/(KG*°C)
        
        #remove after debugging.
        #desired_temp=42
        #tap_water_temp=4
        #required_energy=demand*water_heat_capacity*(desired_temp-tap_water_temp)
        #nominal_power=4000
        #required_minutes_to_charge=required_energy/(nominal_power*60)
        #time_res=15
        #steps_tmp=np.ceil(required_minutes_to_charge/time_res)
        #steps_tmp=adjuster(steps_tmp) 
        #steps_tmp=(required_minutes_to_charge/time_res)
    
        
        required_energy=demand*water_heat_capacity*(desired_temp-tap_water_temp)
        required_minutes_to_charge=required_energy/(self.nominal_power*1000*self.efficiency*60)
        #steps_tmp=np.ceil(required_minutes_to_charge/self.time_r)
        steps_tmp=(required_minutes_to_charge/self.time_r)
        copy_steps_tmp=copy.deepcopy(steps_tmp)
        #deneme=adjuster(copy_steps_tmp) 
        
        deneme=adjuster_v2(copy_steps_tmp) 
        
        #return deneme*self.nominal_power
        #raise NotImplemented
        
        #estimated_demand=copy.deepcopy(deneme*self.nominal_power)
        estimated_demand=copy.deepcopy(deneme)
        
        summ=0
        for i in range(len(estimated_demand)):
            summ+=estimated_demand[i]
            if summ>0 and estimated_demand[i]==0:
                estimated_demand[i]=summ
                summ=0
            else:
                estimated_demand[i]=0
        #estimated_demand=estimated_demand/(average_energy*(3600/(self.time_r*60)))

        #estimated_demand=estimated_demand*1000*15/(water_heat_capacity*self.efficiency)
        estimated_demand=estimated_demand*self.nominal_power*60*15*1000*self.efficiency/(water_heat_capacity*(desired_temp-tap_water_temp))
        return estimated_demand,deneme*self.nominal_power
    

