import numpy as np
import pandas as pd



class other_appliances:
    
    def __init__(self,time_res=15):
        
        """
        capacity depends on ev type.
        nominal_power depends on heating season
        
        nominal power is initialized by outside function. However,
        appliances are more or less same.
        """
        self.time_r=time_res

        
    def load_curve(self,demand):
        return demand*self.nominal_power
        #raise NotImplemented
        
        
class wm(other_appliances):
    
    def __init__(self, time_res=15):
        
        super(wm,self).__init__(time_res)
        #https://www.directenergy.com/learning-center/how-much-energy-washing-machine-use
        self.nominal_power = 0.5
        #https://www.whirlpool.com/blog/washers-and-dryers/how-long-does-a-washing-machine-take-to-wash-clothes.html
        self.duration=60#min
        
    def load_curve(self,demand):
        return demand*self.nominal_power
        
        
class dryer(other_appliances):
    
    def __init__(self, time_res=15):
        
        super(dryer,self).__init__(time_res)
        #Dryers are typically somewhere in the range of 2,000 to 6,000 watts
        #https://www.directenergy.com/learning-center/how-much-energy-dryer-use
        self.nominal_power = 4
        self.duration=60#min

    def load_curve(self,demand):
        return demand*self.nominal_power
        
class oven(other_appliances):
    
    def __init__(self, time_res=15):
        
        super(oven,self).__init__(time_res)
        #Most electric ovens draw between 2,000 and 5,000 watts
        #https://www.directenergy.com/learning-center/how-much-energy-does-oven-and-electric-stove-use
        self.nominal_power = 3.5
        self.duration=60
        
    def load_curve(self,demand):
        return demand*self.nominal_power

        
class refrigerator(other_appliances):
    
    def __init__(self, time_res=15):
        
        super(refrigerator,self).__init__(time_res)
        #Domestic fridge power consumption is typically between 100 and 250 watts
        self.nominal_power = 0.2
        
    def load_curve(self,demand):
        return demand*self.nominal_power

        
        
        
        
