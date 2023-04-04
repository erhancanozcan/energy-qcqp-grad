#import matplotlib.pyplot as plt
import numpy as np
import copy


def home_power_plot(home,real_power,price,plot_type="total",plot=True):
    
    horizon=len(home.hvac_desirable_load)
    if plot_type=="total":
        desirable_power=home.ewh_desirable_load+home.ev_desirable_load+\
            home.hvac_desirable_load+home.refrigerator_desirable_load+\
            home.oven_desirable_load+home.wm_desirable_load+\
            home.dryer_desirable_load
            
            
        real_power_tmp=real_power['ewh']+real_power['ev']+\
                         real_power['hvac']+real_power['refrigerator']+\
                         real_power['oven']+real_power['wm']+\
                         real_power['dryer']
    else:
        real_power_tmp=copy.deepcopy(real_power[plot_type])
        if plot_type=="ewh":
            desirable_power=home.ewh_desirable_load
        elif plot_type=="ev":
            desirable_power=home.ev_desirable_load
        elif plot_type=="hvac":
            desirable_power=home.hvac_desirable_load
        elif plot_type=="refrigerator":
            desirable_power=home.refrigerator_desirable_load
        elif plot_type=="oven":
            desirable_power=home.oven_desirable_load
        elif plot_type=="wm":
            desirable_power=home.wm_desirable_load
        elif plot_type=="dryer":
            desirable_power=home.dryer_desirable_load

        
        
        
                         
    if plot==True:
        
        fig, ax = plt.subplots()
        ax.step(np.arange(horizon),real_power_tmp,label="real power")
        ax.step(np.arange(horizon),desirable_power,label="desirable power",alpha=0.7)
        #ax.legend(loc="upper right",["real power","desirable power"])
        ax.legend(loc="upper right")
        ax.set_title("Load Comparison per Interval  ("+plot_type.upper()+")")
        ax.set_ylabel("Kw")
        ax.set_ylim((0,12))
        ax.set_xlabel("Time Index")
        fig
            
    #note that price is in terms of kwh.
    fee_desirable_load=np.dot(desirable_power,price)*24/horizon#in terms of Kwh.
    fee_real_load= np.dot(real_power_tmp,price)*24/horizon#in terms of Kwh.
    
    real_power_tmp=np.sum(real_power_tmp)*24/horizon#in terms of Kwh.
    desirable_power=np.sum(desirable_power)*24/horizon#in terms of Kwh.
    
    print("Daily fee ("+plot_type+ ") desirable load: %.2f" %fee_desirable_load)
    print("Daily fee ("+plot_type+ ") real load: %.2f" %fee_real_load)
    print(plot_type+ " desirable load: %.2f" %desirable_power)
    print(plot_type+ " real load: %.2f" %real_power_tmp)
    
    return fee_desirable_load,fee_real_load,real_power_tmp,desirable_power


def state_plot(home,states,plot_type):
    
    
    
    horizon=len(home.hvac_desirable_load)
    state=states[plot_type]
    
    fig, ax = plt.subplots()
    ax.set_title("State of Appliance per Interval  ("+plot_type.upper()+")")
    if plot_type=="hvac":
        ax.plot(state,label="Inside Temperature")
        ax.set_ylabel("Inside Temperature (Celcius)")
        ax.plot(home.set_temperature,label="Set Temperature")
        
        lower=home.set_temperature-home.deadband
        upper=home.set_temperature+home.deadband
        ax.fill_between(np.arange(horizon), lower, upper, color='orange', alpha=.2)
        ax.legend(loc="upper right")

        
        
    elif plot_type=="ev":
        ax.plot(state/3600)
        ax.set_ylabel("Available Charge in Battery (Kwh)")
        required=home.ev_demand*0.346#KwH per mile
        ax.step(np.arange(horizon),-required)
    elif plot_type=="ewh":
        ax.plot(state)
        ax.set_ylabel("Available Hot Water (KG)")
        required=home.ewh_demand
        idx=np.where(required>0)[0]
        ax.scatter(idx,-required[idx],marker='o',c="red")
    else:
        #print(state)
        ax.plot(state)
        ax.set_ylabel("Available stored energy in PV (KwH)")
        #required=home.ewh_demand
        #idx=np.where(required>0)[0]
        #ax.scatter(idx,-required[idx],marker='o',c="red")
    
    ax.set_xlabel("Time Index")
    fig
        
            
                         
                         
                         
        
    
    