import numpy as np
import pandas as pd
from gurobipy import *
from energy.code.common.home_change_objective import change_objective_solve




def check_random_feasibility(mean_vec,cov_matrix,horizon,num_homes,power_HVAC_list,dual_list,models,rng,desirable_power_list):
    
    
    sampled_sol = rng.multivariate_normal(mean=mean_vec, cov=cov_matrix, size=1)[0,:]
    #price=mean_vec[-horizon:]
    price=sampled_sol[-horizon:]
    
    sampled_real_power=sampled_sol[:len(sampled_sol)-horizon]
    sampled_real_power=sampled_real_power.reshape(num_homes,int(len(sampled_real_power)/num_homes))
    
    
    found_feasible=True
    for i in range (num_homes):
        print(i)
        i_real_power=sampled_real_power[i,:]
        i_real_power=i_real_power.reshape(int(len(i_real_power)/horizon),horizon)
        
        #np.sum(i_real_power,axis=1)

        m=models[i].copy()
        
        real_p_s_idx=6*horizon
        m.addConstrs((m.getVars()[real_p_s_idx+t]==i_real_power[int(t/horizon),t%horizon]
                                              for t in range(real_p_s_idx)),name='feasibility_of_sampled')
        
        #  [m.getVars()[idx] for idx in range(len(m.getVars()))]
        m.update()

        m.Params.LogToConsole=0
        m.optimize()
        
        #https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html#sec:StatusCodes
        if m.Status !=2: #  status is equal to 2 when it is optimal.
            found_feasible=False
            break
    
    
    if found_feasible==True:
        print("feasible vector is found")


def check_mean_feasibility(mean_vec,cov_matrix,horizon,num_homes,power_HVAC_list,dual_list,models,rng,desirable_power_list):
    
    
    #sampled_sol = rng.multivariate_normal(mean=mean_vec, cov=cov_matrix, size=1)[0,:]
    #price=mean_vec[-horizon:]
    price=mean_vec[-horizon:]
    
    sampled_real_power=mean_vec[:len(mean_vec)-horizon]
    sampled_real_power=sampled_real_power.reshape(num_homes,int(len(sampled_real_power)/num_homes))
    
    
    found_feasible=True
    for i in range (num_homes):
        print(i)
        i_real_power=sampled_real_power[i,:]
        i_real_power=i_real_power.reshape(int(len(i_real_power)/horizon),horizon)
        
        #np.sum(i_real_power,axis=1)

        m=models[i].copy()
        
        real_p_s_idx=6*horizon
        m.addConstrs((m.getVars()[real_p_s_idx+t]==i_real_power[int(t/horizon),t%horizon]
                                              for t in range(real_p_s_idx)),name='feasibility_of_sampled')
        
        #  [m.getVars()[idx] for idx in range(len(m.getVars()))]
        m.update()

        m.Params.LogToConsole=0
        m.optimize()
        
        #https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html#sec:StatusCodes
        if m.Status !=2: #  status is equal to 2 when it is optimal.
            found_feasible=False
            break
    
    
    if found_feasible==True:
        print("feasible vector is found")      
        
        
        
        