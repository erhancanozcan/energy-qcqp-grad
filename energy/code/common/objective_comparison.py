import numpy as np
import pandas as pd
from gurobipy import *
from energy.code.common.home_change_objective import change_objective_solve




def check_objectives_with_price(price,home_reals,horizon,num_homes,power_HVAC_list,dual_list,models,rng,desirable_power_list):
    

    
    for i in range (num_homes):
        
        home_real_power=np.zeros((6,horizon))
        
        home_real_power[0,:]=np.array([home_reals[i][0*horizon+k].X for k in range(horizon)])
        home_real_power[1,:]=np.array([home_reals[i][1*horizon+k].X for k in range(horizon)])
        home_real_power[2,:]=np.array([home_reals[i][2*horizon+k].X for k in range(horizon)])
        home_real_power[3,:]=np.array([home_reals[i][3*horizon+k].X * power_HVAC_list[i] for k in range(horizon)])
        home_real_power[4,:]=np.array([home_reals[i][4*horizon+k].X for k in range(horizon)])
        home_real_power[5,:]=np.array([home_reals[i][5*horizon+k].X for k in range(horizon)])
        
        
        #home_real_power=np.array([home_reals[i][0*horizon+k].X for k in range(horizon)])+\
        #     np.array([home_reals[i][1*horizon+k].X for k in range(horizon)])+\
        #     np.array([home_reals[i][2*horizon+k].X for k in range(horizon)])+\
        #     np.array([home_reals[i][3*horizon+k].X * power_HVAC_list[i] for k in range(horizon)])+\
        #     np.array([home_reals[i][4*horizon+k].X for k in range(horizon)])+\
        #     np.array([home_reals[i][5*horizon+k].X for k in range(horizon)])

        
        electricity_price=np.dot(np.sum(home_real_power,axis=0),price)
        electricity_price=np.sum(electricity_price)
        
        
        cost=dual_list[i]['c']
        dev_cost=cost[:6*horizon]
        #dev_cost=dev_cost.reshape(6,horizon)
        
        des_power=desirable_power_list[i]
        dev_cost_total=np.dot(abs(home_real_power-des_power).flatten(),dev_cost)
        
        
        calculated_obj=electricity_price+dev_cost_total
        
        
        ### calculate the optimal response and objective.
        
        cost=dual_list[i]['c']
        size_of_cost=len(cost)
        
        new_cost=cost[:6*horizon]
        new_cost=np.concatenate((new_cost,price))#wm
        new_cost=np.concatenate((new_cost,price))#oven
        new_cost=np.concatenate((new_cost,price))#dryer
        new_cost=np.concatenate((new_cost,price*power_HVAC_list[i]))#hvac
        new_cost=np.concatenate((new_cost,price))#ewh
        new_cost=np.concatenate((new_cost,price))#ev
        new_cost=np.concatenate((new_cost,-price))#pv
        
        new_cost=np.concatenate((new_cost,np.repeat(0,size_of_cost-len(new_cost))))#cost is zero for other vars.
        
        m=change_objective_solve(models[i].copy(),new_cost)
        
        optimal_objective=m.objVal
        
        print(calculated_obj-optimal_objective)





def check_objectives(mean_vec,cov_matrix,horizon,num_homes,power_HVAC_list,dual_list,models,rng,desirable_power_list):
    
    sampled_sol = rng.multivariate_normal(mean=mean_vec, cov=cov_matrix, size=1)[0,:]
    #price=mean_vec[-horizon:]
    price=sampled_sol[-horizon:]
    
    sampled_real_power=sampled_sol[:len(sampled_sol)-horizon]
    sampled_real_power=sampled_real_power.reshape(num_homes,int(len(sampled_real_power)/num_homes))
    
    for i in range (num_homes):
        

        i_sampled_real_power=sampled_real_power[i,:]
        i_sampled_real_power=i_sampled_real_power.reshape(int(len(i_sampled_real_power)/horizon),horizon)
        
        
        
        #any modifications?
        i_sampled_real_power[3,:]=i_sampled_real_power[3,:]*power_HVAC_list[i]
        i_sampled_real_power[6,:]=0
        
        electricity_price=np.dot(i_sampled_real_power,price)
        electricity_price=np.sum(electricity_price)
        
        
        cost=dual_list[i]['c']
        dev_cost=cost[:6*horizon]
        #dev_cost=dev_cost.reshape(6,horizon)
        
        des_power=desirable_power_list[i]
        dev_cost_total=np.dot(abs(i_sampled_real_power[:6,:]-des_power).flatten(),dev_cost)
        
        
        calculated_obj=electricity_price+dev_cost_total
        
        
        ### calculate the optimal response and objective.
        
        cost=dual_list[i]['c']
        size_of_cost=len(cost)
        
        new_cost=cost[:6*horizon]
        new_cost=np.concatenate((new_cost,price))#wm
        new_cost=np.concatenate((new_cost,price))#oven
        new_cost=np.concatenate((new_cost,price))#dryer
        new_cost=np.concatenate((new_cost,price*power_HVAC_list[i]))#hvac
        new_cost=np.concatenate((new_cost,price))#ewh
        new_cost=np.concatenate((new_cost,price))#ev
        new_cost=np.concatenate((new_cost,-price))#pv
        
        new_cost=np.concatenate((new_cost,np.repeat(0,size_of_cost-len(new_cost))))#cost is zero for other vars.
        
        m=change_objective_solve(models[i].copy(),new_cost)
        
        optimal_objective=m.objVal
        
        print(calculated_obj-optimal_objective)
        
        
        
        
        
        
        
