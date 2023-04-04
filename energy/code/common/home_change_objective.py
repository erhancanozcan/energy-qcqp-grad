import numpy as np
import pandas as pd
from gurobipy import *


def change_objective_solve(model,obj_coeff):
    
    #we use this function in D-W decomposition with a linear objective.
    
    model.setObjective(quicksum(model.getVars()[i]*obj_coeff[i] for i in range(len(obj_coeff))))
    model.update()
    
    model.Params.LogToConsole=0
    model.Params.Method=0
    model.optimize()
    return model


def update_price_and_solve(price,dual_list,i,horizon,desirable_power_list,models,power_HVAC_list):

    #we use this function in GD based approach. Solve home mpc problem with
    #the updated price information.    

    power_HVAC=power_HVAC_list[i]
    
    #prepare dev_cost
    cost_v=dual_list[i]['dev_cost'][np.arange(0,6,1)*horizon]
    
    cost_u_wm=cost_v[0]
    cost_u_oven=cost_v[1]
    cost_u_dryer=cost_v[2]
    cost_u_hvac=cost_v[3]
    cost_u_ewh=cost_v[4]
    cost_u_ev=cost_v[5]
    
    #prepare desirable load
    wm_desirable_load=desirable_power_list[i][0,:]
    oven_desirable_load=desirable_power_list[i][1,:]
    dryer_desirable_load=desirable_power_list[i][2,:]
    hvac_desirable_load=desirable_power_list[i][3,:]
    ewh_desirable_load=desirable_power_list[i][4,:]
    ev_desirable_load=desirable_power_list[i][5,:]
    
    
    m=models[i]
    num_variables=len(m.getVars())
    
    
    #prepare decision variables
    P_wm=[]
    P_oven=[]
    P_dryer=[]
    P_HVAC=[]
    P_ewh=[]
    P_ev=[]
    
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("wm_real["+str(j)+"]")
    P_wm.append([m.getVarByName(name) for name in names_to_retrieve])
    
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("oven_real["+str(j)+"]")
    P_oven.append([m.getVarByName(name) for name in names_to_retrieve])
        
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("dryer_real["+str(j)+"]")
    P_dryer.append([m.getVarByName(name) for name in names_to_retrieve])
    
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("hvac_real["+str(j)+"]")
    P_HVAC.append([m.getVarByName(name) for name in names_to_retrieve])
    
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("ewh_real["+str(j)+"]")
    P_ewh.append([m.getVarByName(name) for name in names_to_retrieve])
        
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("ev_real["+str(j)+"]")
    P_ev.append([m.getVarByName(name) for name in names_to_retrieve])
    
    
    #prepare decision variables
    s_HVAC_neg=[]
    s_HVAC_pos=[]
    T_set=[]
    
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("s_neg["+str(j)+"]")
    s_HVAC_neg.append([m.getVarByName(name) for name in names_to_retrieve])
    
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("s_plus["+str(j)+"]")
    s_HVAC_pos.append([m.getVarByName(name) for name in names_to_retrieve])
    
    names_to_retrieve=[]
    for j in range(horizon):
        names_to_retrieve.append("t_set["+str(j)+"]")
    T_set.append([m.getVarByName(name) for name in names_to_retrieve])
      
    
    
    
    p_obj=quicksum((wm_desirable_load[i]-P_wm[0][i])*(wm_desirable_load[i]-P_wm[0][i])*cost_u_wm for i in range(horizon))+\
            quicksum((oven_desirable_load[i]-P_oven[0][i])*(oven_desirable_load[i]-P_oven[0][i])*cost_u_oven for i in range(horizon))+\
            quicksum((dryer_desirable_load[i]-P_dryer[0][i])*(dryer_desirable_load[i]-P_dryer[0][i])*cost_u_dryer for i in range(horizon))+\
            quicksum((hvac_desirable_load[i]-P_HVAC[0][i]*power_HVAC)*(hvac_desirable_load[i]-P_HVAC[0][i]*power_HVAC)*cost_u_hvac for i in range(horizon))+\
            quicksum((ewh_desirable_load[i]-P_ewh[0][i])*(ewh_desirable_load[i]-P_ewh[0][i])*cost_u_ewh for i in range(horizon))+\
            quicksum((ev_desirable_load[i]-P_ev[0][i])*(ev_desirable_load[i]-P_ev[0][i])*cost_u_ev for i in range(horizon))+\
            quicksum(P_wm[0][i]*price[i] for i in range(horizon))+\
            quicksum(P_oven[0][i]*price[i] for i in range(horizon))+\
            quicksum(P_dryer[0][i]*price[i] for i in range(horizon))+\
            quicksum(P_HVAC[0][i]*power_HVAC*price[i] for i in range(horizon))+\
            quicksum(P_ewh[0][i]*price[i] for i in range(horizon))+\
            quicksum(P_ev[0][i]*price[i] for i in range(horizon))+\
            0#quicksum(P_pv[i]*-price[i] for i in range(horizon))
    
    m.setObjective(p_obj,GRB.MINIMIZE)
    
    m.Params.Method=0
    m.Params.LogToConsole=0
    m.optimize()
    
    P_ewh_a=np.zeros(horizon)
    P_ev_a=np.zeros(horizon)
    P_hvac_a=np.zeros(horizon)
    P_oven_a=np.zeros(horizon)
    P_wm_a=np.zeros(horizon)
    P_dryer_a=np.zeros(horizon)
    P_pv_a=np.zeros(horizon)
    s_HVAC_neg_a=np.zeros(horizon)
    s_HVAC_pos_a=np.zeros(horizon)
    T_set_a=np.zeros(horizon)
    
    for t in range(horizon):
        P_ewh_a[t]=P_ewh[0][t].X
        P_ev_a[t]=P_ev[0][t].X
        P_hvac_a[t]=P_HVAC[0][t].X*power_HVAC
        P_oven_a[t]=P_oven[0][t].X
        P_wm_a[t]=P_wm[0][t].X
        P_dryer_a[t]=P_dryer[0][t].X
        #P_pv_a[t]=P_pv[i].X
        s_HVAC_neg_a[t]=s_HVAC_neg[0][t].X
        s_HVAC_pos_a[t]=s_HVAC_pos[0][t].X
        T_set_a[t]=T_set[0][t].X
        
        
        
        
        
    real_power_array=np.zeros((6,horizon))
    
    real_power_array[0,:]=P_wm_a
    real_power_array[1,:]=P_oven_a
    real_power_array[2,:]=P_dryer_a
    real_power_array[3,:]=P_hvac_a
    real_power_array[4,:]=P_ewh_a
    real_power_array[5,:]=P_ev_a
    
    
    all_values=np.zeros((int(num_variables/horizon),horizon))
    
    all_values[0,:]=P_wm_a
    all_values[1,:]=P_oven_a
    all_values[2,:]=P_dryer_a
    all_values[3,:]=P_hvac_a
    all_values[4,:]=P_ewh_a
    all_values[5,:]=P_ev_a
    all_values[6,:]=s_HVAC_neg_a
    all_values[7,:]=s_HVAC_pos_a
    all_values[8,:]=T_set_a
    
    
    
    dual_list[i]['dual_values']=np.array(m.pi)
    dual_list[i]['price']=price
    dual_list[i]['all_values']=all_values
    
    
    return real_power_array,dual_list

    
    
    
    
    
    

