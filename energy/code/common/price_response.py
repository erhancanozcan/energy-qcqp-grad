import numpy as np
import pandas as pd
from gurobipy import *
from energy.code.common.home_change_objective import change_objective_solve




def opt_response_to_price(new_price,dual_list,num_homes,models,Q,horizon,power_HVAC_list):
    
    
    home_real=0
    home_dev_cost=0
    
    for i in range (num_homes):
        cost=dual_list[i]['c']
        size_of_cost=len(cost)
        
        new_cost=cost[:6*horizon]
        new_cost=np.concatenate((new_cost,new_price))#wm
        new_cost=np.concatenate((new_cost,new_price))#oven
        new_cost=np.concatenate((new_cost,new_price))#dryer
        new_cost=np.concatenate((new_cost,new_price*power_HVAC_list[i]))#hvac
        new_cost=np.concatenate((new_cost,new_price))#ewh
        new_cost=np.concatenate((new_cost,new_price))#ev
        new_cost=np.concatenate((new_cost,-new_price))#pv
        
        new_cost=np.concatenate((new_cost,np.repeat(0,size_of_cost-len(new_cost))))#cost is zero for other vars.

        m=change_objective_solve(models[i].copy(),new_cost)

        vars_i=e_p=np.array([m.getVars()[idx].X for idx in range(len(m.getVars()))])
        dev_vars=vars_i[:6*horizon]
        real_vars=vars_i[6*horizon:12*horizon]
        
        home_i_dev_cost=np.dot(cost[:6*horizon],dev_vars)
        home_dev_cost+=home_i_dev_cost
        
        #print(home_i_dev_cost)
        tmp_real_vars=real_vars.reshape(6,horizon)
        tmp_real_vars[3,:]=tmp_real_vars[3,:]*power_HVAC_list[i]
        #print(np.sum(tmp_real_vars,axis=0))
        home_real+=np.sum(tmp_real_vars,axis=0)
        
        #print("can")
    
    calculated_obj=np.sum(abs(Q-home_real))+home_dev_cost
    #print(calculated_obj)
    
    return home_real,home_dev_cost,calculated_obj



"""
7.248427225798601
5.7997983651008385
5.562183653796652
[ 0.          2.47133331  9.76        2.95862494  0.          2.95862494
  0.5         2.96850545  0.5         3.5         2.93379079  0.
  2.95862494  0.          2.95862494  0.          2.95862494  0.
  2.55513205  0.36308681  2.71530192  0.21895647  2.85597221  0.092373
  2.97732586  9.76        3.264       2.92736372  5.76        2.95862494
  5.76        2.26545096  6.38375902  2.40548516  0.49774796  2.95862494
  0.          6.95862494  0.          6.95862494  4.          2.95862494
  0.          2.19588857  4.68635539  2.05130336  0.81646172  2.95862494
  0.          1.74265004  6.85420628  2.95862494  0.          2.95862494
  0.          2.86226559  0.          0.05842956  2.53169648  0.38417552
  2.95862494  5.76        2.95862494  2.40347019  2.95862494 13.26
  2.95862494  3.5         2.95862494  3.5         2.95862494  0.
  2.95862494  0.          2.95862494  3.5         2.09752296  0.7748706
  2.95862494  0.          2.24244767  0.64445875  2.30181407  0.59103735
  2.34988581  0.54777954  2.38881175  0.51275168  2.42425132  0.48086105
  2.45647532  0.45186399  2.48256859  0.42838371  2.50369754  0.40937063]
3.5945902110211825
[ 0.          1.9673254   9.76        3.          0.5         2.94609694
  0.5         2.96310064  3.5         0.5         2.92482435  0.
  2.95845938  0.          2.95845938  0.          2.95845938  0.
  2.95845938  0.          2.95845938  0.          2.98032485  2.95845938
  0.          8.76        0.          2.92482435  0.          2.95845938
  0.          2.95845938  6.73138574  2.95845938  5.76        8.71845938
  5.76        2.95845938  4.          6.95845938  8.192       2.95845938
  0.          2.95845938  4.          2.95845938  0.          2.95845938
  0.          2.95845938  9.76        2.95845938  0.          2.95845938
  0.          2.95262253  0.          0.          2.94500775  0.
  2.95845938  0.          2.95845938  0.34110148  2.95845938  4.264
  2.95845938  0.          2.95845938  5.76        2.95845938  5.76
 12.21845938  5.76        2.95845938  3.5         5.65904754  4.21933308
  2.95845938  0.          2.95845938  0.          2.3630498   0.53576616
  2.95845938  0.          2.95845938  0.          2.95845938  0.
  2.95845938  0.          2.95845938  0.          2.95845938  0.        ]

"""
