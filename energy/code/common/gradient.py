import numpy as np
from datetime import datetime

def calc_grad(num_homes,homes_list,dual_list,horizon,repeated_price,repeated_Q,desirable_power_list,real_power_list,power_HVAC_list):
    
    #in gradient calculations be careful with the deviation cost and HVAC coefficient.
    #whenever you use real power, do not need to care about HVAC coeff because we already
    #multiplied this in home.
    #
    current_real_power=0
    for i in range(num_homes):
        current_real_power+=np.sum(real_power_list[i],axis=0)
    current_real_power=np.tile(current_real_power,6)
    
    
    grad_f_price=0
    max_time=0
    skip_home=0
    #for i in range(num_homes):
    for i in homes_list:
        home_start_time = datetime.now()
        #hvac_i=power_HVAC_list[i]
        num_variables_in_home_i=dual_list[i]['A'].shape[1]
        num_ineq_constrs_in_home_i=dual_list[i]['A'].shape[0]
        num_power_vars=6*horizon
        num_remaining_vars=num_variables_in_home_i-num_power_vars
        dev_cost_i=dual_list[i]['dev_cost']
        p_hvac=power_HVAC_list[i]
        
        
        lambda_i=dual_list[i]['dual_values']
        p_hvac_coef=np.ones(num_power_vars)
        p_hvac_coef[3*horizon:4*horizon]=p_hvac
        grad_xi_fi=np.multiply(repeated_price,p_hvac_coef)+\
                np.multiply(2*(desirable_power_list[i].flatten()-real_power_list[i].flatten())*-1,
                       dev_cost_i)     
        grad_xi_fi=np.concatenate([grad_xi_fi,np.zeros(num_remaining_vars)])
        
        tmp=np.repeat(2,num_power_vars)
        tmp=np.multiply(tmp,dev_cost_i)
        tmp=np.multiply(tmp,p_hvac_coef)
        tmp=np.diag(tmp)
        hessian_xi_fi=np.block([
                                [tmp, np.zeros((num_power_vars,num_remaining_vars))],
                                [np.zeros((num_remaining_vars,num_power_vars)),np.zeros((num_remaining_vars,num_remaining_vars))]])
        
        #hessian_negxi_fi=np.zeros((num_variables_in_home_i*(num_homes-1),num_variables_in_home_i*(num_homes-1)))
        
        g_i=dual_list[i]['A'].toarray()
        
        diag_lambda_i_g_i=np.dot(np.diag(-lambda_i),g_i)
        
        g_x_star_m_h=np.dot(g_i,dual_list[i]['all_values'].flatten())-dual_list[i]['RHS']
        g_x_star_m_h=np.diag(g_x_star_m_h)
        
        lhs_matrix_i=np.block([
                                [hessian_xi_fi, g_i.T],
                                [diag_lambda_i_g_i,g_x_star_m_h]])
        
        #
        #inverse_lhs_matrix_i=np.linalg.inv(lhs_matrix_i)
        #inverse_lhs_matrix_i=np.linalg.pinv(lhs_matrix_i)
        
        
        
        #rhs_grad_pi_F=np.tile(np.eye(horizon),(6,1))
        rhs_grad_pi_F=np.multiply(np.tile(np.eye(horizon),(6,1)),p_hvac_coef[:, np.newaxis])
        tmp=np.zeros((num_remaining_vars,horizon))
        rhs_grad_pi_F=np.concatenate([rhs_grad_pi_F,tmp])
        
        rhs=np.concatenate([rhs_grad_pi_F, np.zeros((len(lhs_matrix_i)-rhs_grad_pi_F.shape[0],horizon)) ])
        
        #jacob_matrix_i=np.dot(inverse_lhs_matrix_i,-rhs)[:9*horizon,:]
        lhs_matrix_i=np.array(lhs_matrix_i, dtype='f')
        rhs=np.array(rhs, dtype='f')
        lhs_matrix_i[abs(lhs_matrix_i)<=1e-8]=0
        rhs[abs(rhs)<=1e-8]=0
        
        try:
            jacob_matrix_i = np.linalg.lstsq(lhs_matrix_i, -rhs,rcond=-1)[0][:9*horizon,:]
        except:
            skip_home+=1
            continue
        #jacob_matrix_i = np.linalg.lstsq(lhs_matrix_i, -rhs,rcond=-1)[0][:9*horizon,:]
        #grad_i_ca_obj=2*(repeated_Q-real_power_list[i].flatten())*-1+\
        #                np.multiply(2*(desirable_power_list[i].flatten()-real_power_list[i].flatten())*-1,
        #                            dev_cost_i)
        grad_i_ca_obj=2*(repeated_Q-current_real_power)*-1+\
                        np.multiply(2*(desirable_power_list[i].flatten()-real_power_list[i].flatten())*-1,
                                    dev_cost_i)
        grad_i_ca_obj=np.concatenate([grad_i_ca_obj,np.zeros(num_remaining_vars)])
                                    
        #print('can')
        grad_f_i_price=np.dot(jacob_matrix_i.T,grad_i_ca_obj)
        grad_f_price+=grad_f_i_price
        
        home_end_time = datetime.now()
        time_diff=(home_end_time-home_start_time).seconds+\
            (home_end_time-home_start_time).microseconds*1e-6
        if time_diff>max_time:
            max_time=time_diff
    
    if skip_home>0:
        print('For %d homes gradient evaluation failed'%(skip_home))
    return grad_f_price/(num_homes-skip_home),max_time
        
        
        