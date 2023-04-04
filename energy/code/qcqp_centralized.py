import os
import numpy as np
from gurobipy import *
#import cvxpy as cp
import numpy as np
from datetime import datetime
import copy
import pickle
from numpy.linalg import eig

import sys
sys.path.append("/Users/can/Documents/GitHub")




from energy.code.home import Home
from energy.code.common.demand import initialize_demand
from energy.code.common.appliance import initialize_appliance_property
from energy.code.common.dual_constraints import solve_dual
from energy.code.common.arg_parser import create_train_parser, all_kwargs
from energy.code.common.home_change_objective import change_objective_solve,update_price_and_solve
from energy.code.common.price_response import opt_response_to_price
from energy.code.common.objective_comparison import check_objectives
from energy.code.common.feasibility import check_random_feasibility,check_mean_feasibility
from energy.code.common.coordination_agent import explore_with_gurobi
from energy.code.common.gradient import calc_grad

def gather_inputs(args):
    """Organizes inputs to prepare for simulations."""

    args_dict = vars(args)
    inputs_dict = dict()

    for key,param_list in all_kwargs.items():
        active_dict = dict()
        for param in param_list:
            active_dict[param] = args_dict[param]
        inputs_dict[key] = active_dict

    return inputs_dict


def train(inputs_dict):
    
    #np.random.seed(inputs_dict['setup_kwargs']['setup_seed'])
    
    rng = np.random.default_rng(inputs_dict['setup_kwargs']['setup_seed'])
    
    def return_constraint(expression,logic,rhs):
        
        if logic == '<':
            return [expression <= rhs]
        elif logic== '>':
            return [expression >= rhs]
        else:
            return [expression == rhs]
    
    dual_list=[]
    models=[]
    p_obj_list=[]
    d_obj_list=[]
    power_HVAC_list=[]
    real_power_list_start=[] #use this list to understand how optimizing price affects the load consumption.
    real_power_list=[]
    dev_power_list_before_changing_price=[]
    
    s_effect=inputs_dict['home_kwargs']['s_effect']
    num_homes=inputs_dict['ca_kwargs']['num_houses']
    horizon=inputs_dict['ca_kwargs']['horizon']
    mean_price=inputs_dict['ca_kwargs']['price']
    lr=inputs_dict['ca_kwargs']['lr']
    lr_init=inputs_dict['ca_kwargs']['lr']
    max_grad_steps=inputs_dict['ca_kwargs']['max_grad_steps']
    mean_Q=inputs_dict['ca_kwargs']['Q']
    #Q=abs(np.random.normal(mean_Q,10,size=horizon))#Kw supply
    lambda_gap=inputs_dict['ca_kwargs']['lambda_gap']
    MIPGap=inputs_dict['ca_kwargs']['mipgap']
    TimeLimit=inputs_dict['ca_kwargs']['timelimit']
    p_ub=inputs_dict['ca_kwargs']['p_ub']
    p_lb=inputs_dict['ca_kwargs']['p_lb']
    provide_sol=inputs_dict['ca_kwargs']['provide_solution']

    price_kwargs=inputs_dict['price_kwargs']
    
    
    #price=abs(np.random.normal(mean_price,0.1,size=horizon))#0.33$ per KwH
    price=abs(rng.normal(mean_price,0.1,size=horizon))#0.33$ per KwH
    price[price<p_lb]=p_lb
    price[price>p_ub]=p_ub
    #price=np.zeros(horizon)
    
    if inputs_dict['set_price_kwargs']['price_file'] is not None:
        import_filefull = os.path.join(inputs_dict['set_price_kwargs']['price_path'],inputs_dict['set_price_kwargs']['price_file'])
        with open(import_filefull,'rb') as f:
            import_logs = pickle.load(f)
        #print('can')
        price=import_logs['best_price']
        price[price<p_lb]=p_lb
        price[price>p_ub]=p_ub
        #print(price)
        #raise NotImplementedError
    
    
    desirable_power_list=[]
    desirable_power=0
    #Random demand and appliance initialization for each home.
    i=0
    while i<num_homes:
        home=Home(s_effect=s_effect)
        home=initialize_demand(home,rng=rng)
        home=initialize_appliance_property(home,s_effect,rng=rng)
        home.generate_desirable_load()
        
        assert horizon == len(home.wm_desirable_load),"Horizon Change is detected. Check Time resolution of appliances"
        total,cost_u,daily_fee_desirable=home.total_desirable_load(price,mean_price)
        try:
            home_des_power=np.zeros((6,horizon))
            real_power,dev_power,states,dual,m,p_obj=home.optimize_mpc(cost_u,price)
            i=i+1
            models.append(m)
            dual_list.append(dual)
            p_obj_list.append(p_obj)
            power_HVAC_list.append(home.hvac.nominal_power)
            real_power_list_start.append(real_power)
            real_power_list.append(real_power)
            dev_power_list_before_changing_price.append(dev_power)
            desirable_power+=total
            home_des_power[0,:]=home.wm_desirable_load
            home_des_power[1,:]=home.oven_desirable_load
            home_des_power[2,:]=home.dryer_desirable_load
            home_des_power[3,:]=home.hvac_desirable_load
            home_des_power[4,:]=home.ewh_desirable_load
            home_des_power[5,:]=home.ev_desirable_load
            desirable_power_list.append(home_des_power)
        except Exception:
            print("demand was infeasible due to initialization skip this home.")
            pass
    
    
    if mean_Q > 0:
        Q=abs(rng.normal(mean_Q,10,size=horizon))#Kw supply
    else:
        cumulative_desired_energy=0
        generated_PV=0
        for i in range(num_homes):
            real_c=0#real_power_list_before_changing_price[i]
            generated_PV+=0#np.sum(real_c['pv'])
            dev_c=dev_power_list_before_changing_price[i]
            
            cumulative_desired_energy+=np.sum(desirable_power_list[i])
        Q=np.repeat((cumulative_desired_energy-generated_PV)/horizon,horizon)
        
        
    
    ##Coordination Agent Problem Initialization 
    m_c_a=Model("m_c_a")
    price_e=m_c_a.addVars(horizon,lb=0,name="price")
    #price_e=m_c_a.addVars(horizon,lb=p_lb,name="price")
    #price_e=m_c_a.addVars(horizon,lb=p_lb,ub=p_ub,name="price")

    price_lb=m_c_a.addConstrs((price_e[i]-p_lb>=0
                                          for i in range(horizon)),name='price_lower_bounds')
    price_ub=m_c_a.addConstrs((price_e[i]-p_ub<=0
                                          for i in range(horizon)),name='price_upper_bounds')
    
    

    d_obj_list=[]
    d_obj_exp_list=[]
    
    
    d_vars=[]
    #For loop below prepares the constraints of m_c_a problem.
    for i in range(num_homes):
        
        p_m=models[i]
        dual=dual_list[i]
        num_dual_vars=dual['A'].shape[0]
        #d_m,d_obj=solve_dual(dual)
        
        
        tmp_p_name="H"+str(i+1)+"_P_"
        tmp_d_name="H"+str(i+1)+"_D_"
        tmp_dnonneg_name="H"+str(i+1)+"_Dnonneg_"
        
        for v in p_m.getVars():
            v.varname=tmp_p_name+v.varname
        p_m.update()
            

        
        for v in p_m.getVars():
            m_c_a.addVar(lb=v.lb, ub=v.ub, vtype=v.vtype, name=v.varname)
        
        #d_vars.append(m_c_a.addVars(num_dual_vars,lb=0, name=tmp_d_name))
        #m_c_a.addVars(num_dual_vars,lb=0, name=tmp_d_name)
        m_c_a.addVars(num_dual_vars,lb=0, name=tmp_d_name)
        m_c_a.addVars(len(p_m.getVars()),lb=0, name=tmp_dnonneg_name)
        
        m_c_a.update()
        
        
        #for c in p_m.getConstrs():
        #    expr = p_m.getRow(c)
        #    newexpr = LinExpr()
        #    for j in range(expr.size()):
        #        v = expr.getVar(j)
        #        coeff = expr.getCoeff(j)
        #        newv = m_c_a.getVarByName(v.Varname)
        #        newexpr.add(newv, coeff)
                
        #    m_c_a.addConstr(newexpr, c.Sense, c.RHS, name=tmp_p_name+c.ConstrName)
            
        
    
    home_all_vars=[]
    home_reals=[]
    home_reals_except_hvac=[]
    home_reals_hvac=[]
    d_vars=[]
    dnonneg_vars=[]
    
    for i in range(num_homes):
        
        names_to_retrieve=[]
        names_to_retrieve_except_hvac=[]
        names_to_retrieve_hvac=[]
        names_to_retrieve_dual=[]
        names_to_retrieve_dnonneg=[]
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_wm_real["+str(j)+"]")
            names_to_retrieve_except_hvac.append("H"+str(i+1)+"_P_wm_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_oven_real["+str(j)+"]")
            names_to_retrieve_except_hvac.append("H"+str(i+1)+"_P_oven_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_dryer_real["+str(j)+"]")
            names_to_retrieve_except_hvac.append("H"+str(i+1)+"_P_dryer_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_hvac_real["+str(j)+"]")
            names_to_retrieve_hvac.append("H"+str(i+1)+"_P_hvac_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ewh_real["+str(j)+"]")
            names_to_retrieve_except_hvac.append("H"+str(i+1)+"_P_ewh_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ev_real["+str(j)+"]")
            names_to_retrieve_except_hvac.append("H"+str(i+1)+"_P_ev_real["+str(j)+"]")
        
        home_reals.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        
        home_reals_except_hvac.append([m_c_a.getVarByName(name) for name in names_to_retrieve_except_hvac])
        home_reals_hvac.append([m_c_a.getVarByName(name) for name in names_to_retrieve_hvac])
        
        
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_s_neg["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_s_plus["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_t_set["+str(j)+"]")
        
        home_all_vars.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        
        
        dual=dual_list[i]
        num_dual_vars=dual['A'].shape[0]
        
        for j in range(num_dual_vars):
            names_to_retrieve_dual.append("H"+str(i+1)+"_D_["+str(j)+"]")
        d_vars.append([m_c_a.getVarByName(name) for name in names_to_retrieve_dual])
        
        
        for j in range(len(names_to_retrieve)):
            names_to_retrieve_dnonneg.append("H"+str(i+1)+"_Dnonneg_["+str(j)+"]")
        dnonneg_vars.append([m_c_a.getVarByName(name) for name in names_to_retrieve_dnonneg])

      

    ##KKT condiitons (Eq 4)    
    sos_helper_var_list=[]
    for i in range (num_homes):
        
        
        num_variables_in_home_i=dual_list[i]['A'].shape[1]
        num_ineq_constrs_in_home_i=dual_list[i]['A'].shape[0]
        num_power_vars=6*horizon
        num_remaining_vars=num_variables_in_home_i-num_power_vars
        dev_cost_i=dual_list[i]['dev_cost']
        
        flat_desirable_power=desirable_power_list[i].flatten()
        g_i=dual_list[i]['A'].toarray()# need to write the experression for the second term
        h_i=dual_list[i]['RHS']
        p_hvac=power_HVAC_list[i]
        
        dot_prods=[]
        
        for k in range(g_i.shape[1]):
            dot_prods.append(quicksum(g_i[l][k]*d_vars[i][l] for l in range (g_i.shape[0])))

        
        #gradient of home objective must consider HVAC coef!.
        p_hvac_coef=np.ones(num_power_vars)
        p_hvac_coef[3*horizon:4*horizon]=p_hvac
        m_c_a.addConstrs((price_e[t%horizon]*p_hvac_coef[t]-2*dev_cost_i[t]*(flat_desirable_power[t]-home_reals[i][t]*p_hvac_coef[t])*p_hvac_coef[t]+dot_prods[t]-dnonneg_vars[i][t]==0 for t in range(num_power_vars)),name=str(i+1)+"first_order")
        #m_c_a.addConstrs((price[t%horizon]*p_hvac_coef[t]-2*dev_cost_i[t]*(flat_desirable_power[t]-home_reals[i][t]*p_hvac_coef[t])*p_hvac_coef[t]+dot_prods[t]-dnonneg_vars[i][t]==0 for t in range(num_power_vars)),name=str(i+1)+"first_order")

        m_c_a.addConstrs((0+dot_prods[576+t]-dnonneg_vars[i][576+t]==0 for t in range(num_remaining_vars)),name=str(i+1)+"first_order_rest")        
        
        #grad_xi_fi=repeated_price+\
        #        np.multiply(2*(desirable_power_list[i].flatten()-real_power_list[i].flatten())*-1,
        #               dev_cost_i)     
        #grad_xi_fi=np.concatenate([grad_xi_fi,np.zeros(num_remaining_vars)])
        
        sos_helper_var_list.append(m_c_a.addVars(num_ineq_constrs_in_home_i,lb=-GRB.INFINITY,ub=0,name=str(i+1)+'sos_auxiliary_varibles'))
        #sos_helper_var_list.append(m_c_a.addVars(num_ineq_constrs_in_home_i,lb=-GRB.INFINITY,ub=GRB.INFINITY,name=str(i+1)+'sos_vars'))
        
        for k in range(g_i.shape[0]):
            m_c_a.addConstr((sos_helper_var_list[i][k]==quicksum(g_i[k][l]*home_all_vars[i][l] for l in range (g_i.shape[1]))-h_i[k]),name=str(i+1)+'sos_vars_equality'+str(k))
            
        m_c_a.update()
            
        m_c_a.addConstrs((d_vars[i][k] * sos_helper_var_list[i][k] ==0 for k in range (g_i.shape[0])),name=str(i+1)+'cs1')
        m_c_a.addConstrs((home_all_vars[i][k] * dnonneg_vars[i][k] ==0 for k in range (num_variables_in_home_i)),name=str(i+1)+'cs1')
        
        
        #for k in range(g_i.shape[0]):
        #    m_c_a.addSOS( GRB.SOS_TYPE1, [ d_vars[i][k], sos_helper_var_list[i][k] ],[1,2])
        #for k in range(num_variables_in_home_i):
        #    m_c_a.addSOS( GRB.SOS_TYPE1, [ home_all_vars[i][k], dnonneg_vars[i][k] ],[1,2])
            
    m_c_a.update()

    obj_term_list=[]
    for j in range(horizon):

        obj_term1=Q[j]
        for t in range (6):
            if t==3:
                obj_term1=obj_term1- quicksum(home_reals[i][t*horizon+j]*power_HVAC_list[i] for i in range(num_homes))
            else:
                obj_term1=obj_term1 - quicksum(home_reals[i][t*horizon+j] for i in range(num_homes))
        
        obj_term_list.append(obj_term1*obj_term1)
        
    
    obj_term_dev=[]#list holding the cost related to home deviation.
    for i in range(num_homes):
        dev_cost_i=dual_list[i]['dev_cost']
        flat_desirable_power=desirable_power_list[i].flatten()
        p_hvac=power_HVAC_list[i]
        p_hvac_coef=np.ones(len(dev_cost_i))
        p_hvac_coef[3*horizon:4*horizon]=p_hvac
        
        obj_term_dev.append(quicksum((flat_desirable_power[t]-home_reals[i][t]*p_hvac_coef[t])*(flat_desirable_power[t]-home_reals[i][t]*p_hvac_coef[t])*dev_cost_i[t] for t in range(len(dev_cost_i))))
        
        
    m_c_a.setObjective(quicksum(obj_term_list[t] for t in range(horizon))+\
                       quicksum(obj_term_dev[i] for i in range(num_homes)) ,GRB.MINIMIZE) 
        
    
    #m_c_a.setObjective(quicksum(obj_term_dev[i] for i in range(num_homes)) ,GRB.MINIMIZE)
    
    #m_c_a.setObjective(quicksum(home_reals[0][t] for t in range(len(dev_cost_i))) ,GRB.MINIMIZE)
    
    
        
        
    
    #-1 automatic 0 primal 1 dual 2 barrier
    #m_c_a.Params.Method=2
    m_c_a.Params.NonConvex=2
    m_c_a.Params.MIPGap = MIPGap
    #presolved=m_c_a.presolve()
    #m_c_a.Params.DualReductions=0
    #m_c_a.Params.Presolve = 1   m_c_a.Params.Presolve = 0
    #m_c_a.reset()
    #m_c_a.printQuality()
    #m_c_a.Params.Presolve = 0
    m_c_a.Params.BarHomogeneous=1
    #m_c_a.Params.NumericFocus=3
    #m_c_a.Params.FeasibilityTol=1e-6
    m_c_a.Params.TimeLimit = TimeLimit
    #m_c_a.write('/Users/can/Desktop/gurobi_model.lp')
    #m_c_a.write('/Users/can/Desktop/gurobi_model.mps')
    
    if provide_sol == True:
        for t in range (horizon):
            price_e[t].Start=price[t]
            
        for i in range(num_homes):
            all_values=dual_list[i]['all_values']
            all_values[3,:]=all_values[3,:]/power_HVAC_list[i] 
            #all_values=all_values.flatten()
            for j in range(9):
                for t in range(horizon):
                    home_all_vars[i][j*horizon+t].Start=all_values[j,t]
                    #if j == 3:
                    #    home_all_vars[i][j*horizon+t].Start=real_power_list_start[i][j,t]/power_HVAC_list[i]
                    #else:
                    #    home_all_vars[i][j*horizon+t].Start=real_power_list_start[i][j,t]
                    
        for i in range(num_homes):
                       
            dual=dual_list[i]
            num_dual_vars=dual['A'].shape[0]
            d_values=dual['dual_values']
            for j in range (num_dual_vars):
                d_vars[i][j].Start=-d_values[j]
    
    opt_start_time = datetime.now()
    m_c_a.optimize()#1.65313255e+01
    opt_end_time = datetime.now()
    opt_time=(opt_end_time-opt_start_time).seconds+\
        (opt_end_time-opt_start_time).microseconds*1e-6
          
    
    """
    #price start
    for t in range (horizon):
        price_e[t].Start=price[t]
        
    for i in range(num_homes):
        all_values=dual_list[i]['all_values']
        all_values[3,:]=all_values[3,:]/power_HVAC_list[i] 
        #all_values=all_values.flatten()
        for j in range(9):
            for t in range(horizon):
                home_all_vars[i][j*horizon+t].Start=all_values[j,t]
                #if j == 3:
                #    home_all_vars[i][j*horizon+t].Start=real_power_list_start[i][j,t]/power_HVAC_list[i]
                #else:
                #    home_all_vars[i][j*horizon+t].Start=real_power_list_start[i][j,t]
                
    for i in range(num_homes):
                   
        dual=dual_list[i]
        num_dual_vars=dual['A'].shape[0]
        d_values=dual['dual_values']
        for j in range (num_dual_vars):
            d_vars[i][j].Start=-d_values[j]
     """   


    #real_power_list_start
    
    
    #[variables.varname for variables in m_c_a.getVars()]
    
    
    #price_ub=m_c_a.addConstrs((price_e[i]==price[i]
    #                                      for i in range(horizon)),name='price_upper_bounds')
    #constr_matrix=m_c_a.getA().toarray()
    #np.linalg.matrix_rank(constr_matrix)
    #constr_matrix.shape
    
    #concatenated=np.concatenate([constr_matrix,np.vstack(np.array(m_c_a.RHS))],axis=1)
    #np.linalg.matrix_rank(concatenated)
    #concatenated.shape
    
    desirable_power=0
    for i in range(num_homes):
        desirable_power+=np.sum(desirable_power_list[i],axis=0)
        
    best_price=np.array([price_e[i].X for i in range (horizon)])
    
    
    power_schedule=0
    real_power_list= []
    for i in range (num_homes):
    #real power levels according to price
        P_ewh_a=np.zeros(horizon)
        P_ev_a=np.zeros(horizon)
        P_hvac_a=np.zeros(horizon)
        P_oven_a=np.zeros(horizon)
        P_wm_a=np.zeros(horizon)
        P_dryer_a=np.zeros(horizon)
        #P_pv_a=np.zeros(horizon)
        
        for k in range (horizon):
            P_wm_a[k]=home_reals[i][0*horizon+k].X
            P_oven_a[k]=home_reals[i][1*horizon+k].X
            P_dryer_a[k]=home_reals[i][2*horizon+k].X
            P_hvac_a[k]=home_reals[i][3*horizon+k].X *power_HVAC_list[i]#HVAC variable is relaxed binary. Need to have a multiplication.
            P_ewh_a[k]=home_reals[i][4*horizon+k].X
            P_ev_a[k]=home_reals[i][5*horizon+k].X
            #P_pv_a[k]=home_reals[i][6*horizon+k].X
        
        power_schedule+=P_wm_a+P_oven_a+P_dryer_a+P_hvac_a+P_ewh_a+P_ev_a
        
    
        
    

        
    
    
    power_summary={
        
                    'c_a_obj_list':[m_c_a.objVal],
                    'best_obj_bound':m_c_a.ObjBound,
                    'best_real_power_schedule':power_schedule,
                    'best_price':best_price,
                    'final_real_power_schedule':None,
                    'final_price':None,
                    
                    'desirable_power':desirable_power,
                                       
                    'Q'        : Q,
                    'num_homes':num_homes,
                    
                    'grad_norm_list': None,

 
                    'mean_price':mean_price,
                    'price_lb': p_lb,
                    'price_ub': p_ub,
                    'provide_sol':provide_sol,
                    
                    'lr_init':lr_init,
                    'lr':lr,
                    'max_grad_steps':max_grad_steps,
                    
                    'opt_time':opt_time,
                    'optimality_gap':m_c_a.MIPGap,
                    'timelimit':TimeLimit
                    

                   
                   }

    return power_summary









def main():
    
    start_time = datetime.now()
    
    parser = create_train_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args)
    
    seeds = np.random.SeedSequence(args.seed).generate_state(3)
    setup_seeds = np.random.SeedSequence(seeds[0]).generate_state(
     args.runs+args.runs_start)[args.runs_start:]
    #array([2968811710, 3677149159,  745650761]
    
    inputs_list = []
    
    for run in range(args.runs):
        setup_dict = dict()
        setup_dict['idx'] = run + args.runs_start
        if args.setup_seed is None:
            setup_dict['setup_seed'] = int(setup_seeds[run])

        inputs_dict['setup_kwargs'] = setup_dict
        inputs_list.append(copy.deepcopy(inputs_dict))

    if args.cores is None:
        args.cores = args.runs

    power_summary=train(inputs_list[0])
    
    
    
    #with mp.get_context('spawn').Pool(args.cores) as pool:
    #    power_summary_list = pool.map(train,inputs_list)
    
    
    

    """
    #creates a folder named as logs
    os.makedirs("/Users/can/Desktop/logs",exist_ok=True)
    #names the file name
    save_file = "deneme"
    save_filefull = os.path.join("/Users/can/Desktop/logs",save_file)
    """
    
    #os.makedirs("/home/erhan/energy/logs",exist_ok=True)
    os.makedirs(args.save_path,exist_ok=True)
    
    save_date=datetime.today().strftime('%m%d%y_%H%M%S')
    
    if args.save_file is None:
        save_file = '%s_%s_%s_%s_%s_%s_%s'%(args.num_houses,args.horizon,
            args.price,args.Q,args.lambda_gap,args.mipgap,save_date)
    else:
        save_file = '%s_%s'%(args.save_file,save_date)
    
    #save_filefull = os.path.join("/home/erhan/energy/logs",save_file)
    save_filefull = os.path.join(args.save_path,save_file)
    

    with open(save_filefull,'wb') as f:
        pickle.dump(power_summary,f)

    ########
    
    end_time = datetime.now()
    
    print('Time Elapsed: %s'%(end_time-start_time))
    
    
    

    

if __name__=='__main__':
    main()