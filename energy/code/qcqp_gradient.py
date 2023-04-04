import os
os.environ["MKL_NUM_THREADS"] = "8"
import numpy as np
from gurobipy import *
#import cvxpy as cp
import numpy as np
from datetime import datetime
import copy
import pickle
#from mosek.fusion import *
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
from energy.code.common.update_learning_rate import update_lr
from energy.code.common.price_optimizer import optimizer




#data = np.loadtxt('/Users/can/data.csv', delimiter=',')


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
    #d_obj_list=[]
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
    alpha=inputs_dict['ca_kwargs']['alpha']
    #max_grad_steps=inputs_dict['ca_kwargs']['max_grad_steps']
    mean_Q=inputs_dict['ca_kwargs']['Q']
    #Q=abs(np.random.normal(mean_Q,10,size=horizon))#Kw supply
    #lambda_gap=inputs_dict['ca_kwargs']['lambda_gap']
    #MIPGap=inputs_dict['ca_kwargs']['mipgap']
    TimeLimit=inputs_dict['ca_kwargs']['timelimit']
    p_ub=inputs_dict['ca_kwargs']['p_ub']
    p_lb=inputs_dict['ca_kwargs']['p_lb']
    lr_rule=inputs_dict['ca_kwargs']['lr_update_rule']
    num_epochs=inputs_dict['ca_kwargs']['num_epochs']
    batch_size=inputs_dict['ca_kwargs']['batch_size']
    min_grad_norm=inputs_dict['ca_kwargs']['min_grad_norm']
    obj_threshold=inputs_dict['ca_kwargs']['obj_threshold']
    min_iter=inputs_dict['ca_kwargs']['min_iter']
    

    price_kwargs=inputs_dict['price_kwargs']
    
    p_optimizer=optimizer(lr,alpha,lr_rule)
    
    
    #price=abs(np.random.normal(mean_price,0.1,size=horizon))#0.33$ per KwH
    price=abs(rng.normal(mean_price,0.1,size=horizon))#0.33$ per KwH
    price[price<p_lb]=p_lb
    price[price>p_ub]=p_ub
    #price=np.zeros(horizon)
    
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
    
    ## Coordination Agent Initialization
    
    
    #repeated_price=np.repeat(price, 6)
    repeated_price=np.tile(price, 6)
    repeated_Q=np.tile(Q, 6)
    c_a_obj_list=[]
    grad_norm_list=[]
    
    best_ca_obj=np.inf
    opt_time=0
    running_mean=[1e16]
    message="iteration limit has been reached"
    for ep in range(num_epochs):
        train_idx = np.arange(num_homes)
        rng.shuffle(train_idx)
        sections = np.arange(0,num_homes,batch_size)[1:]
        batches = np.array_split(train_idx,sections)  
        
        if (num_homes % batch_size != 0):
            batches = batches[:-1]
        
        
        
        
        #below is logging
        print(ep)
        if ep%5==1:
            print("Epoch Number : %d and C.A. obj: %f.3"%(ep,c_a_obj_list[-1]))
        
        
        #calculate the C.A. objective
        p_at_t=0
        for i in range(num_homes):
            p_at_t+=np.sum(real_power_list[i],axis=0)
        dev_from_target=np.sum((Q-p_at_t)**2)
        
        dev_cost=0
        for i in range(num_homes):
            cost_v=dual_list[i]['dev_cost'][np.arange(0,6,1)*horizon]
            dev_cost+=np.dot(np.sum(desirable_power_list[i],axis=1),cost_v)
        
        c_a_obj_list.append(dev_from_target+dev_cost)
        
        #logging
        if c_a_obj_list[-1]<best_ca_obj:
            best_ca_obj=c_a_obj_list[-1]
            best_real_power_schedule=p_at_t
            best_price=price
        p_optimizer.update_best(best_ca_obj,best_price)
        #end of logging.    
        
        
        grad_tmp=[]
        for homes_list in batches:
            #calculate the total gradient.
            grad_f_price,max_time=calc_grad(num_homes,homes_list,dual_list,horizon,repeated_price,repeated_Q,desirable_power_list,real_power_list,power_HVAC_list)
            grad_tmp.append(np.dot(grad_f_price,grad_f_price))
            #grad_norm_list.append(np.dot(grad_f_price,grad_f_price)**0.5)
            opt_time+=max_time
        
            #update price and project into feasible region.
            #new_lr=update_lr(lr_rule,lr,ep,grad_norm_list)
            #price=price-new_lr*grad_f_price
            price=p_optimizer.update_price(price,grad_f_price)
            price[price<p_lb]=p_lb
            price[price>p_ub]=p_ub
        
            #solve home MPC with the updated price information.
            real_power_list=[]
            max_time=0
            for i in range (num_homes):
                home_start_time = datetime.now()
                real_power,dual=update_price_and_solve(price,dual_list,i,horizon,desirable_power_list,models,power_HVAC_list)
                real_power_list.append(real_power)        
                home_end_time = datetime.now()
                time_diff=(home_end_time-home_start_time).seconds+\
                    (home_end_time-home_start_time).microseconds*1e-6
                if time_diff>max_time:
                    max_time=time_diff
            opt_time+=max_time
        grad_norm_list.append(np.sum(np.array(grad_tmp))**0.5)
        
        
        if opt_time>TimeLimit-20:
            message="time is up"
            print('Time is Up!')
            break
        if grad_norm_list[-1]<min_grad_norm:
            message="small gradients"
            print('Gradient at he current iterate is small. Local minima is attained.')
            break
        
             
        if ep>=min_iter:
            obj_vals=np.array(c_a_obj_list)
            tmp_mean=np.mean(obj_vals[-min_iter:])
            running_mean.append(tmp_mean)
            
            if abs(running_mean[-1]-running_mean[-2])/running_mean[-2] < obj_threshold:
                message="no improvement"
                print('No considerable improvement')
                break
            
        
    
    
    #calculate the c.a. objective at the end.
    p_at_t=0
    for i in range(num_homes):
        p_at_t+=np.sum(real_power_list[i],axis=0)
    dev_from_target=np.sum((Q-p_at_t)**2)
    
    dev_cost=0
    for i in range(num_homes):
        cost_v=dual_list[i]['dev_cost'][np.arange(0,6,1)*horizon]
        dev_cost+=np.dot(np.sum(desirable_power_list[i],axis=1),cost_v)
    
    c_a_obj_list.append(dev_from_target+dev_cost)
    if c_a_obj_list[-1]<best_ca_obj:
        best_ca_obj=c_a_obj_list[-1]
        best_real_power_schedule=p_at_t
        best_price=price
    
    p_optimizer.update_best(best_ca_obj,best_price)
    
    desirable_power=0
    for i in range(num_homes):
        desirable_power+=np.sum(desirable_power_list[i],axis=0)
        
    
    
    power_summary={
        
                    'c_a_obj_list':c_a_obj_list,
                    'best_real_power_schedule':best_real_power_schedule,
                    'best_price':best_price,
                    'final_real_power_schedule':p_at_t,
                    'final_price':price,
                    
                    'desirable_power':desirable_power,
                                       
                    'Q'        : Q,
                    'num_homes':num_homes,
                    
                    'grad_norm_list':grad_norm_list,

 
                    'mean_price':mean_price,
                    'price_lb': p_lb,
                    'price_ub': p_ub,
                    
                    'lr_init':lr_init,
                    'lr':lr,
                    'alpha':alpha,
                    'beta1':p_optimizer.beta1,
                    'beta2':p_optimizer.beta2,
                    'num_epochs':num_epochs,
                    'batch_size':batch_size,
                    'update_rule':lr_rule,
                    'min_grad_norm':min_grad_norm,
                    'obj_threshold':obj_threshold,
                    'min_iter':min_iter,
                    'message':message,
                    
                    'opt_time':opt_time,
                    'ep':ep
                    

                   
                   }

    return power_summary
    


#conda install -c mosek mosek


#sum(np.array([var.name()==v.Varname for var in variables]))
#var279377
#var270524






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
    
