from gurobipy import *
import numpy as np
import copy
from energy.code.common.dual_constraints import solve_dual






def explore_with_gurobi(mean_vec,cov_matrix,horizon,mean_price,price,p_ub,num_homes,models,dual_list,power_HVAC_list,Q,lambda_gap,rng,price_kwargs):
    
    
    m_c_a=Model("m_c_a")
    price_e=m_c_a.addVars(horizon,lb=0,name="price")
    #price_lb=m_c_a.addConstrs((price_e[i]-price[i]>=0
    #                                      for i in range(horizon)),name='price_lower_bounds')
    price_lb=m_c_a.addConstrs((price_e[i]-mean_price>=0
                                          for i in range(horizon)),name='price_lower_bounds')
    #price_ub=m_c_a.addConstrs((price_e[i]-price[i]<=p_ub
    #                                      for i in range(horizon)),name='price_upper_bounds')
    price_ub=m_c_a.addConstrs((price_e[i]-mean_price<=p_ub
                                          for i in range(horizon)),name='price_upper_bounds')
    epsilon=m_c_a.addVars(num_homes,lb=0,name="epsilon_home")

    d_obj_list=[]
    d_obj_exp_list=[]
    
    #For loop below prepares the constraints of m_c_a problem.
    for i in range(num_homes):
        
        p_m=models[i]
        dual=dual_list[i]
        d_m,d_obj=solve_dual(dual)
        
        
        tmp_p_name="H"+str(i+1)+"_P_"
        tmp_d_name="H"+str(i+1)+"_D_"
        
        #for v in p_m.getVars():
        #    v.varname=tmp_p_name+v.varname
        #p_m.update()
            
        for v in d_m.getVars():
            v.varname=tmp_d_name+v.varname
        d_m.update()
        
        for v in p_m.getVars():
            m_c_a.addVar(lb=v.lb, ub=v.ub, vtype=v.vtype, name=v.varname)
        for v in d_m.getVars():
            m_c_a.addVar(lb=v.lb, ub=v.ub, vtype=v.vtype, name=v.varname)
        m_c_a.update()
        
        
        for c in p_m.getConstrs():
            expr = p_m.getRow(c)
            newexpr = LinExpr()
            for j in range(expr.size()):
                v = expr.getVar(j)
                coeff = expr.getCoeff(j)
                newv = m_c_a.getVarByName(v.Varname)
                newexpr.add(newv, coeff)
                
            m_c_a.addConstr(newexpr, c.Sense, c.RHS, name=tmp_p_name+c.ConstrName)
            
        
        ind=0
        for c in d_m.getConstrs():
            #print(c)
            expr = d_m.getRow(c)
            #print(expr)
            newexpr = LinExpr()
            for j in range(expr.size()):
                v = expr.getVar(j)
                coeff = expr.getCoeff(j)
                newv = m_c_a.getVarByName(v.Varname)
                newexpr.add(newv, coeff)
            """
            This if-else blog handles the dual constraints note that rhs in duals must be decision variable!
            """
            if (ind <= horizon*6-1) or (ind >= horizon*13): # horizon is 96. 
            #We have 6 because deviations in 6 appliances. We have 13 because 6+7 where 7 real power decision.
                m_c_a.addConstr(newexpr, c.Sense, c.RHS, name=tmp_d_name+c.ConstrName)
            else:
                """
                If else below handles the cost modification of HVAC. Please see check HOMEMPC and observe
                that we multiply price by power_HVAC
                """
                #print("cntrl")
                if (ind >= 9*horizon) and (ind <= horizon*10-1):
                    newexpr.add(price_e[ind%horizon], -1*power_HVAC_list[i])
                elif (ind >= 12*horizon):
                    newexpr.add(price_e[ind%horizon], +1)
                else:
                    newexpr.add(price_e[ind%horizon], -1)
                #m_c_a.addConstr(newexpr, c.Sense, c.RHS, name=tmp_d_name+c.ConstrName)
                m_c_a.addConstr(newexpr, c.Sense, 0.0, name=tmp_d_name+c.ConstrName)
            ind+=1
                
            
        m_c_a.update()
        
        #
        
        newexpr_d = LinExpr()
        expr=d_obj
        for j in range(expr.size()):
            #print(i)
            #break
            v = expr.getVar(j)
            #print(v)
            coeff = expr.getCoeff(j)
            #newv = varDict[v.Varname]
            newv = m_c_a.getVarByName(v.Varname)
            #print(newv)
            #print(newv==v)
            newexpr_d.add(newv, coeff)
        d_obj_exp_list.append(newexpr_d)
        #m_c_a.addConstr(newexpr_p-newexpr_d, "==", 0, name="H_"+str(i+1)+'_objective')
        m_c_a.update()
    
    
    home_devs=[]
    home_reals=[]
    home_signed_devs=[]
    
    home_reals_except_hvac_and_pv=[]
    home_reals_hvac=[]
    home_reals_pv=[]
    for i in range(num_homes):
        names_to_retrieve=[]
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_wm_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_oven_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_dryer_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_hvac_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ewh_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ev_dev["+str(j)+"]")
            
        home_devs.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        
        """
        names_to_retrieve=[]
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_wm_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_oven_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_dryer_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_hvac_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ewh_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ev_signed_dev["+str(j)+"]")
            
        home_signed_devs.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        """
        

        names_to_retrieve=[]
        names_to_retrieve_except_hvac_and_pv=[]
        names_to_retrieve_hvac=[]
        names_to_retrieve_pv=[]
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_wm_real["+str(j)+"]")
            names_to_retrieve_except_hvac_and_pv.append("H"+str(i+1)+"_P_wm_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_oven_real["+str(j)+"]")
            names_to_retrieve_except_hvac_and_pv.append("H"+str(i+1)+"_P_oven_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_dryer_real["+str(j)+"]")
            names_to_retrieve_except_hvac_and_pv.append("H"+str(i+1)+"_P_dryer_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_hvac_real["+str(j)+"]")
            names_to_retrieve_hvac.append("H"+str(i+1)+"_P_hvac_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ewh_real["+str(j)+"]")
            names_to_retrieve_except_hvac_and_pv.append("H"+str(i+1)+"_P_ewh_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ev_real["+str(j)+"]")
            names_to_retrieve_except_hvac_and_pv.append("H"+str(i+1)+"_P_ev_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_pv_real["+str(j)+"]")
            names_to_retrieve_pv.append("H"+str(i+1)+"_P_pv_real["+str(j)+"]")
        
        home_reals.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        
        home_reals_except_hvac_and_pv.append([m_c_a.getVarByName(name) for name in names_to_retrieve_except_hvac_and_pv])
        home_reals_hvac.append([m_c_a.getVarByName(name) for name in names_to_retrieve_hvac])
        home_reals_pv.append([m_c_a.getVarByName(name) for name in names_to_retrieve_pv])
        
    
    #Note that primal objective function is somewhat different 
    #at the following indices. Let' s keep record of them.
    real_pow_idx=np.arange(len(home_reals[0]))
    real_pow_HVAC_idx=np.arange(3*horizon,4*horizon,1)
    real_pow_PV_idx=np.arange(6*horizon,7*horizon,1)

    real_pow_idx_remaining=np.setdiff1d(real_pow_idx,real_pow_HVAC_idx)
    real_pow_idx_remaining=np.setdiff1d(real_pow_idx_remaining,real_pow_PV_idx)
    
    #this for loop add constraints related to primal objective == dual objective
    for i in range(num_homes):
        

            
        cost=dual_list[i]['c']
        cost_dev=cost[:6*horizon]
        
            
        #m_c_a.addConstr(quicksum(home_devs[i][t]*cost_dev[t] for t in range(len(cost_dev))) +\
        #                quicksum(home_reals[i][t]*price_e[t%horizon] for t in real_pow_idx_remaining) +\
        #                quicksum(home_reals[i][t]*power_HVAC_list[i]*price_e[t%horizon] for t in real_pow_HVAC_idx) +\
        #                quicksum(home_reals[i][t]*-price_e[t%horizon] for t in real_pow_PV_idx ) -\
        #                d_obj_exp_list[i] <= epsilon[i],name="H_"+str(i+1)+"_objective") 
            
    
        m_c_a.addConstr(quicksum(home_devs[i][t]*cost_dev[t] for t in range(len(cost_dev))) +\
                        quicksum(home_reals_except_hvac_and_pv[i][t]*price_e[t%horizon] for t in range(len(home_reals_except_hvac_and_pv[0]))) +\
                        quicksum(home_reals_hvac[i][t]*power_HVAC_list[i]*price_e[t%horizon] for t in range(horizon)) +\
                        quicksum(home_reals_pv[i][t]*-price_e[t%horizon] for t in range(horizon) ) -\
                        d_obj_exp_list[i] <= epsilon[i],name="H_"+str(i+1)+"_objective") 
        
        
        #m_c_a.addConstr(quicksum(home_devs[i][t]*cost_dev[t] for t in range(len(cost_dev))) +\
        #                quicksum(home_reals_except_hvac_and_pv[i][t]*price_e[t%horizon] for t in range(len(home_reals_except_hvac_and_pv[0]))) +\
        #                quicksum(home_reals_hvac[i][t]*power_HVAC_list[i]*price_e[t%horizon] for t in range(horizon)) +\
        #                quicksum(home_reals_pv[i][t]*-price_e[t%horizon] for t in range(horizon) ) -\
        #                d_obj_exp_list[i] <= 0.001,name="H_"+str(i+1)+"_objective") 
            
        
        #m_c_a.addConstr(quicksum(home_devs[i][t]*cost_dev[t] for t in range(len(cost_dev))) +\
        #                quicksum(home_reals[i][t]*price_e[t%horizon] for t in real_pow_idx_remaining) +\
        #                quicksum(home_reals[i][t]*power_HVAC_list[i]*price_e[t%horizon] for t in real_pow_HVAC_idx) +\
        #                quicksum(home_reals[i][t]*-price_e[t%horizon] for t in real_pow_PV_idx ) -\
        #                d_obj_exp_list[i] <= 0.001,name="H_"+str(i+1)+"_objective") 
                
    #Set objective function of coordination agent
    deviation_loss=m_c_a.addVars(horizon,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="dev_loss")
    dev_loss_helper=m_c_a.addVars(horizon,lb=0,name="dev_loss_obj")
    
    #obj_term1=[]#holds the deviation from desired power consumption level.
    for j in range(horizon):
        #obj_term1=Q[j]-quicksum(home_reals[i][t*horizon+j] for i in range(num_homes)for t in range(7))
        #obj_term1=Q[j]-quicksum(home_reals[i][t*horizon+j] for i in range(num_homes)for t in range(6))
        obj_term1=Q[j]
        for t in range (6):
            if t==3:
                obj_term1=obj_term1- quicksum(home_reals[i][t*horizon+j]*power_HVAC_list[i] for i in range(num_homes))
            else:
                obj_term1=obj_term1 - quicksum(home_reals[i][t*horizon+j] for i in range(num_homes))
        
        
        m_c_a.addConstr(deviation_loss[j]== obj_term1)
        
        #m_c_a.addConstr(dev_loss_helper[j]>= obj_term1)
        #m_c_a.addConstr(dev_loss_helper[j]>= -obj_term1)

    m_c_a.addConstrs((dev_loss_helper[j]==abs_(deviation_loss[j]) for j in range(horizon)),name="tmp")
    
    
    obj_term2=[]#list holding the cost related to home deviation.
    for i in range(num_homes):
        cost=dual_list[i]['c']
        cost_dev=cost[:6*horizon]
        obj_term2.append(quicksum(home_devs[i][t]*cost_dev[t] for t in range(len(cost_dev))))
        
    
    #m_c_a.setObjective(quicksum(dev_loss_helper[i] for i in range(horizon))+\
    #                   quicksum(obj_term2[i] for i in range(num_homes))    ,GRB.MINIMIZE) 
    
    
    m_c_a.setObjective(quicksum(dev_loss_helper[i] for i in range(horizon))+\
                       quicksum(obj_term2[i] for i in range(num_homes))+\
                       quicksum(epsilon[i]*price_kwargs['gurobi_lambda_gap'] for i in range(num_homes))    ,GRB.MINIMIZE) 
    
    
    
        
        
    new_price=mean_vec[-horizon:]    
    #new_price=np.array([price_e[i].X for i in range(96)])
    """
    #gurobi price
    new_price=np.array([1.56464191, 1.56464191, 1.56464191, 1.90674554, 1.56464191,
           1.58193606, 1.56464191, 1.56767332, 1.56464191, 1.56464191,
           1.58193606, 1.56464191, 1.58193606, 1.56464191, 1.58193606,
           1.56464191, 1.58193606, 1.56464191, 1.61523773, 1.56464191,
           1.58193606, 1.56464191, 1.9281045 , 1.56464191, 1.58029538,
           1.56464191, 1.56464191, 1.58193606, 1.56464191, 1.58193606,
           1.56464191, 1.58193606, 1.56464191, 1.64708089, 1.66743077,
           1.67443156, 1.66743077, 1.66743077, 1.66743077, 1.66743077,
           1.66743077, 1.65260886, 1.64317953, 1.64708089, 1.64317953,
           1.64317953, 1.56464191, 1.58193606, 1.56464191, 1.58193606,
           1.56464191, 1.58193606, 1.56464191, 1.58193606, 1.56464191,
           1.58193606, 1.56464191, 1.56464191, 1.61523773, 1.56464191,
           1.61523773, 1.56464191, 1.58193606, 1.56464191, 1.64962463,
           1.56464191, 1.64962463, 1.64621094, 1.66743077, 1.66743077,
           1.67443156, 1.66743077, 1.66743077, 1.66743077, 1.6553367 ,
           1.64621094, 1.64621094, 0.35      , 0.55048684, 0.41473036,
           0.65422329, 0.50031335, 0.52226904, 0.35      , 0.58228943,
           0.42344514, 0.55725924, 0.42593405, 0.66667379, 0.51048041,
           0.5279501 , 0.3563135 , 0.58930552, 0.42817372, 0.52226904,
           0.35      ])
    
    #sdp price
    new_price=np.array([1.06309482, 1.05437505, 1.06476672, 1.05014277, 1.06448098,
           1.04912533, 1.06452775, 1.04925221, 1.06597739, 1.06616883,
           1.06721673, 1.06424964, 1.06696804, 1.06294483, 1.06477167,
           1.06176173, 1.05985013, 1.06141636, 1.05518226, 1.06120239,
           1.05127686, 1.0610979 , 1.04755825, 1.06202709, 1.06165095,
           1.06170664, 1.06160841, 1.06302611, 1.06032893, 1.06291758,
           1.06018772, 1.06020185, 1.05996137, 1.04674091, 1.06096138,
           1.02319362, 1.06038249, 0.99022251, 1.02571772, 0.93724797,
           1.02373579, 0.97609313, 1.0628527 , 1.04332149, 1.06272872,
           1.06329585, 1.05754987, 1.06068479, 1.05862111, 1.06151878,
           1.05925116, 1.06170017, 1.05930715, 1.0612664 , 1.05853544,
           1.06016581, 1.05701966, 1.05642302, 1.05027289, 1.05641311,
           1.05667539, 1.0558815 , 1.05783215, 1.05526063, 1.06067141,
           1.05748156, 1.04727546, 1.05965581, 1.04441328, 1.05291793,
           1.03409413, 1.0499468 , 1.03437801, 1.0503463 , 1.04045805,
           1.05455237, 1.05340231, 1.05219757, 1.04779234, 1.04493554,
           1.04587292, 1.04302349, 1.04366494, 1.04109581, 1.04182657,
           1.03973557, 1.04067082, 1.03912822, 1.04029128, 1.03934121,
           1.04080687, 1.0407376 , 1.04290911, 1.04443569, 1.04751436,
           1.04944949])
    #sdp price with less than equal to.
    new_price=np.array([1.35916017, 1.34678237, 1.34508655, 1.34522421, 1.34513781,
           1.3460792 , 1.34737801, 1.34740643, 1.34741828, 1.34609764,
           1.34515179, 1.34510779, 1.34515208, 1.34681545, 1.34930356,
           1.34929075, 1.34934175, 1.34930983, 1.34932642, 1.34928777,
           1.34927408, 1.34920562, 1.34910503, 1.34890834, 1.34853849,
           1.34786793, 1.34858813, 1.34895509, 1.34908315, 1.34921153,
           1.34920029, 1.34925117, 1.34917351, 1.34686492, 1.34529188,
           1.34518925, 1.34526288, 1.34598453, 1.34736251, 1.3469989 ,
           1.34726607, 1.34565451, 1.345005  , 1.34481923, 1.34459786,
           1.34563958, 1.34839555, 1.34883364, 1.34898191, 1.3491109 ,
           1.34911367, 1.34916716, 1.34913446, 1.34917007, 1.34913024,
           1.34915905, 1.34910717, 1.34909131, 1.34912499, 1.34910473,
           1.34914099, 1.34910057, 1.34912984, 1.34905776, 1.34686454,
           1.34679842, 1.34526634, 1.34529652, 1.34614255, 1.34617698,
           1.3473781 , 1.34742567, 1.34611978, 1.34612403, 1.34521444,
           1.34522644, 1.34675384, 1.34675033, 1.34902086, 1.3490041 ,
           1.34905599, 1.3490199 , 1.34906095, 1.34902038, 1.34906255,
           1.34902196, 1.34906867, 1.34903217, 1.34908955, 1.34907012,
           1.34916488, 1.34922051, 1.34946904, 1.34981924, 1.35063868,
           1.34904538])
    """
    
    set_price_constraints=m_c_a.addConstrs((price_e[i]==new_price[i] for i in range(horizon)),name="fixing price")
    m_c_a.update()
    m_c_a.Params.NonConvex=2
    #-1 automatic 0 primal 1 dual 2 barrier
    #m_c_a.Params.Method=0
    #m_c_a.Params.BarHomogeneous=1
    #m_c_a.Params.TimeLimit = 60
        
    
    #m_c_a.write('/Users/can/Desktop/sdp_model.lp')
    m_c_a.optimize()

    
    best_obj=m_c_a.objVal#244.398
    gurobi_mosek_price_gurobi_obj=m_c_a.objVal
    best_price=new_price
    
    
    gurobi_mosek_price_weak_duality_epsilon=[]
    gurobi_mosek_price_real_power=0
    for i in range (num_homes):
        
           gurobi_mosek_price_weak_duality_epsilon.append(epsilon[i].X)
           gurobi_mosek_price_real_power+=np.array([home_reals[i][0*horizon+k].X for k in range(horizon)])+\
                np.array([home_reals[i][1*horizon+k].X for k in range(horizon)])+\
                np.array([home_reals[i][2*horizon+k].X for k in range(horizon)])+\
                np.array([home_reals[i][3*horizon+k].X * power_HVAC_list[i] for k in range(horizon)])+\
                np.array([home_reals[i][4*horizon+k].X for k in range(horizon)])+\
                np.array([home_reals[i][5*horizon+k].X for k in range(horizon)])
    
    gurobi_mosek_price_obj_without_epsilons=m_c_a.objVal- np.sum(np.array([epsilon[i].X*price_kwargs['gurobi_lambda_gap'] for i in range (num_homes)]))
    print('QCQP objective without epsilon is %f'%(gurobi_mosek_price_obj_without_epsilons))
    
    counter=0
    for i in range(int(price_kwargs['num_samples'])):
        if i%1000 == 0:
            print('Price sampling Iteration: %d'%(i))
        sampled_sol = rng.multivariate_normal(mean=mean_vec, cov=cov_matrix, size=1)[0,:]
        new_price=sampled_sol[-horizon:]
        
        if price_kwargs['project_sampled_price'] == True:
            new_price[new_price<mean_price]=mean_price
            new_price[new_price>mean_price+p_ub]=mean_price+p_ub
        
        if price_kwargs['scale_sampled_price'] == True:
            #scale sampled price
            new_price=(new_price-min(new_price))/(max(new_price)-min(new_price))
            new_price=mean_price+p_ub*new_price
        
        #if np.all(new_price>0):
        if np.all(new_price>=mean_price) and np.all(new_price<=mean_price+p_ub):
        
            for j in range (horizon):
                f_c=m_c_a.getConstrByName('fixing price['+str(j)+']')
                m_c_a.remove(f_c)
            
            m_c_a.update()
            set_price_constraints=m_c_a.addConstrs((price_e[t]==new_price[t] for t in range(horizon)),name="fixing price")
            m_c_a.update()
            m_c_a.Params.LogToConsole=0
            #m_c_a.Params.NonConvex=2
            m_c_a.optimize()
            
            sum_eps=np.sum(np.array([epsilon[i].X*price_kwargs['gurobi_lambda_gap'] for i in range (num_homes)]))
            if m_c_a.objVal-sum_eps < best_obj:
                best_obj=m_c_a.objVal-sum_eps
                best_price=new_price
                counter+=1
                print("objective is improved")
                print(best_obj)
    
    
    print("objective has been improved %d times in %d trials."%(counter,price_kwargs['num_samples']))
    
    for j in range (horizon):
        f_c=m_c_a.getConstrByName('fixing price['+str(j)+']')
        m_c_a.remove(f_c)
    m_c_a.update()
    set_price_constraints=m_c_a.addConstrs((price_e[t]==best_price[t] for t in range(horizon)),name="fixing price")
    m_c_a.update()
    #m_c_a.Params.NonConvex=2
    m_c_a.optimize()
    
    gurobi_best_price_gurobi_obj=m_c_a.objVal
    
    
    
    
    
    
    QCQP_real_power=0
    #desirable_power=0#is calculated above.
    home_weak_duality_epsilon=[]
    for i in range (num_homes):
        
           home_weak_duality_epsilon.append(epsilon[i].X) 
           QCQP_real_power+=np.array([home_reals[i][0*horizon+k].X for k in range(horizon)])+\
                np.array([home_reals[i][1*horizon+k].X for k in range(horizon)])+\
                np.array([home_reals[i][2*horizon+k].X for k in range(horizon)])+\
                np.array([home_reals[i][3*horizon+k].X * power_HVAC_list[i] for k in range(horizon)])+\
                np.array([home_reals[i][4*horizon+k].X for k in range(horizon)])+\
                np.array([home_reals[i][5*horizon+k].X for k in range(horizon)])
                
    
    
    QCQP_obj=m_c_a.objVal- np.sum(np.array([epsilon[i].X*price_kwargs['gurobi_lambda_gap'] for i in range (num_homes)]))
    
    
    new_price=np.array([price_e[i].X for i in range (horizon)])
    
    
    gurobi_summary={
        
                    'gurobi_mosek_price_gurobi_obj':gurobi_mosek_price_gurobi_obj,
                    'gurobi_mosek_price_home_weak_duality_epsilon':gurobi_mosek_price_weak_duality_epsilon,
                    'gurobi_mosek_price_real_power':gurobi_mosek_price_real_power,
                    'gurobi_mosek_price_obj_without_epsilons':gurobi_mosek_price_obj_without_epsilons,
                    'gurobi_lambda_gap':price_kwargs['gurobi_lambda_gap'],
                    
                    
                    'optimal_price':new_price,#after price samplings.
                    'gurobi_best_price_gurobi_obj':gurobi_best_price_gurobi_obj,
                    'gurobi_best_price_home_weak_duality_epsilon':home_weak_duality_epsilon,
                    'gurobi_best_price_real_power':QCQP_real_power,
                    'gurobi_best_price_obj_without_epsilons':QCQP_obj,
                    
        

                   
                   'price_lb': mean_price,
                   'price_ub': mean_price+p_ub,
                   
                   
                   'Q'        : Q,
                   'MIPGAP':m_c_a.MIPGap,
                   'num_homes':num_homes,
                   
                   
                   'num_samples':price_kwargs['num_samples'],
                   'project_sampled_price':price_kwargs['project_sampled_price'],
                   'scale_sampled_price':price_kwargs['scale_sampled_price']
                   }
    
    return gurobi_summary
    
    
    
    
    
    
    
    
    
    
            