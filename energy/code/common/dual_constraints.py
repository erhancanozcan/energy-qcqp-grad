
from gurobipy import *
import numpy as np


def solve_dual(dual):
    
        RHS=dual['RHS']
        A=dual['A']
        c=dual['c']
        dual_var_sign=dual['dual_var_sign']
        dual_const_sign=dual['dual_const_sign']
        
        num_duals=len(dual_var_sign)
        
        
        
        coeff=A.toarray()
        
        leq_index_list=np.array(dual_var_sign)==-1
        eq_index_list=np.array(dual_var_sign)==0
        #no positive index list because we do not have >= sign in primal constraints.
        #contraints in primal are either <= or ==. Try to keep this setting, otherwise
        #you need to add here positive index_list, and you need a quicksum over positive variables.
        #Instead manipulate the primal constraint.

        coeff_leq=coeff[leq_index_list]
        coeff_eq=coeff[eq_index_list]  

        m_d = Model("dual")
        
        
        for i in range(len(dual_var_sign)):
            if dual_var_sign[i]==1:
                m_d.addVar(lb=0,name=("dual_pos"+str(i)))
            elif dual_var_sign[i]==0:
                m_d.addVar(lb=-GRB.INFINITY,name=("dual_free"+str(i)))
            else:
                m_d.addVar(lb=-GRB.INFINITY,ub=0,name=("dual_neg"+str(i)))
                
        m_d.update()   
        
        
        neg=[var for var in m_d.getVars() if "dual_neg" in var.VarName]
        free=[var for var in m_d.getVars() if "dual_free" in var.VarName]
        pos=[var for var in m_d.getVars() if "dual_pos" in var.VarName]
        var=[var for var in m_d.getVars() ]
        
        d_obj=quicksum(var[i]*RHS[i] for i in range (num_duals))
        m_d.setObjective(d_obj, GRB.MAXIMIZE)
        #m_d.setObjective(quicksum(var[i]*RHS[i] for i in range (num_duals)), GRB.MAXIMIZE)
        
        
        
        for j in range (len(dual_const_sign)):
            sign=dual_const_sign[j]
            #dual_constrained_sign is determined according to sign of decision variables in primal
            #mpc. 
            
            if sign == -1:
                m_d.addConstr(quicksum(neg[i]*coeff_leq[i,j] for i in range (len(neg))) +\
                              quicksum(free[i]*coeff_eq[i,j] for i in range (len(free)))<=c[j])
            elif sign == 0:
                m_d.addConstr(quicksum(neg[i]*coeff_leq[i,j] for i in range (len(neg))) +\
                              quicksum(free[i]*coeff_eq[i,j] for i in range (len(free)))==c[j])
                    
                    
        m_d.Params.Method=0
        m_d.optimize()
        return m_d,d_obj
                
            
#solve_dual(dual) 
#deneme=m_d.pi


def add_dual_constraints(dual,m_d):
    
        RHS=dual['RHS']
        A=dual['A']
        c=dual['c']
        dual_var_sign=dual['dual_var_sign']
        dual_const_sign=dual['dual_const_sign']
        
        num_duals=len(dual_var_sign)
        
        
        
        coeff=A.toarray()
        
        leq_index_list=np.array(dual_var_sign)==-1
        eq_index_list=np.array(dual_var_sign)==0
        #no positive index list because we do not have >= sign in primal constraints.
        #contraints in primal are either <= or ==. Try to keep this setting, otherwise
        #you need to add here positive index_list, and you need a quicksum over positive variables.
        #Instead manipulate the primal constraint.

        coeff_leq=coeff[leq_index_list]
        coeff_eq=coeff[eq_index_list]  

        #m_d = Model("dual")
        
        
        for i in range(len(dual_var_sign)):
            if dual_var_sign[i]==1:
                m_d.addVar(lb=0,name=("dual_pos"+str(i)))
            elif dual_var_sign[i]==0:
                m_d.addVar(lb=-GRB.INFINITY,name=("dual_free"+str(i)))
            else:
                m_d.addVar(lb=-GRB.INFINITY,ub=0,name=("dual_neg"+str(i)))
                
        m_d.update()   
        
        
        neg=[var for var in m_d.getVars() if "dual_neg" in var.VarName]
        free=[var for var in m_d.getVars() if "dual_free" in var.VarName]
        pos=[var for var in m_d.getVars() if "dual_pos" in var.VarName]
        var=[var for var in m_d.getVars() ]
        
        #m_d.setObjective(quicksum(var[i]*RHS[i] for i in range (num_duals)), GRB.MAXIMIZE)
        d_obj=quicksum(var[i]*RHS[i] for i in range (num_duals))
        
        
        
        for j in range (len(dual_const_sign)):
            sign=dual_const_sign[j]
            #dual_constrained_sign is determined according to sign of decision variables in primal
            #mpc. 
            
            if sign == -1:
                m_d.addConstr(quicksum(neg[i]*coeff_leq[i,j] for i in range (len(neg))) +\
                              quicksum(free[i]*coeff_eq[i,j] for i in range (len(free)))<=c[j])
            elif sign == 0:
                m_d.addConstr(quicksum(neg[i]*coeff_leq[i,j] for i in range (len(neg))) +\
                              quicksum(free[i]*coeff_eq[i,j] for i in range (len(free)))==c[j])
                    
                    
        #m_d.Params.Method=0
        #m_d.optimize()
        m_d.update()
        return m_d,d_obj
    
    


    
    
