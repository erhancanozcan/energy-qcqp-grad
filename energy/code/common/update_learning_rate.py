import numpy as np



def update_lr(rule,lr,iter_number,grad_list):
    if iter_number==0:
        return lr
    else:
        if rule==1:
            return lr
        elif rule==2:
            return lr/(iter_number)**0.5
        elif rule==3:
            #print('can')
            g_s=np.array(grad_list)
            return lr/(np.sum(g_s**2)**0.5)
        
        else:
            raise NotImplementedError