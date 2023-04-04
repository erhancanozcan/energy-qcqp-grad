import numpy as np



class optimizer:
    
    def __init__(self,lr=1e-5,alpha=1e-3,rule=1):
        
        self.lr=lr
        self.alpha=alpha
        self.rule=rule
        self.t=1
        self.change=0
        self.cumulative_grad_square=0
        
        self.beta1=0.9
        self.beta2=0.999
        
        self.m=0
        self.v=0
        
    
    def update_price(self,price,grad):
        
        if self.rule==1:
            #constant learning rate
            price=price-self.lr*grad
            
            self.t+=1
            return price
            
            
        
        elif self.rule==2:
            #scale learning rate by the square root of iteration number
            price=price-(self.lr/(self.t**0.5))*grad
            
            self.t+=1
            return price
        
        elif self.rule==3:
            #scale lr by the square root of squares sum of the gradients
            self.cumulative_grad_square=np.dot(grad,grad)
            
            price=price-(self.lr/(self.cumulative_grad_square**0.5))*grad
            
            self.t+=1
            return price
        
        elif self.rule==4:
            #use momentum idea
            
            self.change = self.lr*grad + self.alpha*self.change
            
            price= price - self.change
            
            self.t+=1
            return price
        
        elif self.rule==5:
            #adam
            self.m=self.beta1*self.m + (1-self.beta1)*grad
            self.v=self.beta2*self.v + (1-self.beta2)*np.dot(grad,grad)
            
            m_hat=self.m/(1-self.beta1)
            v_hat=self.v/(1-self.beta2)
            
            price= price - self.lr*m_hat/(v_hat**0.5+1e-8)
            
            self.t+=1
            return price
        
        else:
            #whenever you add need rule, you may need to modify update_best method.
            raise NotImplementedError
            
            
    
    def update_best(self,obj,price):
        
        self.best_obj=obj
        self.best_price=price
        if self.t>1:
            #otherwise, we may have zero by division.
            if self.rule==2:
                self.best_lr=(self.lr/((self.t-1)**0.5))
            elif self.rule==3:
                self.best_lr=(self.lr/(self.cumulative_grad_square**0.5))
            else:
                self.best_lr=None
            
        
        
    
    def reset_stats(self):
        self.t=1
        self.change=0
        self.cumulative_grad_square=0
        
            
            
        
        
        