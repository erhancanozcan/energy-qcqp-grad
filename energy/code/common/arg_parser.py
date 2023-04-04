"""Creates command line parser for train.py."""
import argparse

parser = argparse.ArgumentParser()


# Setup
##########################################
parser.add_argument('--runs',help='number of trials',type=int,default=1)
parser.add_argument('--runs_start',help='starting trial index',
    type=int,default=0)
parser.add_argument('--cores',help='number of processes',type=int)
parser.add_argument('--seed',help='master seed',type=int,default=0)
parser.add_argument('--setup_seed',help='setup seed',type=int)

parser.add_argument('--save_path',help='save path',type=str,default='./energy/logs')
parser.add_argument('--save_file',help='save file name',type=str)

set_price_kwargs = ['price_file','price_path']

parser.add_argument('--price_path',help='price path',type=str,default='./energy/price')
parser.add_argument('--price_file',help='price file name',type=str)

#Home
home_kwargs=['s_effect']
parser.add_argument('--s_effect',help='seasonal effect for HVAC 1 heating minus 1 cooling',type=int,default=1)

#Coordination Agent
ca_kwargs=['num_houses', 'horizon', 'price', 'Q','lambda_gap', 'mipgap', 'timelimit',
           'p_lb','p_ub','max_grad_steps','lr', 'alpha','provide_solution','lr_update_rule',
           'num_epochs','batch_size','min_grad_norm','obj_threshold','min_iter']

parser.add_argument('--num_houses',help='number of houses in the community',type=int,default=10)
parser.add_argument('--horizon',help='number of time intervals in next 24 hours',type=int,default=96)
parser.add_argument('--price',help='mean electricity price Kwh',type=float,default=0.35)
parser.add_argument('--Q',help='desired agregated power level in KwH',type=float,default=-1.0)
parser.add_argument('--lambda_gap',help='duality gap penalizer coefficient',type=float,default=10.0)
parser.add_argument('--mipgap',help='mipgap value of the QCQP problem',type=float,default=1e-4)
parser.add_argument('--timelimit',help='timelimit in seconds for coordination agent problem',type=float,default=60)
parser.add_argument('--p_ub',help='upper bound on the price',type=float,default=1.0)
parser.add_argument('--p_lb',help='lower bound on the price',type=float,default=0.1)
parser.add_argument('--max_grad_steps',help='maximum number of gradient steps',type=int,default=100)
parser.add_argument('--lr',help='learning rate of gradient descent',type=float,default=1.0)#1e-5
parser.add_argument('--alpha',help='momentum coefficient',type=float,default=9e-1)
parser.add_argument('--lr_update_rule',help='1 constant, 2 scale by iteration number,  3 scale by gradient, 4 momentum,5 Adam',type=int,default=5)
parser.add_argument('--provide_solution',help='provides an initial feasible solution',action='store_true')
parser.add_argument('--num_epochs',help='number of epochs',type=int,default=60)
parser.add_argument('--batch_size',help='its maximum value must be less than num_houses',type=int,default=5)
parser.add_argument('--min_grad_norm',help='minimum gradient norm to stop iterations',type=float,default=1e-3)
parser.add_argument('--obj_threshold',help='objective improvement rate',type=float,default=1e-3)
parser.add_argument('--min_iter',help='minimum number of iterations before stopping',type=int,default=5)



price_kwargs=['project_sampled_price','scale_sampled_price','num_samples','gurobi_lambda_gap']
parser.add_argument('--project_sampled_price',help='project random price to the closest feasible',action='store_true')#default must be store_true
parser.add_argument('--scale_sampled_price',help='scale random price to feasible interval',action='store_false')#default must be store_true
parser.add_argument('--num_samples',help='number of random price samples',type=float, default=10)
parser.add_argument('--gurobi_lambda_gap',help='duality gap penalizer of gurobi',type=float,default=10.0)

slp_kwargs=['n_repeat']

parser.add_argument('--n_repeat',help='number of successive linear program optimizations',type=int,default=25)







# For export to solver.py
#########################################
def create_train_parser():
    return parser

all_kwargs={
    'set_price_kwargs':set_price_kwargs,
    'home_kwargs': home_kwargs,
    'ca_kwargs':   ca_kwargs,
    'slp_kwargs':  slp_kwargs,
    'price_kwargs':  price_kwargs
    }









