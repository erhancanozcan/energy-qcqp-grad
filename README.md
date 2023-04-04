# A Stackelberg Game to Control the Overall Load Consumption of a Residential Neighborhood

This repository contains the official implementation for the following paper on demand response program:


* [GitHub Pages](https://pages.github.com/).


While the first command below can be used to solve the centralized Quadratically Constrained Quadratic Program, the second command employs the proposed gradient-based optimization framework to solve the problem in a distributed way. Both commands require Gurobi to be installed.


```
nohup python -u -m energy.code.qcqp_gradient --s_effect 1 \
        --num_houses 50 --batch_size 25 --seed 0  \
        --timelimit 900 \
        --lr 1e-1 --lr_update_rule 5 --num_epochs 5 \
        --Q -1.0 --obj_threshold 1e-3 \
        --save_file qcqp_grad_s0_h50bs25_lur5_lr1en1_numeps5_obj1en3_Qn1w &
  
nohup python -u -m energy.code.qcqp_centralized --s_effect 1 --num_houses 10  \
        --seed 0 --timelimit 900 --provide_solution \
        --Q -1.0 \
        --save_file qcqp_centralized_s0_wthsol_10h_Qn1winter &
```  
