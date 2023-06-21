# A Stackelberg Game to Control the Overall Load Consumption of a Residential Neighborhood

This repository contains the official implementation for the following paper on demand response program:


* [A Stackelberg Game to Control the Overall Load Consumption of a Residential Neighborhood](https://arxiv.org/abs/2306.10935)


This paper proposes a demand response program with dynamic pricing to control the overall load consumption of a residential neighborhood. The complexity of the proposed problem grows as the number of participating homes increases. To be able to solve the proposed problem efficiently, we develop a gradient-based distributed optimization framework. We show the benefits of utilizing our optimization approach over solving the centralized problem using a commercial solver by conducting various experiments in a simulated environment.

Please consider citing our paper as follows:

```
@misc{ozcan2023stackelberg,
      title={A Stackelberg Game Approach to Control the Overall Load Consumption of a Residential Neighborhood}, 
      author={Erhan Can Ozcan and Ioannis Ch. Paschalidis},
      year={2023},
 howpublished={arXiv preprint},
 note={{arXiv:2306.10935}},
}
``` 

## Solvers and Results

While the first command below can be used to solve the centralized Quadratically Constrained Quadratic Program, the second command utilizes the proposed gradient-based optimization framework to solve the problem in a distributed way. Both commands require Gurobi to be installed. 


```
nohup python -u -m energy.code.qcqp_gradient --s_effect 1 \
        --num_houses 50 --batch_size 25 --seed 0  \
        --timelimit 900 \
        --lr 1e-1 --lr_update_rule 5 --num_epochs 5 \
        --Q -1.0 --obj_threshold 1e-3 \
        --save_file qcqp_grad_s0_h50bs25_lur5_lr1en1_numeps5_obj1en3_Qn1w &
  
nohup python -u -m energy.code.qcqp_centralized --s_effect 1 --num_houses 50  \
        --seed 0 --timelimit 900 --provide_solution \
        --Q -1.0 \
        --save_file qcqp_centralized_s0_wthsol_50h_Qn1winter &
```  

For more information on the inputs accepted by qcqp_gradient and qcqp_centralized, use the --help option or reference energy/code/common/arg_parser.py. The results of the experiments are saved in the energy/logs/ folder upon completion.
