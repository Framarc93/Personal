import numpy as np
from GP import gp_fun
from MGGP import mggp_fun
from MGGP_SNGP import mggp_sngp_fun
from SNGP import sngp_fun

def GP_choice(gp, eval_fun, interval, Ngen, nEph, terminals, ev):
    limit_height = 17  # Max height (complexity) of the controller law
    limit_size = 100  # Max size
    size_pop = 200
    if gp == "GP":
        best_fit, best_ind = gp_fun(nEph, limit_height, limit_size, size_pop, eval_fun, interval, Ngen*3, terminals, ev)
    elif gp == "MGGP":
        best_fit, best_ind = mggp_fun(nEph, eval_fun, interval, Ngen, terminals)
    elif gp == "SNGP":
        best_fit, best_ind = sngp_fun(eval_fun, interval, Ngen, nEph, terminals)
    elif gp == "MGGP+SNGP":
        best_fit, best_ind = mggp_sngp_fun(eval_fun, interval, Ngen, nEph, terminals)
    return best_fit, best_ind