from hyperopt import Trials, fmin, hp, tpe
import numpy as np
from GP_Goddard_forR import GP_param_tuning

space = {#'pop': hp.uniform('pop', 50, 200),
         #'gen': hp.uniform('gen', 10, 30),
         'lambda': hp.uniform('lambda', 1, 1.6),
         'cxpb': hp.uniform('cxpb', 0.5, 0.8),
         'strp': hp.uniform('strp', 0.1, 0.7),
         'nEph': hp.randint('nEph', 20),
         'TriAdd': hp.choice('TriAdd', [0, 1]),
         'Add': hp.choice('Add', [0, 1]),
         'sub': hp.choice('sub', [0, 1]),
         'mul': hp.choice('mul', [0, 1]),
         'div': hp.choice('div', [0, 1]),
         'pow': hp.choice('pow', [0, 1]),
         'abs': hp.choice('abs', [0, 1]),
         'sqrt': hp.choice('sqrt', [0, 1]),
         'log': hp.choice('log', [0, 1]),
         'exp': hp.choice('exp', [0, 1]),
         'sin': hp.choice('sin', [0, 1]),
         'cos': hp.choice('cos', [0, 1]),
         'pi': hp.choice('pi', [0, 1]),
         'e': hp.choice('e', [0, 1]),
         'fit1': hp.uniform('fit1', 0.0, 1.0),
         'fit2': hp.uniform('fit2', 0.0, 1.0),
         'fit3': hp.uniform('fit3', 0.1, 1.0),
         'fit4': hp.uniform('fit4', 0.0, 1.0),
         'fit5': hp.uniform('fit5', 0.1, 1.0)}

### step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 50

# Optimize
best = fmin(GP_param_tuning, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=bayes_trials)
print(best)

