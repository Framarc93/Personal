def GP_param_tuning(param):
  
    '''param = [size_pop, num_gen, mu_coeff, lambda_coeff, cxpb, str_mutpb, fitness_size, parsimony_size, add_flag, sub_flag, mul_flag, div_flag, pow_flag, abs_flag, sqrt_flag, log_flag, exp_flag,
                sin_flag, cos_flag, pi_flag, e_flag, r1_flag, r2_flag, r3_flag, r4_flag]
       param_init = [100, 100, 1, 1.5, 0.6, 0.35, 0.6, 4, 1.4, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
       param_max = [300, 300, 1.5, 2, 1.0, 1.0, 0.8, , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
       param_min = [10, 10, 1, 1, 0.0, 0.0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'''
    from scipy.integrate import solve_ivp
    import numpy as np
    import operator
    #import pygraphviz as pgv
    import random
    from deap import gp, algorithms, base, creator, tools
    #import matplotlib.pyplot as plt
    #import sys
    import timeit
    import multiprocessing
    from scipy.interpolate import PchipInterpolator
    from functools import partial
    import datetime
    import math

    '''References:
        [1] - Automatic creation of human competitive Programs and Controllers by Means of Genetic Programming. Koza, Keane, Yu, Bennet, Mydlowec. 2000'''
    if 'pset' in locals() or 'pset' in globals():
      del pset, toolbox
    print(param)
    def Abs(x):
        return abs(x)

    def Div(left, right):
        try:
            x = left / right
            return x
        except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
            return 0.0


    def Mul(left, right):
        try:
            #np.seterr(invalid='raise')
            return left * right
        except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
            return left


    def Sqrt(x):
        try:
            if x > 0:
                return np.sqrt(x)
            else:
                return abs(x)
        except (
        RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            return 0


    def Log(x):
        try:
            if x > 0:
                return np.log(x)
            else:
                return abs(x)
        except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            return np.e


    def Exp(x):
        try:
            return np.exp(x)
        except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            return 1


    def Sin(x):
        try:
            return np.sin(x)
        except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            return 0


    def Cos(x):
        try:
            return np.cos(x)
        except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            return 1


    def xmate(ind1, ind2):
        i1 = random.randrange(len(ind1))
        i2 = random.randrange(len(ind2))
        ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
        return ind1, ind2


    def xmut(ind, expr, strp):

        i1 = random.randrange(len(ind))
        i2 = random.randrange(len(ind[i1]))
        choice = random.random()
        if choice < strp:
            indx = gp.mutUniform(ind[i1], expr, pset=pset)
            ind[i1] = indx[0]
            return ind,
        else:
            '''this part execute the mutation on a random constant'''
            try:
                val = float(ind[i1][i2].value)
                new_val = round(random.uniform(-10, 10), 4)
                #new_val_gauss = np.random.normal(ind[i1][i2].value, 1.0)  # new value of the constant determined by gaussian distribution suggested by Koza in [1]
                ind[i1][i2].value = new_val
                ind[i1][i2].name = "{}".format(new_val)
                return ind,
            except (ValueError, AttributeError):
                indx = gp.mutUniform(ind[i1], expr, pset=pset)
                ind[i1] = indx[0]
                return ind,


    # Direct copy from tools - modified for individuals with GP trees in an array
    def xselDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first):
        assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."
        fitness_size = int(fitness_size)
        def _sizeTournament(individuals, k, select):
            chosen = []
            for i in range(k):
                # Select two individuals from the population
                # The first individual has to be the shortest
                prob = parsimony_size / 2.
                ind1, ind2 = select(individuals, k=2)

                lind1 = sum([len(gpt) for gpt in ind1]) #extra
                lind2 = sum([len(gpt) for gpt in ind2]) #extra
                if lind1 > lind2:
                    ind1, ind2 = ind2, ind1
                elif lind1 == lind2:
                    # random selection in case of a tie
                    prob = 0.5

                # Since size1 <= size2 then ind1 is selected
                # with a probability prob
                chosen.append(ind1 if random.random() < prob else ind2)

            return chosen

        def _fitTournament(individuals, k, select):
            chosen = []
            for i in range(k):
                aspirants = select(individuals, k=fitness_size)
                chosen.append(max(aspirants, key=operator.attrgetter("fitness")))
            return chosen

        if fitness_first:
            tfit = partial(_fitTournament, select=tools.selRandom)
            return _sizeTournament(individuals, k, tfit)
        else:
            tsize = partial(_sizeTournament, select=tools.selRandom)
            return _fitTournament(individuals, k, tsize)


    start = timeit.default_timer()

    ###############################  S Y S T E M - P A R A M E T E R S  ####################################################

    class Rocket:

        def __init__(self):
            self.GMe = 3.986004418 * 10 ** 14  # Earth gravitational constant [m^3/s^2]
            self.Re = 6371.0 * 1000  # Earth Radius [m]
            self.Vr = np.sqrt(self.GMe / self.Re)  # m/s
            self.H0 = 10.0  # m
            self.V0 = 0.0
            self.M0 = 100000.0  # kg
            self.Mp = self.M0 * 0.99
            self.Cd = 0.6
            self.A = 4.0  # m2
            self.Isp = 300.0  # s
            self.g0 = 9.80665  # m/s2
            self.Tmax = self.M0 * self.g0 * 1.5
            self.MaxQ = 14000.0  # Pa
            self.MaxG = 8.0  # G
            self.Htarget = 400.0 * 1000  # m
            self.Rtarget = self.Re + self.Htarget  # m/s
            self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s
            self.cosOld = 0
            self.eqOld = np.zeros((0))
            self.ineqOld = np.zeros((0))
            self.varOld = np.zeros((0))

        @staticmethod
        def air_density(h):
            beta = 1/8500.0  # scale factor [1/m]
            rho0 = 1.225  # kg/m3
            return rho0*np.exp(-beta*h)

    Nstates = 5
    Ncontrols = 2

    old = 0
    
    size_pop = int(param[0])  # Pop size
    
    size_gen = int(param[1])  # Gen size
    Mu = int(size_pop)
    Lambda = int(size_pop * param[2])

    limit_height = 20  # Max height (complexity) of the controller law
    limit_size = 400  # Max size (complexity) of the controller law

    #nbCPU = multiprocessing.cpu_count()

    ################################# M A I N ###############################################

    def main(size_gen, size_pop, Mu, Lambda, param):

        global Rfun, Thetafun, Vrfun, Vtfun, mfun, flag, pas
        global tfin, fitness_old1, fitness_old2, fitness_old3, fitness_old4, fitness_old5
        flag = False
        pas = False
        fitness_old1 = 1e5
        fitness_old3 = 1e5
        fitness_old2 = 1e5
        fitness_old4 = 1e5
        fitness_old5 = 1e5
        
        Rref = np.load("R.npy")
        Thetaref = np.load("Theta.npy")
        Vrref = np.load("Vr.npy")
        Vtref = np.load("Vt.npy")
        mref = np.load("m.npy")
        tref = np.load("time.npy")
        tfin = tref[-1]
        
        Rfun = PchipInterpolator(tref, Rref)
        Thetafun = PchipInterpolator(tref, Thetaref)
        Vrfun = PchipInterpolator(tref, Vrref)
        Vtfun = PchipInterpolator(tref, Vtref)
        mfun = PchipInterpolator(tref, mref)

        del Rref, Thetaref, Vrref, Vtref, mref, tref
        #pool = multiprocessing.Pool(nbCPU)

        toolbox.register("map", map)
        
        print("INITIAL POP SIZE: %d" % size_pop)

        print("GEN SIZE: %d" % size_gen)

        print("\n")

        random.seed()

        pop = toolbox.population(n=size_pop)
        history.update(pop)
        # hof = tools.HallOfFame(size_gen) ### OLD ###
        hof = tools.HallOfFame(1) ### NEW ###
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        #stats_size = tools.Statistics(len)
        #stats_height = tools.Statistics(operator.attrgetter("height"))
        mstats = tools.MultiStatistics(fitness=stats_fit)

        mstats.register("avg", np.mean, axis=0)
        mstats.register("min", np.min, axis=0)
        mstats.register("max", np.max, axis=0)
        
        
        ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

        # pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, 0.6, 0.2, size_gen, stats=mstats, halloffame=hof,
                                             # verbose=True)  ### OLD ###
        try:
          pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=Mu, lambda_=Lambda, cxpb=param[3], mutpb=param[4], ngen=size_gen,
                                stats=mstats, halloffame=hof, verbose=True)  ### NEW ###
  
      ####################################################################################################################
  
          stop = timeit.default_timer()
          total_time = stop - start
          tformat = str(datetime.timedelta(seconds=int(total_time)))
          sys.stdout.write("TOTAL RUNNING TIME: {} \n".format(tformat))
          
          best_fit = sum(hof[0].fitness.values)
        
          #pool.close()
          return pop, log, hof, best_fit
        except(RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, IndexError):
          return 1, 1, 1, 1e10
    

    ##################################  F I T N E S S    F U N C T I O N    ################################################
  
    def evaluate(individual):
        global flag
        global pas
        global fitnnesoldvalue, fitness_old1, fitness_old2, fitness_old3, fitness_old4, fitness_old5
        global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun
        global tfin

        old = 0

        flag = False
        pas = False

        # Transform the tree expression in a callable function

        fTr = toolbox.compile(expr=individual[0])
        fTt = toolbox.compile(expr=individual[1])
        x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

        def sys(t, x):

            global oldTr, oldTt
            global flag

            # State Variables

            R = x[0]
            theta = x[1]
            Vr = x[2]
            Vt = x[3]
            m = x[4]

            if R<0 or np.isnan(R):
                R = obj.Re
                flag = True
            if np.isinf(R) or R > obj.Rtarget:
                R = obj.Rtarget
                flag = True
            if m<0 or np.isnan(m):
                m = obj.M0-obj.Mp
                flag = True
            elif m>obj.M0 or np.isinf(m):
                m = obj.M0
                flag = True
            if abs(Vr) > 1e4 or np.isinf(Vr):
                if Vr > 0:
                    Vr = 1e4
                    flag = True
                else:
                    Vr = -1e4
                    flag = True
            if abs(Vt) > 1e4 or np.isinf(Vt):
                if Vt > 0:
                    Vt = 1e4
                    flag = True
                else:
                    Vt = -1e4
                    flag = True

            r = Rfun(t)
            th = Thetafun(t)
            vr = Vrfun(t)
            vt = Vtfun(t)
            mf = mfun(t)

            er = r - R
            et = th - theta
            evr = vr - Vr
            evt = vt - Vt
            em = mf - m
            dxdt = np.zeros(Nstates)
            #print("Ft: ", fTt(er, et, evr, evt, em))
            #print("Fr: ", fTr(er, et, evr, evt, em))
            rho = obj.air_density(R - obj.Re)
            Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
            Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
            g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
            g0 = obj.g0
            Isp = obj.Isp

            Tr = fTr(er, et, evr, evt, em)
            Tt = fTt(er, et, evr, evt, em)

            if abs(fTr(er, et, evr, evt, em)) > obj.Tmax or np.isinf(fTr(er, et, evr, evt, em)):
                Tr = obj.Tmax
                flag = True

            elif fTr(er, et, evr, evt, em) < 0.0 or np.isnan(fTr(er, et, evr, evt, em)):
                Tr = 0.0
                flag = True

            if abs(fTt(er, et, evr, evt, em)) > obj.Tmax or np.isinf(fTt(er, et, evr, evt, em)):
                Tt = obj.Tmax
                flag = True

            elif fTt(er, et, evr, evt, em) < 0.0 or np.isnan(fTt(er, et, evr, evt, em)):
                Tt = 0.0
                flag = True

            dxdt[0] = Vr
            dxdt[1] = Vt / R
            dxdt[2] = Tr / m - Dr / m - g + Vt ** 2 / R
            dxdt[3] = Tt / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(Tr ** 2 + Tt ** 2) / g0 / Isp

            return dxdt

        sol = solve_ivp(sys, [0.0, tfin], x_ini, first_step=0.0001)
        y1 = sol.y[0, :]
        y2 = sol.y[1, :]
        y3 = sol.y[2, :]
        y4 = sol.y[3, :]
        y5 = sol.y[4, :]
        tt = sol.t
        if sol.t[-1] != tfin:
            flag = True
        pp = 0
        r = np.zeros(len(tt), dtype='float')
        theta = np.zeros(len(tt), dtype='float')
        vr = np.zeros(len(tt), dtype='float')
        vt = np.zeros(len(tt), dtype='float')
        m = np.zeros(len(tt), dtype='float')
        for i in tt:
            r[pp] = Rfun(i)
            theta[pp] = Thetafun(i)
            vr[pp] = Vrfun(i)
            vt[pp] = Vtfun(i)
            m[pp] = mfun(i)
            pp += 1

        err1 = (r - y1) / obj.Htarget
        err2 = np.rad2deg(theta - y2)/60
        err3 = (vr - y3) / obj.Vtarget
        err4 = (vt - y4) / obj.Vtarget
        err5 = (m - y5) / obj.M0

        # STEP TIME SIZE
        i = 0
        pp = 1
        step = np.zeros(len(y1), dtype='float')
        step[0] = 0.0001
        while i < len(tt) - 1:
            step[pp] = tt[i + 1] - tt[i]
            i = i + 1
            pp = pp + 1

        # INTEGRAL OF ABSOLUTE ERROR (PERFORMANCE INDEX)
        IAE = np.zeros((5, len(err1)))
        j = 0
        alpha = 0.1
        for a, b, c, d, e, n in zip(err1, err2, err3, err4, err5, step):
            IAE[0][j] = n * abs(a)
            IAE[1][j] = n * abs(b)
            IAE[2][j] = n * abs(c)  # + alpha * abs(m))
            IAE[3][j] = n * abs(d)
            IAE[4][j] = n * abs(e)
            j = j + 1

        # PENALIZING INDIVIDUALs
        #For the stats if the multiprocessing is used, there could be problems to print the correct values (parallel process(?))

        if flag is True:
            pas = True
            x = [np.random.uniform(fitness_old1 * 1.5, fitness_old1 * 1.6),
             np.random.uniform(fitness_old2 * 1.5, fitness_old2 * 1.6),
             np.random.uniform(fitness_old3 * 1.5, fitness_old4 * 1.6),
             np.random.uniform(fitness_old4 * 1.5, fitness_old4 * 1.6),
             np.random.uniform(fitness_old5 * 1.5, fitness_old5 * 1.6)]

        if flag is False:
            fitness1 = sum(IAE[0])
            fitness2 = sum(IAE[1])
            fitness3 = sum(IAE[2])
            fitness4 = sum(IAE[3])
            fitness5 = sum(IAE[4])
            if fitness1 < fitness_old1:
                fitness_old1 = fitness1
            if fitness2 < fitness_old2:
                fitness_old2 = fitness2
            if fitness3 < fitness_old3:
                fitness_old3 = fitness3
            if fitness4 < fitness_old4:
                fitness_old4 = fitness4
            if fitness5 < fitness_old5:
                fitness_old5 = fitness5
            fitness = [fitness1,
                       fitness2,
                       fitness3,
                       fitness4,
                       fitness5]

        return x if pas is True else fitness

    ####################################    P R I M I T I V E  -  S E T     ################################################
    #try:
    #del pset
    pset = gp.PrimitiveSet("MAIN", 5)
    if param[8] == 1:
        pset.addPrimitive(operator.add, 2, name="Add")
    if param[9] == 1:
        pset.addPrimitive(operator.sub, 2, name="Sub")
    if param[10] == 1:
        pset.addPrimitive(operator.mul, 2, name="Mul")
    if param[11] == 1:
        pass#pset.addPrimitive(Div, 2)
    if param[12] == 1:
        pass#pset.addPrimitive(operator.pow, 2, name="Pow")
    if param[13] == 1:
        pset.addPrimitive(Abs, 1)
    if param[14] == 1:
        pass#pset.addPrimitive(Sqrt, 1)
    if param[15] == 1:
        pset.addPrimitive(Log, 1)
    if param[16] == 1:
        pass#pset.addPrimitive(Exp, 1)
    if param[17] == 1:
        pset.addPrimitive(Sin, 1)
    if param[18] == 1:
        pset.addPrimitive(Cos, 1)
    if param[19] == 1:
        pset.addTerminal(np.pi, "pi")
    if param[20] == 1:
        pset.addTerminal(np.e, name="nap")  # e Napier constant number
    '''if param[21] == 1:
        pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-5, 5), 4))
    if param[21] == 1:
        pset.addEphemeralConstant("rand102", lambda: round(random.uniform(-10, 10), 4))
    if param[22] == 1:
        pset.addEphemeralConstant("rand103", lambda: round(random.uniform(-15, 15), 4))
    if param[23] == 1:
        pset.addEphemeralConstant("rand104", lambda: round(random.uniform(-20, 20), 4))'''

    pset.renameArguments(ARG0='errR')
    pset.renameArguments(ARG1='errTheta')
    pset.renameArguments(ARG2='errVr')
    pset.renameArguments(ARG3='errVt')
    pset.renameArguments(ARG4='errm')

    ################################################## TOOLBOX #############################################################

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))  # MINIMIZATION OF THE FITNESS FUNCTION

    creator.create("Individual", list, fitness=creator.Fitness, height=1)

    creator.create("SubIndividual", gp.PrimitiveTree, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    # toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)   #### OLD ####
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, type_=pset.ret, min_=1, max_=3)  ### NEW ###

    toolbox.register("leg", tools.initIterate, creator.SubIndividual, toolbox.expr)  ### NEW ###
    toolbox.register("legs", tools.initRepeat, list, toolbox.leg, n=Ncontrols)  ### NEW ###

    # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)  ### NEW ###

    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###

    # toolbox.register("lambdify", gp.compile, pset=pset) ### NEW ###
    # toolbox.register("stringify", gp.compile, pset=pset) ### NEW ###

    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evaluate)  ### OLD ###
    # toolbox.register('evaluate', evaluate, toolbox=toolbox, sourceData=data, minTrades=minTrades, log=False) ###NEW ###

    # toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1, fitness_first=True) ### OLD ###
    toolbox.register("select", xselDoubleTournament, fitness_size=param[6], parsimony_size=param[7], fitness_first=True)  ### NEW ###

    toolbox.register("mate", xmate)  ### NEW ###
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)  ### NEW ###
    toolbox.register("mutate", xmut, expr=toolbox.expr_mut, strp=param[5])  ### NEW ###

    # toolbox.register("mate", gp.cxOnePointLeafBiased,termpb=0.1) ### OLD ###
    # toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) ### OLD ###

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    # toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    # toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

    history = tools.History()
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)
    #except(RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, IndexError):
      #return 1e10

    ########################################################################################################################
    if __name__ == "__main__":
      obj = Rocket()
      pop, log, hof, best_fit = main(size_gen, size_pop, Mu, Lambda, param)
      print(best_fit)
      #print(vars())
      #sys.modules[__name__].__dict__.clear()
      return best_fit

if 'pset' in locals() or 'pset' in globals():
      del pset, toolbox
