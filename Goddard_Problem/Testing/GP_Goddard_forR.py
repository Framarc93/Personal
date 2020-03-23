'''Script used for hyperparameter optimization in R'''

'''param = [size_pop, num_gen, lambda_coeff, cxpb, mutpb, strp, nEph, TriAdd, Add, sub, mul, div, pow, abs, sqrt, log, exp, sin, cos, pi, e]
   param_init = [100, 10,      1.3,          0.6,  0.25,  0.6,  10,   1,      1,   1,   1,   0,   0,   1,   0,    0,   0,   0,   0,   0,  0]
   param_max =  [200, 30,      1.6,          0.8,  0.5,   0.7,  20,  1,      1,   1,   1,   1,   1,   1,   1,    1,   1,   1,   1,   1, ,1]
   param_min =  [100, 10,      1,            0.5,  0.05,  0.1,  0,    0,      0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,  0]'''
import sys
from scipy.integrate import solve_ivp
import numpy as np
import operator
import random
from deap import gp, algorithms, base, creator, tools
from deap.algorithms import varOr
import timeit
import multiprocessing
from scipy.interpolate import PchipInterpolator
from functools import partial
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq

def GP_param_tuning(param):
    class HallOfFame(object):
        """The hall of fame contains the best individual that ever lived in the
        population during the evolution. It is lexicographically sorted at all
        time so that the first element of the hall of fame is the individual that
        has the best first fitness value ever seen, according to the weights
        provided to the fitness at creation time.
        The insertion is made so that old individuals have priority on new
        individuals. A single copy of each individual is kept at all time, the
        equivalence between two individuals is made by the operator passed to the
        *similar* argument.
        :param maxsize: The maximum number of individual to keep in the hall of
                        fame.
        :param similar: An equivalence operator between two individuals, optional.
                        It defaults to operator :func:`operator.eq`.
        The class :class:`HallOfFame` provides an interface similar to a list
        (without being one completely). It is possible to retrieve its length, to
        iterate on it forward and backward and to get an item or a slice from it.
        """

        def __init__(self, maxsize, similar=eq):
            self.maxsize = maxsize
            self.keys = list()
            self.items = list()
            self.similar = similar

        def update(self, population):
            """Update the hall of fame with the *population* by replacing the
            worst individuals in it by the best individuals present in
            *population* (if they are better). The size of the hall of fame is
            kept constant.

            :param population: A list of individual with a fitness attribute to
                               update the hall of fame with.
            """
            for ind in population:
                if len(self) == 0 and self.maxsize != 0:
                    # Working on an empty hall of fame is problematic for the
                    # "for else"
                    self.insert(population[0])
                    continue
                if (np.array(ind.fitness.wvalues) > np.array(self[-1].fitness.wvalues)).all() or len(
                        self) < self.maxsize:
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(-1)
                        self.insert(ind)

        def insert(self, item):
            """Insert a new individual in the hall of fame using the
            :func:`~bisect.bisect_right` function. The inserted individual is
            inserted on the right side of an equal individual. Inserting a new
            individual in the hall of fame also preserve the hall of fame's order.
            This method **does not** check for the size of the hall of fame, in a
            way that inserting a new individual in a full hall of fame will not
            remove the worst individual to maintain a constant size.
            :param item: The individual with a fitness attribute to insert in the
                         hall of fame.
            """
            item = deepcopy(item)
            i = bisect_right(self.keys, item.fitness)
            self.items.insert(len(self) - i, item)
            self.keys.insert(i, item.fitness)

        def remove(self, index):
            """Remove the specified *index* from the hall of fame.
            :param index: An integer giving which item to remove.
            """
            del self.keys[len(self) - (index % len(self) + 1)]
            del self.items[index]

        def clear(self):
            """Clear the hall of fame."""
            del self.items[:]
            del self.keys[:]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

        def __iter__(self):
            return iter(self.items)

        def __reversed__(self):
            return reversed(self.items)

        def __str__(self):
            return str(self.items)

    def eaMuPlusLambdaTol(population, toolbox, mu, lambda_, ngen, cxpb, mutpb, tol, stats=None, halloffame=None, verbose=__debug__):
        global cxpb_orig, mutpb_orig
        """This is the :math:`(\mu + \lambda)` evolutionary algorithm.

        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param mu: The number of individuals to select for the next generation.
        :param lambda\_: The number of children to produce at each generation.
        :param cxpb: The probability that an offspring is produced by crossover.
        :param mutpb: The probability that an offspring is produced by mutation.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                      inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                           contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                  evolution.

        The algorithm takes in a population and evolves it in place using the
        :func:`varOr` function. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evalutions for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *cxpb* and *mutpb* arguments are passed to the
        :func:`varOr` function. The pseudocode goes as follow ::

            evaluate(population)
            for g in range(ngen):
                offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
                evaluate(offspring)
                population = select(population + offspring, mu)

        First, the individuals having an invalid fitness are evaluated. Second,
        the evolutionary loop begins by producing *lambda_* offspring from the
        population, the offspring are generated by the :func:`varOr` function. The
        offspring are then evaluated and the next generation population is
        selected from both the offspring **and** the population. Finally, when
        *ngen* generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.

        This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox. This algorithm uses the :func:`varOr`
        variation.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        min_fit = np.array(logbook.chapters["fitness"].select("min"))
        # Begin the generational process
        flag_change = False
        flag_limit = False
        gen = 1
        while gen < ngen + 1 and not (min_fit[-1] <= tol).all():
            # Vary the population
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu)

            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            min_fit = np.array(logbook.chapters["fitness"].select("min"))
            min_actual = np.array(logbook.chapters["fitness"].select("min"))[-1]
            min_old = np.array(logbook.chapters["fitness"].select("min"))[-2]
            if verbose:
                print(logbook.stream)
            if (abs(min_actual - min_old) < 0.01).all() and flag_limit is False:
                cxpb = cxpb - 0.01
                mutpb = mutpb + 0.01
                print("change")
                flag_change = True
                if cxpb < 0.4:
                    cxpb = 0.4
                    mutpb = 0.5
                    print("limits")
                    flag_limit = True

            else:
                cxpb = cxpb_orig
                mutpb = mutpb_orig
                print("back to orig")
                if flag_change is True:
                    cxpb = cxpb_orig + 0.1
                    mutpb = mutpb_orig - 0.15
                    flag_change = False
                    print("orig after change")
                flag_limit = False

            gen += 1

        return population, logbook

    def Minn(ind):
        # global min_fit_log
        min_fit_log = ind[0]
        for i in range(len(ind)):
            if (np.array(ind[i]) < np.array(min_fit_log)).all():
                min_fit_log = ind[i]
        return min_fit_log

    def TriAdd(x, y, z):
        return x + y + z

    def Eph():
        return round(random.uniform(-1000, 1000), 6)

    def Abs(x):
        return abs(x)

    def Div(left, right):
        global flag
        try:
            x = left / right
            return x
        except (
        RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
        FloatingPointError, OverflowError):
            flag = True
            return 0.0

    def Mul(left, right):
        global flag
        try:
            # np.seterr(invalid='raise')
            return left * right
        except (
        RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
        FloatingPointError, OverflowError):
            flag = True
            return left

    def Sqrt(x):
        global flag
        try:
            if x > 0:
                return np.sqrt(x)
            else:
                return abs(x)
        except (
        RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            flag = True
            return 0

    def Log(x):
        global flag
        try:
            if x > 0:
                return np.log(x)
            else:
                return abs(x)
        except (
        RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            flag = True
            return 0

    def Exp(x):
        try:
            return np.exp(x)
        except (
        RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            return 0

    def Sin(x):
        global flag
        try:
            return np.sin(x)
        except (
        RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            flag = True
            return 0

    def Cos(x):
        global flag
        try:
            return np.cos(x)
        except (
        RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
            flag = True
            return 0

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
            indx = gp.mutEphemeral(ind[i1], "one")
            ind[i1] = indx[0]
            return ind,

    # Direct copy from tools - modified for individuals with GP trees in an array
    def xselDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first):
        assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

        def _sizeTournament(individuals, k, select):
            chosen = []
            for i in range(k):
                # Select two individuals from the population
                # The first individual has to be the shortest
                prob = parsimony_size / 2.
                ind1, ind2 = select(individuals, k=2)

                lind1 = sum([len(gpt) for gpt in ind1])
                lind2 = sum([len(gpt) for gpt in ind2])
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

    size_pop = 100 # int(param["pop"])  # Pop size
    
    size_gen = 15  # Gen size
    Mu = int(size_pop)
    Lambda = int(size_pop * param["lambda"])
    cxpb = param["cxpb"]
    mutpb = (1 - param["cxpb"] - 0.1)
    mutpb_orig = mutpb
    cxpb_orig = cxpb
    limit_height = 20  # Max height (complexity) of the controller law
    limit_size = 400  # Max size (complexity) of the controller law
    nEph = param["nEph"]

    ii = 0

    ################################# M A I N ###############################################

    def main(size_gen, size_pop, Mu, Lambda, mutpb, cxpb, mutpb_orig, cxpb_orig):

        global Rfun, Thetafun, Vrfun, Vtfun, mfun
        global tfin, ii

        best_fit = sum([1e3, 1e3, 1e3, 1e3, 1e3])
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
        hof = HallOfFame(100)  ### NEW ###
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        # stats_size = tools.Statistics(len)
        # stats_height = tools.Statistics(operator.attrgetter("height"))
        mstats = tools.MultiStatistics(fitness=stats_fit)

        mstats.register("avg", np.mean, axis=0)
        mstats.register("min", Minn)
        mstats.register("max", np.max, axis=0)
        pset = 1

        ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

        # pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, 0.6, 0.2, size_gen, stats=mstats, halloffame=hof,
                                             # verbose=True)  ### OLD ###
        try:
            pop, log = eaMuPlusLambdaTol(pop, toolbox, Mu, Lambda, size_gen, cxpb, mutpb, 10, stats=mstats, halloffame=hof, verbose=True)  ### NEW ###
            fit = np.zeros((size_gen))
            for i in range(size_gen):
                fit[i] = sum(np.array(log.chapters["fitness"].select("min")[i]))
            best_fit = min(fit)
            print(best_fit)
            return pop, log, hof, best_fit
        except (RuntimeWarning, RuntimeError, OverflowError, TypeError, ZeroDivisionError):
            return 1, 1, 1, best_fit
    ####################################################################################################################

    ##################################  F I T N E S S    F U N C T I O N    ################################################

    def evaluate(individual):
        global flag
        global pas
        global fitnnesoldvalue, fitness_old1, fitness_old2, fitness_old3, fitness_old4, fitness_old5
        global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun
        global tfin

        penalty = np.zeros((17))

        flag = False
        pas = False

        # Transform the tree expression in a callable function

        fTr = toolbox.compile(expr=individual[0])
        fTt = toolbox.compile(expr=individual[1])
        x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

        def sys(t, x):
            global flag

            # State Variables

            R = x[0]
            theta = x[1]
            Vr = x[2]
            Vt = x[3]
            m = x[4]

            if R < obj.Re or np.isnan(R):
                penalty[0] = penalty[0] + abs(R - obj.Re) / obj.Htarget
                R = obj.Re
                flag = True
            if R > obj.Rtarget or np.isinf(R):
                penalty[1] = penalty[1] + abs(R - obj.Rtarget) / obj.Htarget
                R = obj.Rtarget
                flag = True
            if m < obj.M0 - obj.Mp or np.isnan(m):
                penalty[2] = penalty[2] + abs(m - (obj.M0 - obj.Mp)) / obj.M0
                m = obj.M0 - obj.Mp
                flag = True
            elif m > obj.M0 or np.isinf(m):
                penalty[3] = penalty[3] + abs(m - obj.M0) / obj.M0
                m = obj.M0
                flag = True
            if abs(Vr) > 1e3 or np.isinf(Vr):
                penalty[4] = penalty[4] + abs(Vr - 1e3) / 1e3
                if Vr > 0:
                    Vr = 1e3
                else:
                    Vr = -1e3
                flag = True
            if abs(Vt) > 1e4 or np.isinf(Vt):
                penalty[5] = penalty[5] + abs(Vt - 1e4) / 1e4
                if Vt > 0:
                    Vt = 1e4
                else:
                    Vt = -1e4
                flag = True
            if theta < 0 or np.isnan(theta):
                penalty[6] = penalty[6] + abs(np.rad2deg(theta))
                theta = 0
                flag = True
            elif np.rad2deg(theta) > 60 or np.isinf(theta):
                penalty[7] = penalty[7] + abs(np.rad2deg(theta) - 60)
                theta = np.deg2rad(60)
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

            if Tr < 0.0 or np.isnan(Tr):
                penalty[10] = 10000  # penalty[10] + abs(Tr)/obj.Tmax
                Tr = 0.0
                flag = True

            elif Tr > obj.Tmax or np.isinf(Tr):
                penalty[11] = 10000  # penalty[7] + abs(Tr - obj.Tmax)/obj.Tmax
                Tr = obj.Tmax
                flag = True

            if Tt < 0.0 or np.isnan(Tt):
                penalty[12] = 10000  # penalty[12] + abs(Tt)/obj.Tmax
                Tt = 0.0
                flag = True

            elif Tt > obj.Tmax or np.isinf(Tt):
                penalty[13] = 10000  # penalty[9] + abs(Tt - obj.Tmax)/obj.Tmax
                Tt = obj.Tmax
                flag = True

            dxdt[0] = Vr
            dxdt[1] = Vt / R
            dxdt[2] = Tr / m - Dr / m - g + Vt ** 2 / R
            dxdt[3] = Tt / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(Tr ** 2 + Tt ** 2) / g0 / Isp

            return dxdt

        sol = solve_ivp(sys, [0.0, tfin], x_ini)
        y1 = sol.y[0, :]
        y2 = sol.y[1, :]
        y3 = sol.y[2, :]
        y4 = sol.y[3, :]
        y5 = sol.y[4, :]
        tt = sol.t
        if sol.t[-1] < tfin:
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
        err3 = (vr - y3) / 1e3
        err4 = (vt - y4) / 1e4
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
        for a, b, c, d, e, n in zip(err1, err2, err3, err4, err5, step):
            IAE[0][j] = n * abs(a)
            IAE[1][j] = n * abs(b)
            IAE[2][j] = n * abs(c)
            IAE[3][j] = n * abs(d)
            IAE[4][j] = n * abs(e)
            j = j + 1

        fitness = [sum(IAE[0]), sum(IAE[1]), sum(IAE[2]), sum(IAE[3]), sum(IAE[4])]

        if flag is True:
            x = np.array([fitness[0] + (sum(penalty[0:2]) + sum(penalty[8:16])) / fitness[0],
                          fitness[1] + (sum(penalty[6:15]) + penalty[-1]) / fitness[1],
                          fitness[2] + (penalty[4] + sum(penalty[8:15])) / fitness[2],
                          fitness[3] + (penalty[5] + sum(penalty[8:15])) / fitness[3],
                          fitness[4] + (sum(penalty[2:4]) + sum(penalty[8:15])) / fitness[4]])
        fitness = [sum(IAE[0]), sum(IAE[1]), sum(IAE[2]), sum(IAE[3]), sum(IAE[4])]

        return x if flag is True else fitness


    pset = gp.PrimitiveSet("MAIN", 5)
    if param["TriAdd"] == 1:
        pset.addPrimitive(TriAdd, 3)
    if param["Add"] == 1:
        pset.addPrimitive(operator.add, 2, name="Add")
    if param["sub"] == 1:
        pset.addPrimitive(operator.sub, 2, name="Sub")
    if param["mul"] == 1:
        pset.addPrimitive(operator.mul, 2, name="Mul")
    if param["div"] == 1:
        pset.addPrimitive(Div, 2)
    if param["pow"] == 1:
        pset.addPrimitive(operator.pow, 2, name="Pow")
    if param["abs"] == 1:
        pset.addPrimitive(Abs, 1)
    if param["sqrt"] == 1:
        pset.addPrimitive(Sqrt, 1)
    if param["log"] == 1:
        pset.addPrimitive(Log, 1)
    if param["exp"] == 1:
        pset.addPrimitive(Exp, 1)
    if param["sin"] == 1:
        pset.addPrimitive(Sin, 1)
    if param["cos"] == 1:
        pset.addPrimitive(Cos, 1)
    if param["pi"] == 1:
        pset.addTerminal(np.pi, "pi")
    if param["e"] == 1:
        pset.addTerminal(np.e, name="nap")  # e Napier constant number

    for i in range(nEph):
        pset.addEphemeralConstant("rand{}{}".format(i, sum(param.values())), Eph)

    pset.renameArguments(ARG0='errR')
    pset.renameArguments(ARG1='errTheta')
    pset.renameArguments(ARG2='errVr')
    pset.renameArguments(ARG3='errVt')
    pset.renameArguments(ARG4='errm')

    ################################################## TOOLBOX #############################################################

    creator.create("Fitness", base.Fitness, weights=(-param['fit1'], -param['fit2'], -param['fit3'], -param['fit4'], -param['fit5']))  # MINIMIZATION OF THE FITNESS FUNCTION

    creator.create("Individual", list, fitness=creator.Fitness, height=1)

    creator.create("SubIndividual", gp.PrimitiveTree, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    # toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)   #### OLD ####
    toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=2, max_=5)  ### NEW ###

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
    toolbox.register("select", tools.selNSGA2)  ### NEW ###

    toolbox.register("mate", xmate)  ### NEW ###
    toolbox.register("expr_mut", gp.genFull, min_=2, max_=5)  ### NEW ###
    toolbox.register("mutate", xmut, expr=toolbox.expr_mut, strp=param["strp"])  ### NEW ###

    # toolbox.register("mate", gp.cxOnePointLeafBiased,termpb=0.1) ### OLD ###
    # toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) ### OLD ###

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    # toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

    history = tools.History()
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    #if __name__ == "__main__":
    obj = Rocket()
    print(param)
    pop, log, hof, best_fit = main(size_gen, size_pop, Mu, Lambda, mutpb, cxpb, mutpb_orig, cxpb_orig)

    return best_fit


