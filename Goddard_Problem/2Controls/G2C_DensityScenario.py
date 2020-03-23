'''GP_Goddard_2Controls: intelligent control of goddard rocket with 2 controls

This script simulates the Intelligent Control of a Goddard Rocket with 2 controls where the optimal trajectory is defined
with a simplified density model and during the flight a more complex density model is used. When the tracking error becomes
greater then a certain threshold, the new Gp control law is evaluated

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk

References:
relevant references for the algorithm
[1] Entropy-Driven Adaptive Representation. J. P. Rosca. Proceedings of the Workshop on Genetic Programming: From Theory to Real-World Applications, 23-32. 1995
[2] Exploration and Exploitation in Evolutionary Algorithms: a Survey. M. Crepinsek, S.Liu, M. Mernik. ACM Computer Surveys. 2013
[3] Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms: A Survey of the State of the Art. C. A. Coello Coello. Computer Methods in Applied Mechanics and Engineering 191. 2002'''

from scipy.integrate import solve_ivp, simps
import numpy as np
import operator
import random
from deap import gp, base, creator, tools
import matplotlib.pyplot as plt
import multiprocessing
from scipy.interpolate import PchipInterpolator
from time import time, strftime
from copy import deepcopy
from functools import partial, wraps
import GP_PrimitiveSet as gpprim
from operator import eq, mul, truediv
from collections import Sequence
import sys
import os


######################### SAVE DATA #############################################
timestr = strftime("%Y%m%d-%H%M%S")
flag_save = True
if flag_save:
    os.makedirs(
        "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Results/40s_evalTime_2CDensityChange_NoLearning_1000it_Res_{}_{}".format(
            os.path.basename(__file__), timestr))
    savefig_file = "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Results/40s_evalTime_2CDensityChange_NoLearning_1000it_Res_{}_{}/Plot_".format(
        os.path.basename(__file__), timestr)
    savedata_file = "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Results/40s_evalTime_2CDensityChange_NoLearning_1000it_Res_{}_{}/".format(
        os.path.basename(__file__), timestr)


############## MODIFIED BLOAT CONTROL #########################################
def staticLimit(key, max_value):
    """Implement a static limit on some measurement on a GP tree, as defined
    by Koza in [Koza1989]. It may be used to decorate both crossover and
    mutation operators. When an invalid (over the limit) child is generated,
    it is simply replaced by one of its parents, randomly selected.

    This operator can be used to avoid memory errors occuring when the tree
    gets higher than 90 levels (as Python puts a limit on the call stack
    depth), because it can ensure that no tree higher than this limit will ever
    be accepted in the population, except if it was generated at initialization
    time.

    :param key: The function to use in order the get the wanted value. For
                instance, on a GP tree, ``operator.attrgetter('height')`` may
                be used to set a depth limit, and ``len`` to set a size limit.
    :param max_value: The maximum value allowed for the given measurement.
    :returns: A decorator that can be applied to a GP operator using \
    :func:`~deap.base.Toolbox.decorate`

    .. note::
       If you want to reproduce the exact behavior intended by Koza, set
       *key* to ``operator.attrgetter('height')`` and *max_value* to 17.

    .. [Koza1989] J.R. Koza, Genetic Programming - On the Programming of
        Computers by Means of Natural Selection (MIT Press,
        Cambridge, MA, 1992)

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                check = max(ind[0], ind[1], key=key)
                if key(ind[0]) == key(check):
                    j = 0
                else:
                    j = 1
                if key(check) > max_value:
                    new_inds[i][j] = random.choice(keep_inds)[j]
            return new_inds

        return wrapper

    return decorator


####################  MODIFIED SELECTION ALGORITHM ##########################

def selDoubleTournament(individuals, k, fitness_size, parsimony_size, enough, fitness_first):
    '''Modified from DEAP library'''
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, enough, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            if len(individuals) == 1:
                return random.choice(individuals)
            else:
                prob = parsimony_size / 2.
                ind1, ind2 = select(individuals, enough, k=2)
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

            return chosen[0]

    def _fitTournament(individuals, enough, k):
        chosen = []
        for _ in range(k):
            a1, a2 = random.sample(individuals, 2)
            if enough is False:
                if a1.fitness.values[-1] == 0 and a2.fitness.values[-1] == 0:
                    if sum(a1.fitness.wvalues) > sum(a2.fitness.wvalues):
                        chosen.append(a1)
                    else:
                        chosen.append(a2)
                elif a1.fitness.values[-1] == 0 and a2.fitness.values[-1] != 0:
                    chosen.append(a1)
                elif a1.fitness.values[-1] != 0 and a2.fitness.values[-1] == 0:
                    chosen.append(a2)
                elif a1.fitness.values[-1] != 0 and a2.fitness.values[-1] != 0:
                    if a1.fitness.values[-1] < a2.fitness.values[-1]:
                        chosen.append(a1)
                    else:
                        chosen.append(a2)
            else:
                if sum(a1.fitness.wvalues) > sum(a2.fitness.wvalues):
                    chosen.append(a1)
                else:
                    chosen.append(a2)
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament)
        return _sizeTournament(individuals, enough, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, enough, k)


def selBest(individuals):
    best = individuals[0]
    choice = random.random()
    for ind in individuals:
        if ind.fitness.values[-1] == 0 and best.fitness.values[-1] == 0:
            if ind.fitness.values[0] < best.fitness.values[0]:
                best = ind
        elif ind.fitness.values[-1] == 0 and best.fitness.values[-1] != 0:
            best = ind
        elif ind.fitness.values[-1] != 0 and best.fitness.values[-1] != 0:
            if choice > 0.9:
                if ind.fitness.values[-1] < best.fitness.values[-1]:
                    best = ind
            else:
                if sum(ind.fitness.values) < sum(best.fitness.values):
                    best = ind
    return best


def InclusiveTournament(individuals, mu, to_mate):
    '''Rationale behind InclusiveTournament: a double tournament selection is performed in each category, so to maintain
     diversity. Double Tournamet is used so to avoid bloat. An exploited measure is used to point out when a category is
     completely exploited. For example, if in a category are present only 4 individuals, the tournament will be
     performed at maximum 4 times in that category. This to avoid a spreading of clones of the same individuals.'''

    organized_pop, good_indexes = subset_diversity(individuals)
    chosen = []
    exploited = np.zeros((len(good_indexes)))
    enough = False
    if to_mate is True:
        enough = True
    j = 0
    count = 0
    while len(chosen) < mu:
        if j > len(good_indexes) - 1:
            j = 0
        i = good_indexes[j]

        if exploited[j] < len(organized_pop["cat{}".format(i)]):
            selected = selDoubleTournament(organized_pop["cat{}".format(i)], 1, 2, 1.6, enough, fitness_first=True)
            chosen.append(selected)
            if selected.fitness.values[-1] == 0:
                count += 1
            exploited[j] += 1
        j += 1
        choice = random.random()
        if choice > 0.8:
            enough = True
        elif choice <= 0.8 and to_mate is False:
            enough = False
        if count >= 2 * mu / 3:
            enough = True

    if enough is True and to_mate is False:
        print("Greed prevention")
    elif enough is True and to_mate is True:
        print("Mating constraints on feasibles")
    # best = selBest(individuals) # always select also the best individual from the previous population
    # chosen.append(best)
    return chosen


################### POPULATION CLASS  ###############################

class POP(object):
    '''This class is used to collect data about a population. Used at the beginning for the selection of the initial
        population. Entropy measure comes from [1]'''

    def __init__(self, population):
        self.items = list()
        self.max = len(max(population[0], key=len)) + len(max(population[1], key=len))
        self.min = len(min(population[0], key=len)) + len(min(population[1], key=len))
        self.maxDiff = self.max - self.min
        self.categories, self.indexes = subset_diversity(population)
        self.entropy = -sum(
            self.categories["distribution"]["percentage"] * np.log(self.categories["distribution"]["percentage"]))
        for ind in population:
            item = deepcopy(ind)
            self.items.append(item)

    def output_stats(self, Title):
        print("\n")
        print("---------- {} STATISTICS ------------".format(Title))
        print("-- Min len: {}, Max len: {}, Max Diff: {} ---".format(self.min, self.max, self.maxDiff))
        print("-- Entropy: {0:.3f} -------------------------".format(self.entropy))
        print("-- Distribution: {} --------------------".format(self.categories["distribution"]["percentage"] * 100))
        print("---------------------------------------------")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        print("it is used")
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


######################## MODIFIED FITNESS CLASS ###########################


class FitnessMulti(object):
    '''Fitness class modified from DEAP library. Only modification is the sum inserted in the comparison functions'''

    weights = None
    """The weights are used in the fitness comparison. They are shared among
    all fitnesses of the same type. When subclassing :class:`Fitness`, the
    weights must be defined as a tuple where each element is associated to an
    objective. A negative weight element corresponds to the minimization of
    the associated objective and positive weight to the maximization.

    .. note::
        If weights is not defined during subclassing, the following error will
        occur at instantiation of a subclass fitness object:

        ``TypeError: Can't instantiate abstract <class Fitness[...]> with
        abstract attribute weights.``
    """

    wvalues = ()
    """Contains the weighted values of the fitness, the multiplication with the
    weights is made when the values are set via the property :attr:`values`.
    Multiplication is made on setting of the values for efficiency.

    Generally it is unnecessary to manipulate wvalues as it is an internal
    attribute of the fitness used in the comparison operators.
    """

    def __init__(self, values=()):
        if self.weights is None:
            raise TypeError("Can't instantiate abstract %r with abstract "
                            "attribute weights." % (self.__class__))

        if not isinstance(self.weights, Sequence):
            raise TypeError("Attribute weights of %r must be a sequence."
                            % self.__class__)

        if len(values) > 0:
            self.values = values

    def getValues(self):
        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError("Both weights and assigned values must be a "
                            "sequence of numbers when assigning to values of "
                            "%r. Currently assigning value(s) %r of %r to a "
                            "fitness with weights %s."
                            % (self.__class__, values, type(values),
                               self.weights)).with_traceback(traceback)

    def delValues(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValues,
                      ("Fitness values. Use directly ``individual.fitness.values = values`` "
                       "in order to set the fitness and ``del individual.fitness.values`` "
                       "in order to clear (invalidate) the fitness. The (unweighted) fitness "
                       "can be directly accessed via ``individual.fitness.values``."))

    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return sum(self.wvalues) <= sum(other.wvalues)

    def __lt__(self, other):
        return sum(self.wvalues) < sum(other.wvalues)

    def __eq__(self, other):
        return sum(self.wvalues == other.wvalues)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.

        It assumes that the elements in the :attr:`values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_.wvalues = self.wvalues
        return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
                              self.values if self.valid else tuple())


############################# MODIFIED HALL OF FAME ###################################################################

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

    def shuffle(self):
        arr_start = deepcopy(self.items)
        arr_end = []
        while len(arr_start) > 0:
            ind = random.randint(0, len(arr_start) - 1)
            arr_end.append(arr_start[ind])
            arr_start.pop(ind)
        return arr_end

    def update(self, population, for_feasible):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        if len(self) == 0 and self.maxsize != 0 and len(population) > 0:
            # Working on an empty hall of fame is problematic for the
            # "for else"
            self.insert(population[0], for_feasible)

        if for_feasible is True:
            for ind in population:
                if ind.fitness.values[-1] == 0.0:  ### NEW PART - REMOVE IF DOESNT WORK ####
                    if self[0].fitness.values[-1] == 0.0:
                        if sum(ind.fitness.values) < sum(self[0].fitness.values) or len(self) < self.maxsize:
                            for hofer in self:
                                # Loop through the hall of fame to check for any
                                # similar individual
                                if self.similar(ind, hofer):
                                    break
                            else:
                                # The individual is unique and strictly better than
                                # the worst
                                if len(self) >= self.maxsize:
                                    self.remove(0)
                                self.insert(ind, for_feasible)
                    else:
                        for hofer in self:
                            # Loop through the hall of fame to check for any
                            # similar individual
                            if self.similar(ind, hofer):
                                break
                        else:
                            # The individual is unique and strictly better than
                            # the worst
                            if len(self) >= self.maxsize:
                                self.remove(0)
                            self.insert(ind, for_feasible)  #### END NEW PART ######
                elif (sum(ind.fitness.values) < sum(self[0].fitness.values)) or len(self) < self.maxsize:
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(0)
                        self.insert(ind, for_feasible)
        else:
            for ind in population:
                if ind.fitness.values[0] < 10 and ind.fitness.values[1] < 40 and sum(ind.fitness.values) < sum(
                        self[0].fitness.values) or len(self) < self.maxsize:
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(0)
                        self.insert(ind, for_feasible)

    def insert(self, item, for_feasible):
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

        def bisect_right(a, x, lo=0, hi=None):
            """Return the index where to insert item x in list a, assuming a is sorted.
            The return value i is such that all e in a[:i] have e <= x, and all e in
            a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
            insert just after the rightmost x already there.
            Optional args lo (default 0) and hi (default len(a)) bound the
            slice of a to be searched.
            """

            if lo < 0:
                raise ValueError('lo must be non-negative')
            if hi is None:
                hi = len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                '''must indentify 4 cases: if both are feasible, if the new is feasible and the one in the list is not, viceversa and if both are infeasible'''
                if for_feasible is True:
                    # 1st case: both are feasible
                    if x.values[-1] == 0 and a[mid].values[-1] == 0:
                        if sum(x.values) < sum(a[mid].values):
                            hi = mid
                        else:
                            lo = mid + 1
                    # 2nd case: value to insert is feasible, the one in the list is not
                    elif x.values[-1] == 0 and a[mid].values[-1] != 0:
                        hi = mid
                    # 3rd case: value to insert is not feasible, the one in the list is feasible
                    elif x.values[-1] != 0 and a[mid].values[-1] == 0:
                        lo = mid + 1
                    # 4th case: both are infeasible
                    elif x.values[-1] != 0 and a[mid].values[-1] != 0:
                        if x.values[-1] < a[mid].values[-1]:
                            hi = mid
                        else:
                            lo = mid + 1
                else:
                    if sum(x.values) < sum(a[mid].values):
                        hi = mid
                    else:
                        lo = mid + 1
            return lo

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


##############################  MODIFIED ALGORITHMS  ##################################################################


def subset_feasible(population):
    sub_pop = []
    for ind in population:
        if ind.fitness.values[-1] == 0:
            sub_pop.append(ind)
    return sub_pop


def subset_unfesible(population):
    sub_pop = []
    for ind in population:
        if ind.fitness.values[0] < 10 and ind.fitness.values[1] < 30 and ind.fitness.values[-1] < 10 and ind.fitness.values[-1] != 0:
            sub_pop.append(ind)
    return sub_pop


def subset_diversity(population):
    cat_number = 10  # here the number of categories is selected
    lens = []
    categories = {}
    distribution = []
    distr_stats = {}
    invalid_ind = []

    for ind in population:
        lens.append((len(ind[0]) + len(ind[1])))
    cat = np.linspace(min(lens), max(lens), cat_number + 1)
    useful_ind = np.linspace(0, len(cat) - 2, len(cat) - 1)

    for i in range(len(cat) - 1):
        categories["cat{}".format(i)] = []
    for ind in population:
        for i in range(len(cat) - 1):
            if (len(ind[0]) + len(ind[1])) >= cat[i] and (len(ind[0]) + len(ind[1])) <= cat[i + 1]:
                categories["cat{}".format(i)].append(ind)

    for i in range(len(cat) - 1):
        if categories["cat{}".format(i)] == []:
            invalid_ind.append(i)
        distribution.append(len(categories["cat{}".format(i)]))
    distr_stats["individuals"] = distribution
    distr_stats["percentage"] = np.array(distribution) / len(population)
    categories["distribution"] = distr_stats
    if invalid_ind != []:
        useful_ind = np.delete(useful_ind, invalid_ind, 0)
    return categories, np.asarray(useful_ind, dtype=int)


def varOr(population, toolbox, lambda_, box, sub_div, good_indexes_original):
    '''Modified from DEAP library. Modifications:
    - '''
    global mutpb, cxpb
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    sub_pop = subset_feasible(population)

    len_pop = len(box)
    len_subpop = len(sub_pop)

    # worst = box[:int(len_pop/2)]  # best individuals are in the first third of the container
    #best = box[int(len_pop / 2):]  # worst individuals are in the last 2/3 of the container
    if sub_pop == []:
        # mutpb = 0.7
        # cxpb = 0.15
        print("Exploring for feasible individuals. Mutpb: {}, Cxpb:{}".format(mutpb, cxpb))
    else:
        if cxpb < 0.65:
            mutpb = mutpb - 0.01
            cxpb = cxpb + 0.01
        print("\n")
        print("{}/{} ({}%) FEASIBLE INDIVIDUALS".format(len_subpop, len(population),
                                                        round(len(sub_pop) / len(population) * 100, 2)))
        print("Mutpb: {}, Cxpb:{}".format(mutpb, cxpb))

    #exploited = np.zeros((len(good_indexes_original)))
    good_indexes = list(good_indexes_original)
    good_list = list(good_indexes)
    for _ in range(lambda_):
        op_choice = random.random()
        cat = np.zeros((2))  ### selection of 2 different categories for crossover
        for i in range(2):
            if good_list == []:
                good_list = list(good_indexes)
            used = random.choice(good_list)
            cat[i] = used
            good_list.remove(used)
        if op_choice < cxpb:  # Apply crossover

            ind1, ind2 = map(toolbox.clone, [selBest(sub_div["cat{}".format(int(cat[0]))]), random.choice(
                sub_div["cat{}".format(int(cat[1]))])])  # select one individual from the best and one from the worst
            while sum(ind1.fitness.values) == sum(ind2.fitness.values):
                cat = np.zeros((2))  ### selection of 2 different categories for crossover
                for i in range(2):
                    if good_list == []:
                        good_list = list(good_indexes)
                    used = random.choice(good_list)
                    cat[i] = used
                    good_list.remove(used)
                ind1, ind2 = map(toolbox.clone, [selBest(sub_div["cat{}".format(int(cat[0]))]), random.choice(sub_div["cat{}".format(int(cat[1]))])])  # select one individual from the best and one from the worst

            ind1, ind2 = toolbox.mate(ind1, ind2)
            if sum(ind1.fitness.wvalues) > sum(ind2.fitness.wvalues):
                del ind1.fitness.values
                offspring.append(ind1)
            else:
                del ind2.fitness.values
                offspring.append(ind2)
        elif op_choice < cxpb + mutpb:  # Apply mutation

            ind = toolbox.clone(selBest(population))  # maybe better to use a sort of roulette
            old = ind
            ind, = toolbox.mutate(ind)
            if sum(old.fitness.values) <= sum(ind.fitness.values) and len(ind) > limit_size:
                ind = old
            del ind.fitness.values
            offspring.append(ind)
        else:  # Apply reproduction
            if len_subpop >= 1:
                offspring.append(random.choice(sub_pop))
            else:
                offspring.append(selBest(population))  # reproduce only from the best

    return offspring, len_subpop


def eaMuPlusLambdaTol(population, toolbox, mu, lambda_, ngen, tol, stats=None, halloffame=None, verbose=__debug__):
    '''Modified from DEAP library. Modifications:
    '''
    global mutpb, cxpb
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population, for_feasible=True)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    min_fit = np.array(logbook.chapters["fitness"].select("min"))
    mh = 0
    min_fit_history = np.zeros((30))
    min_fit_history[mh] = sum(min_fit[0])
    # Begin the generational process
    gen = 1
    # hof_unfeasible = HallOfFame(100)
    its_time = False
    while gen < ngen + 1 and (min_fit[-1][0] > tol or min_fit[-1][-1] > 0.0):
        # Vary the population
        # filled = False  # used to check if the hall of fame of unfeasible individuals is not empy
        #sub_unfeas = subset_unfesible(population)
        # if sub_unfeas != []:
        #   hof_unfeasible.update(sub_unfeas, for_feasible=False)
        #  filled = True
        if (abs(min_fit_history[0] - min_fit_history[-1] < 1e-6) or gen % 200 == 0) and len_feas > 10:
            print("\n")
            print("----------------------------- RESEEDING -------------------------------")
            print("\n")
            mutpb = 0.7
            cxpb = 0.2
            chosen = toolbox.select(population, int(len(population) / 3), to_mate=False)

            old_entropy = 0
            for i in range(200):
                pop = POP(toolbox.population(n=len(population) - len(chosen)))
                if pop.entropy > old_entropy:  # and len(pop.indexes) == len(pop.categories) - 1:
                    best_pop = pop.items
                    old_entropy = pop.entropy

            invalid_ind = [ind for ind in best_pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population = chosen + best_pop
            # hof_unfeasible.update(population, for_feasible=False)
            min_fit_history = np.zeros((30))
            mh = 0
        # sub_bestUnfeasible = subset_unfesible(population)
        # hof_unfeasible.update(sub_bestUnfeasible)
        # population[:] = toolbox.select(population, int(mu/2))
        #if its_time is True:
        #    to_mate = InclusiveTournament(population, int(2 * len(population) / 3), to_mate=True)
        #else:
        #    to_mate = InclusiveTournament(population, int(2 * len(population) / 3), to_mate=False)
        sub_div, good_index = subset_diversity(population)
        offspring, len_feas = varOr(population, toolbox, mu, halloffame, sub_div, good_index)
        #if len_feas > 200:
        #    its_time = True
        #else:
        #    its_time = False
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring, for_feasible=True)
        # if filled is True:
        #   hof_unfeasible.update(offspring, for_feasible=False)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu, to_mate=False)
        # stats_pop = POP(population)
        # stats_pop.output_stats("FINAL POPULATION")
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        min_fit = np.array(logbook.chapters["fitness"].select("min"))
        if mh == len(min_fit_history):
            mh = 0
            min_fit_history = np.zeros((30))
        min_fit_history[mh] = sum(min_fit[-1])
        mh += 1
        if verbose:
            print(logbook.stream)
        gen += 1

    return population, logbook


########################### GENETIC OPERATORS FOR MULTIPLE TREE OUTPUT   #####################################


def xmate(ind1, ind2):
    ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2


def xmut(ind, expr, unipb, shrpb, inspb):
    choice = random.random()

    if choice < unipb:
        indx1 = gp.mutUniform(ind[0], expr, pset=psetR)
        ind[0] = indx1[0]
        indx2 = gp.mutUniform(ind[1], expr, pset=psetT)
        ind[1] = indx2[0]
    elif unipb <= choice < unipb + shrpb:
        indx1 = gp.mutShrink(ind[0])
        ind[0] = indx1[0]
        indx2 = gp.mutShrink(ind[1])
        ind[1] = indx2[0]
    elif unipb + shrpb <= choice < unipb + shrpb + inspb:
        indx1 = gp.mutInsert(ind[0], pset=psetR)
        ind[0] = indx1[0]
        indx2 = gp.mutInsert(ind[1], pset=psetT)
        ind[1] = indx2[0]
    else:
        choice2 = random.random()
        if choice2 < 0.5:
            indx1 = gp.mutEphemeral(ind[0], "all")
            ind[0] = indx1[0]
            indx2 = gp.mutEphemeral(ind[1], "all")
            ind[1] = indx2[0]
        else:
            indx1 = gp.mutEphemeral(ind[0], "one")
            ind[0] = indx1[0]
            indx2 = gp.mutEphemeral(ind[1], "one")
            ind[1] = indx2[0]
    return ind,


############################ MODIFIED STATISTICS FUNCTIONS  #########################################################

def Min(pop):
    min = pop[0]
    w = np.array([-0.5, -1.0])
    for ind in pop:
        if ind[-1] == 0:
            if min[-1] == 0 and sum(ind[0:2] * w) > sum(min[0:2] * w):
                min = ind
            elif min[-1] != 0:
                min = ind
        elif ind[-1] < min[-1]:
            min = ind
    return min


def Max(inds):
    max = inds[0]
    for fit in inds:
        if fit[-1] == 0:
            if max[-1] == 0 and sum(fit[0:2]) > sum(max[0:2]):
                max = fit
        elif sum(fit) > sum(max):
            max = fit
    return max


a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]


def air_density(h):
    global flag
    beta = 1 / 8500.0  # scale factor [1/m]
    rho0 = 1.225  # kg/m3
    return rho0 * np.exp(-beta * h)


def isa(altitude, singl_val):
    t0 = 288.15
    p0 = 101325
    prevh = 0.0
    R = 287.00
    m0 = 28.9644
    Rs = 8314.32
    m0 = 28.9644
    g0 = 9.80665
    r = 6371.0 * 1000
    if singl_val == 1:
        altitude = np.array([altitude])
    temperature = np.zeros(len(altitude))
    pressure = np.zeros(len(altitude))
    tempm = np.zeros(len(altitude))
    density = np.zeros(len(altitude))
    csound = np.zeros(len(altitude))
    k = 0

    def cal(ps, ts, av, h0, h1):
        if av != 0:
            t1 = ts + av * (h1 - h0)
            p1 = ps * (t1 / ts) ** (-g0 / av / R)
        else:
            t1 = ts
            p1 = ps * np.exp(-g0 / R / ts * (h1 - h0))
        return t1, p1

    def atm90(a90v, z, hi, tc1, pc, tc2, tmc):
        for num in hi:
            if z <= num:
                ind = hi.index(num)
                if ind == 0:
                    zb = hi[0]
                    b = zb - tc1[0] / a90v[0]
                    t = tc1[0] + tc2[0] * (z - zb) / 1000
                    tm = tmc[0] + a90v[0] * (z - zb) / 1000
                    add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                    add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                    p = pc[0] * np.exp(-m0 / (a90v[0] * Rs) * g0 * r ** 2 * (add1 - add2))
                else:
                    zb = hi[ind - 1]
                    b = zb - tc1[ind - 1] / a90v[ind - 1]
                    t = tc1[ind - 1] + (tc2[ind - 1] * (z - zb)) / 1000
                    tm = tmc[ind - 1] + a90v[ind - 1] * (z - zb) / 1000
                    add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                    add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                    p = pc[ind - 1] * np.exp(-m0 / (a90v[ind - 1] * Rs) * g0 * r ** 2 * (add1 - add2))
                break
        return t, p, tm

    for alt in altitude:
        if alt < 0:
            t = t0
            p = p0
            d = p / (R * t)
            c = np.sqrt(1.4 * R * t)
            density[k] = d
            csound[k] = c
            # temperature[k] = t
            pressure[k] = p
            # tempm[k] = t
        elif 0 <= alt < 90000:

            for i in range(0, 8):

                if alt <= hv[i]:
                    t, p = cal(p0, t0, a[i], prevh, alt)
                    d = p / (R * t)
                    c = np.sqrt(1.4 * R * t)
                    density[k] = d
                    csound[k] = c
                    temperature[k] = t
                    pressure[k] = p
                    tempm[k] = t
                    t0 = 288.15
                    p0 = 101325
                    prevh = 0
                    break
                else:

                    t0, p0 = cal(p0, t0, a[i], prevh, hv[i])
                    prevh = hv[i]

        elif 90000 <= alt <= 190000:
            t, p, tpm = atm90(a90, alt, h90, tcoeff1, pcoeff, tcoeff2, tmcoeff)
            d = p / (R * tpm)
            c = np.sqrt(1.4 * R * tpm)
            density[k] = d
            csound[k] = c
            # temperature[k] = t
            pressure[k] = p
            # tempm[k] = t
        elif alt > 190000:
            zb = h90[6]
            z = h90[-1]
            b = zb - tcoeff1[6] / a90[6]
            t = tcoeff1[6] + (tcoeff2[6] * (z - zb)) / 1000
            tm = tmcoeff[6] + a90[6] * (z - zb) / 1000
            add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
            add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
            p = pcoeff[6] * np.exp(-m0 / (a90[6] * Rs) * g0 * r ** 2 * (add1 - add2))
            d = p / (R * t)
            c = np.sqrt(1.4 * R * tm)
            density[k] = d
            csound[k] = c
            # temperature[k] = t
            pressure[k] = p
            # tempm[k] = t
        k += 1
    return pressure, density, csound


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
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s


tref = np.load("time.npy")
total_time_simulation = tref[-1]
tfin = tref[-1]

Rref = np.load("R.npy")
Thetaref = np.load("Theta.npy")
Vrref = np.load("Vr.npy")
Vtref = np.load("Vt.npy")
mref = np.load("m.npy")
Ttref = np.load("Tt.npy")
Trref = np.load("Tr.npy")

Rfun = PchipInterpolator(tref, Rref)
Thetafun = PchipInterpolator(tref, Thetaref)
Vrfun = PchipInterpolator(tref, Vrref)
Vtfun = PchipInterpolator(tref, Vtref)
mfun = PchipInterpolator(tref, mref)
Ttfun = PchipInterpolator(tref, Ttref)
Trfun = PchipInterpolator(tref, Trref)

Nstates = 5
Ncontrols = 2
obj = Rocket()
nEph = 2

limit_height = 8  # Max height (complexity) of the controller law
limit_size = 100  # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()
################################ M A I N ###############################################


def initPOP1():
    global old_hof
    # this function outputs the first n individuals of the hall of fame of the first GP run
    res = old_hof.shuffle()
    # for i in range(10):
    #   res.append(hof[0])
    return res


def main():
    global tfin, flag, n, old_hof
    global size_gen, size_pop, Mu, Lambda, mutpb, cxpb, Trfun, Ttfun
    global Rfun, Thetafun, Vrfun, Vtfun, mfun

    flag = False

    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)
    old_entropy = 0
    for i in range(100):
        pop = POP(toolbox.population(n=size_pop))
        if pop.entropy > old_entropy and len(pop.indexes) == len(pop.categories) - 1:
            best_pop = pop.items
            old_entropy = pop.entropy

    #if nt > 0:
     #   pop2 = toolbox.popx()
      #  for ind in pop2:
       #     del ind.fitness.values
        #best_pop = pop2 + best_pop

    hof = HallOfFame(10)

    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", Min)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################

    pop, log = eaMuPlusLambdaTol(best_pop, toolbox, Mu, Lambda, size_gen, 0.7, stats=mstats, halloffame=hof,
                                 verbose=True)
    ####################################################################################################################

    pool.close()
    pool.join()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ##############################################


def evaluate(individual):
    global flag
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun, Ttfun
    global tfin, t_eval2, penalty, fit_old, mutpb, cxpb, Cd_new

    penalty = []

    flag = False

    fTr = toolbox.compileR(expr=individual[0])
    fTt = toolbox.compileT(expr=individual[1])

    def sys(t, x):
        global penalty, flag, flag_offdesign, v_wind, flag_thrust, change_time, height_start, delta
        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if R < obj.Re - 0.5 or np.isnan(R):
            penalty.append((R - obj.Re) / obj.Htarget)
            R = obj.Re
            flag = True

        if m < obj.M0 - obj.Mp or np.isnan(m):
            penalty.append((m - (obj.M0 - obj.Mp)) / obj.M0)
            m = obj.M0 - obj.Mp
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
        # em = mf - m

        rho = rho_newmodel(R - obj.Re)
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        Tr = Trfun(t) + fTr(er, evr)
        Tt = Ttfun(t) + fTt(et, evt)

        if np.iscomplex(Tr):
            flag = True
            Tr = 0
        elif Tr < 0.0 or np.isnan(Tr):
            penalty.append((Tr) / obj.Tmax)
            Tr = 0.0
            flag = True
        elif Tr > obj.Tmax or np.isinf(Tr):
            penalty.append((Tr - obj.Tmax) / obj.Tmax)
            Tr = obj.Tmax
            flag = True
        if np.iscomplex(Tt):
            flag = True
            Tt = 0
        elif Tt < 0.0 or np.isnan(Tt):
            penalty.append((Tt) / obj.Tmax)
            Tt = 0.0
            flag = True
        elif Tt > obj.Tmax or np.isinf(Tt):
            penalty.append((Tt - obj.Tmax) / obj.Tmax)
            Tt = obj.Tmax
            flag = True

        dxdt = np.array((Vr,
                         Vt / R,
                         Tr / m - Dr / m - g + Vt ** 2 / R,
                         Tt / m - Dt / m - (Vr * Vt) / R,
                         -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))
        return dxdt

    def find_close(t,x):
        return abs(x[0] - Rfun(t)) - 20
    find_close.terminal = True
    sol = solve_ivp(sys, [t_init, tfin], init_cond, events=find_close)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    #y3 = sol.y[2, :]
    #y4 = sol.y[3, :]
    #y5 = sol.y[4, :]
    tt = sol.t

    #if tt[-1] < tfin:
     #   flag = True
      #  penalty.append((tt[-1] - tfin) / tfin)

    r = Rfun(tt)
    theta = Thetafun(tt)
    #vr = Vrfun(tt)
    #vt = Vtfun(tt)
    #m = mfun(tt)
    # penalty[15] = abs(y1[-1] - r[-1]) / obj.Htarget
    # penalty[16] = np.rad2deg(abs(y2[-1] - theta[-1])) / 60
    err1 = (r - y1) / obj.Htarget
    err2 = (theta - y2) / 0.9
    #err3 = (vr - y3) / 1e3
    #err4 = (vt - y4) / 1e4
    # err5 = (m - y5) / obj.M0

    # STEP TIME SIZE
    #i = 0
    #pp = 1
    #step = np.zeros(len(y1), dtype='float')
    #step[0] = tt[1] - tt[0]
    #while i < len(tt) - 1:
     #   step[pp] = tt[i + 1] - tt[i]
      #  i = i + 1
       # pp = pp + 1

    fitness1 = abs(simps(abs(err1), tt))  # np.sqrt(sum(err1**2)) #+ sum(err3**2))#simps(abs(err1), tt)
    fitness2 = abs(simps(abs(err2), tt))  # np.sqrt(sum(err2**2))# + sum(err4**2))#simps(abs(err2), tt)
    if fitness1 > fitness2:
        use = fitness1
    else:
        use = fitness2
    if penalty != []:
        pen = np.sqrt(sum(np.array(penalty) ** 2))

    if flag is True:
        x = [use, pen]
        return x
    else:
        return [use, 0.0]


####################################    P R I M I T I V E  -  S E T     ################################################

psetR = gp.PrimitiveSet("Radial", 2)
psetR.addPrimitive(operator.add, 2, name="Add")
psetR.addPrimitive(operator.sub, 2, name="Sub")
psetR.addPrimitive(operator.mul, 2, name='Mul')
psetR.addPrimitive(gpprim.TriAdd, 3)
psetR.addPrimitive(np.tanh, 1, name="Tanh")
psetR.addPrimitive(gpprim.Sqrt, 1)
psetR.addPrimitive(gpprim.Log, 1)
psetR.addPrimitive(gpprim.ModExp, 1)
psetR.addPrimitive(gpprim.Sin, 1)
psetR.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    psetR.addEphemeralConstant("randR{}".format(i), lambda: round(random.uniform(-10, 10), 6))

psetR.renameArguments(ARG0='errR')
psetR.renameArguments(ARG1='errVr')
# psetR.renameArguments(ARG2='errm')


psetT = gp.PrimitiveSet("Tangential", 2)
psetT.addPrimitive(operator.add, 2, name="Add")
psetT.addPrimitive(operator.sub, 2, name="Sub")
psetT.addPrimitive(operator.mul, 2, name='Mul')
psetT.addPrimitive(gpprim.TriAdd, 3)
psetT.addPrimitive(np.tanh, 1, name="Tanh")
psetT.addPrimitive(gpprim.Sqrt, 1)
psetT.addPrimitive(gpprim.Log, 1)
psetT.addPrimitive(gpprim.ModExp, 1)
psetT.addPrimitive(gpprim.Sin, 1)
psetT.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    psetT.addEphemeralConstant("randT{}".format(i), lambda: round(random.uniform(-10, 10), 6))

psetT.renameArguments(ARG0='errTheta')
psetT.renameArguments(ARG1='errVt')
# psetT.renameArguments(ARG2='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", FitnessMulti,
               weights=(-1.0, -1.0))  # , -0.1, -0.08, -1.0))    # MINIMIZATION OF THE FITNESS FUNCTION
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", gp.PrimitiveTree)

toolbox = base.Toolbox()

toolbox.register("exprR", gp.genFull, pset=psetR, type_=psetR.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("exprT", gp.genFull, pset=psetT, type_=psetT.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("legR", tools.initIterate, creator.SubIndividual, toolbox.exprR)  ### NEW ###
toolbox.register("legT", tools.initIterate, creator.SubIndividual, toolbox.exprT)  ### NEW ###
toolbox.register("legs", tools.initCycle, list, [toolbox.legR, toolbox.legT], n=1)  ### NEW ###
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)  ### NEW ###
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###
toolbox.register("popx", tools.initIterate, list, initPOP1)
toolbox.register("compileR", gp.compile, pset=psetR)
toolbox.register("compileT", gp.compile, pset=psetT)
toolbox.register("evaluate", evaluate)
toolbox.register("select", InclusiveTournament)
# toolbox.register("select", xselDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True) ### NEW ###
# toolbox.register("select", tools.selNSGA2) ### NEW ###
toolbox.register("mate", xmate)  ### NEW ##
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)  ### NEW ###
toolbox.register("mutate", xmut, expr=toolbox.expr_mut, unipb=0.5, shrpb=0.25, inspb=0.15)  ### NEW ###

toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

toolbox.decorate("mate", staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", staticLimit(key=len, max_value=limit_size))

########################################################################################################################

nt = 0
ntot = 1000
success_time = 0
success_range = 0
success_range_time = 0
pos_final_cond = []
theta_final_cond = []
n_eval = []
t_offdesign = []
#old_hof = HallOfFame(300)
while nt <= ntot:
    print("Iter={}".format(nt))
    obj = Rocket()
    rho_newmodel = []

    def sys2GP_ISA(t, x):
        Cd = obj.Cd

        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        Tr = Trfun(t)
        Tt = Ttfun(t)
        if m <= obj.M0 - obj.Mp:
            Tr = 0.0
            Tt = 0.0
            m = obj.M0 - obj.Mp
        if Tr > obj.Tmax:
            Tr = obj.Tmax
        elif Tr < 0:
            Tr = 0.0
        if Tt > obj.Tmax:
            Tt = obj.Tmax
        elif Tt < 0:
            Tt = 0.0

        rho = isa(R - obj.Re, 1)[1]

        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        dxdt = np.array((Vr,
                         Vt / R,
                         Tr / m - Dr / m - g + Vt ** 2 / R,
                         Tt / m - Dt / m - (Vr * Vt) / R,
                         -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

        return dxdt

    def sys_rho(t, x, expr1, expr2):
        Cd = obj.Cd
        fTr = toolbox.compileR(expr1)
        fTt = toolbox.compileT(expr2)
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        er = Rfun(t) - R
        et = Thetafun(t) - theta
        evr = Vrfun(t) - Vr
        evt = Vtfun(t) - Vt
        # em = mfun(t) - m

        Tr = Trfun(t) + fTr(er, evr)
        Tt = Ttfun(t) + fTt(et, evt)
        if m <= obj.M0 - obj.Mp:
            Tr = 0.0
            Tt = 0.0
            m = obj.M0 - obj.Mp
        if Tr > obj.Tmax:
            Tr = obj.Tmax
        elif Tr < 0:
            Tr = 0.0
        if Tt > obj.Tmax:
            Tt = obj.Tmax
        elif Tt < 0:
            Tt = 0.0

        rho = rho_newmodel(R - obj.Re)

        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        dxdt = np.array((Vr,
                         Vt / R,
                         Tr / m - Dr / m - g + Vt ** 2 / R,
                         Tt / m - Dt / m - (Vr * Vt) / R,
                         -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

        return dxdt

    def sys_rho_init(t, x):
        Cd = obj.Cd

        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        Tr = Trfun(t)
        Tt = Ttfun(t)
        if m <= obj.M0 - obj.Mp:
            Tr = 0.0
            Tt = 0.0
            m = obj.M0 - obj.Mp
        if Tr > obj.Tmax:
            Tr = obj.Tmax
        elif Tr < 0:
            Tr = 0.0
        if Tt > obj.Tmax:
            Tt = obj.Tmax
        elif Tt < 0:
            Tt = 0.0

        rho = rho_newmodel(R - obj.Re)

        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        dxdt = np.array((Vr,
                         Vt / R,
                         Tr / m - Dr / m - g + Vt ** 2 / R,
                         Tt / m - Dt / m - (Vr * Vt) / R,
                         -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

        return dxdt

    '''alt = np.linspace(0, 400000, 1e3)
    alt1 = np.linspace(0, 50000, 1e3)
    plt.figure(0, dpi=230)
    plt.plot(alt1 / 1e3, isa(alt1, 0)[1], marker='.', color='k', linewidth=3, label="ISA")
    plt.plot(alt1 / 1e3, air_density(alt1), 'r--', linewidth=3, label="Density simple model")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Density [kg/m3]")
    plt.legend(loc='best')
    plt.grid()

    plt.figure(1, dpi=230)
    plt.semilogy(alt1/1e3, isa(alt1, 0)[1], marker='.', color='k', linewidth=3, label="ISA")
    plt.semilogy(alt1/1e3, air_density(alt1), 'r--', linewidth=3, label="Density simple model")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Density [kg/m3]")
    plt.legend(loc='best')
    plt.grid()'''

    xini = [obj.Re, 0, 0, 0, obj.M0]
    tt = np.linspace(0, tfin, 1e3)
    ######### for plot ####################
    res_ini = solve_ivp(sys2GP_ISA, [0, tfin], xini, t_eval=tt)  # integrate with standard thrust and ISA atm model
    '''plt.figure(2, dpi=230)
    plt.plot(tref, (Rref - obj.Re)/1e3, 'r--', linewidth=3, label='Reference')
    plt.plot(res_ini.t, (res_ini.y[0, :] - obj.Re)/1e3, linewidth=3, color='k', label='With ISA')
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [km]")
    plt.grid()
    plt.legend(loc='best')

    plt.figure(3, dpi=230)
    plt.plot(tref, np.rad2deg(Thetaref), "r--", linewidth=3, label="Reference")
    plt.plot(res_ini.t, np.rad2deg(res_ini.y[1, :]), linewidth=3, color='k', label="With ISA")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.grid()
    plt.legend(loc='best')'''

    R_true = PchipInterpolator(res_ini.t, res_ini.y[0, :])
    Th_true = PchipInterpolator(res_ini.t, res_ini.y[1, :])
    Vr_true = PchipInterpolator(res_ini.t, res_ini.y[2, :])
    Vt_true = PchipInterpolator(res_ini.t, res_ini.y[3, :])
    m_true = PchipInterpolator(res_ini.t, res_ini.y[4, :])


    def find_diff(t, x):
        '''detects when position error i greater than 50 meters'''
        return 100 - abs(x[0] - Rfun(t))


    find_diff.terminal = True

    ########## start mission #####################
    res = solve_ivp(sys2GP_ISA, [0, tfin], xini, events=find_diff, t_eval=tt)  # integrate with standard thrust and ISA atm model until condition is broke

    r = res.y[0, :]  # values up to the difference in height
    t = res.t

    ########### create new density model #########################

    new_alt = np.linspace(r[-1], 400000 + obj.Re, 1e3)
    rho_new = isa(r - obj.Re, 0)[1]  # new measured values
    rho_old = air_density(new_alt - obj.Re)
    rho_newdata = np.hstack((rho_new, rho_old))
    r_comp = np.hstack((r, new_alt))
    i = 1
    to_remove = []
    while i < len(r_comp):
        if r_comp[i] <= r_comp[i - 1]:
            to_remove.append(i)
        i += 1
    r_comp_i = np.delete(r_comp, to_remove)
    rho_newdata_i = np.delete(rho_newdata, to_remove)
    rho_newmodel = PchipInterpolator(r_comp_i - obj.Re, rho_newdata_i)  # this atm model uses atm data (ISA) until the decision point and then uses standard model

    n = 0

    '''plt.figure(0)
    plt.plot(alt1 / 1e3, rho_newmodel(alt1), label="Model {}".format(n))
    plt.xlabel("Altitude [km]")
    plt.ylabel("Density [kg/m3]")
    #plt.grid()
    plt.legend(loc='best')

    plt.figure(1)
    plt.semilogy(alt1/1e3, rho_newmodel(alt1), label="Model {}".format(n))
    plt.xlabel("Altitude [km]")
    plt.ylabel("Density [kg/m3]")
    #plt.grid()
    plt.legend(loc='best')

    xini_test = [R_true(t[-1]), Th_true(t[-1]), Vr_true(t[-1]), Vt_true(t[-1]), m_true(t[-1])]
    ttt = np.linspace(t[-1], tfin, 1e3)
    test = solve_ivp(sys_rho_init, [t[-1], tfin], xini_test, t_eval=ttt)  # integrate to see where I end up with the new model

    plt.figure(2)
    plt.plot(res.t, (res.y[0, :]-obj.Re)/1e3, label="Model {}".format(n))
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    #plt.grid()
    plt.legend(loc='best')


    plt.figure(3)
    plt.plot(res.t, np.rad2deg(res.y[1, :]), label="Model {}".format(n))
    plt.xlabel("time [s]")
    plt.ylabel("Angle [deg]")
    #plt.grid()
    plt.legend(loc='best')'''

    n += 1

    init_cond = [res.y[0,:][-1], res.y[1,:][-1], res.y[2,:][-1], res.y[3,:][-1], res.y[4,:][-1]]
    t_init = res.t[-1]

    while t[-1] < tfin and n < 10:
        ########################### evaluates new controller ###########################
        delta_eval = 40
        x_ini_evaltime = init_cond
        res_evaltime = solve_ivp(sys2GP_ISA, [t_init, t_init + delta_eval], x_ini_evaltime)  # integrate with standard thrust and ISA to find initial condition for GP after evaluation time
        init_cond = [res_evaltime.y[0, :][-1], res_evaltime.y[1, :][-1], res_evaltime.y[2, :][-1], res_evaltime.y[3, :][-1], res_evaltime.y[4, :][-1]]  # intial conditions for GP
        t_init = res_evaltime.t[-1]

        size_pop = 500# - len(old_hof)  # Pop size
        size_gen = 50  # Gen size
        size_pop_tot = 500
        Mu = int(size_pop_tot)
        Lambda = int(size_pop_tot * 1.4)
        mutpb = 0.7
        cxpb = 0.2
        start = time()
        pop, log, hof = main()  #GP

        #print("Tr: ", hof[-1][0])
        #print("Tt: ", hof[-1][1])
        #old_hof.update(hof[-3:], for_feasible=True)
        end = time()
        t_offdesign.append(end - start)
        print("Time elapsed: {}".format(t_offdesign))
        #stats.append([v_wind, t_offdesign])
        #output = open(
        #    "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Results/2ControlsDelta_Res_{}_{}/hof_Offline.pkl".format(
        #        os.path.basename(__file__), timestr), "wb")  # save of hall of fame after first GP run
        #cPickle.dump(hof, output, -1)
        #output.close()

        ######### here new controller is created based on new model ###############


        tt = np.linspace(t_init, tfin, 1e3)
        find_diff.terminal = True

        res = solve_ivp(partial(sys_rho, expr1=hof[-1][0], expr2=hof[-1][1]), [t_init, tfin], init_cond, events=find_diff, t_eval=tt)  # check where the condition is broke with the new controller

        r = res.y[0, :]
        t = res.t
        init_cond = [res.y[0, :][-1], res.y[1, :][-1], res.y[2, :][-1], res.y[3, :][-1], res.y[4, :][-1]]
        t_init = res.t[-1]
        ############ creation of new model ###################
        new_alt = np.linspace(r[-1], 400000 + obj.Re, 1e3)
        r_isa = np.linspace(0, r[-1], 1e3)
        rho_new = isa(r_isa - obj.Re, 0)[1]  # new measured values
        rho_old = rho_newmodel(new_alt - obj.Re)
        rho_newdata = np.hstack((rho_new, rho_old))
        r_comp = np.hstack((r_isa, new_alt))
        i = 1
        to_remove = []
        while i < len(r_comp):
            if r_comp[i] <= r_comp[i - 1]:
                to_remove.append(i)
            i += 1
        r_comp_i = np.delete(r_comp, to_remove)
        rho_newdata_i = np.delete(rho_newdata, to_remove)
        rho_newmodel = PchipInterpolator(r_comp_i - obj.Re, rho_newdata_i)

        '''plt.figure(0)
        plt.plot(alt1 / 1e3, rho_newmodel(alt1), label="Model {}".format(n))
        plt.legend(loc='best')

        plt.figure(1)
        plt.semilogy(alt1/1e3, rho_newmodel(alt1), label="Model {}".format(n))
        plt.legend(loc='best')

        #test = solve_ivp(sys_rho, [t_init, tfin], xini, t_eval=tt)

        plt.figure(2)
        plt.plot(res.t, (res.y[0, :]-obj.Re)/1e3, label="Model {}".format(n))
        plt.legend(loc='best')

        plt.figure(3)
        plt.plot(res.t, np.rad2deg(res.y[1, :]), label="Model {}".format(n))
        plt.legend(loc='best')'''
        #print(t[-1], tfin)
        if t_offdesign[-1] < delta_eval:
            success_time += 1
        n += 1
    pos_final_cond.append(r[-1])
    theta_final_cond.append(res.y[1, :][-1])

    if Rref[-1]*0.99 < pos_final_cond[-1] < Rref[-1]*1.01 and Thetaref[-1]*0.99 < theta_final_cond[-1] < Thetaref[-1]*1.01:  # tolerance of 1%
        success_range += 1
    all_good = True
    for i in range(n-1):
        if t_offdesign[-1-i] > delta_eval:
            all_good = False
    if all_good is True and Rref[-1]*0.99 < pos_final_cond[-1] < Rref[-1]*1.01 and Thetaref[-1]*0.99 < theta_final_cond[-1] < Thetaref[-1]*1.01:
        success_range_time += 1
    n_eval.append(n-1)
    nt += 1

np.save(savedata_file + "Final_pos", pos_final_cond)
np.save(savedata_file + "Final_ang", theta_final_cond)
np.save(savedata_file + "Success_time", success_time)
np.save(savedata_file + "Success_range", success_range)
np.save(savedata_file + "Success_range_time", success_range_time)
np.save(savedata_file + "N_of_evals", n_eval)
np.save(savedata_file + "Total_iterations", ntot)
np.save(savedata_file + "Real_eval_times", t_offdesign)

plt.show(block=True)




