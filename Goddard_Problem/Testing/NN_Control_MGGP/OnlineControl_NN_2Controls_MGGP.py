import tensorflow as tf
import random
from deap import gp, base, creator
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
import time
import operator
import sys
from copy import deepcopy
import _pickle as cPickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")

import GP_PrimitiveSet as gpprim

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

    def __init__(self, maxsize, similar=operator.eq):
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()
        self.similar = similar

    def shuffle(self):
        arr_start = deepcopy(self.items)
        while len(arr_start) > 0:
            arr_end = []
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

    @staticmethod
    def air_density(h):
        global flag
        beta = 1 / 8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        try:
            return rho0 * np.exp(-beta * h)
        except RuntimeWarning:
            flag = True
            return rho0 * np.exp(-beta * obj.Rtarget)


obj = Rocket()

'''definition of GP parameters used'''

nEph = 6
nCont = 3
psetR = gp.PrimitiveSet("Radial", 2)
psetR.addPrimitive(operator.add, 2, name="Add")
psetR.addPrimitive(operator.sub, 2, name="Sub")
psetR.addPrimitive(operator.mul, 2, name='Mul')
psetR.addPrimitive(gpprim.TriAdd, 3)
psetR.addPrimitive(np.tanh, 1, name="Tanh")
psetR.addPrimitive(gpprim.Sqrt, 1)
psetR.addPrimitive(gpprim.Log, 1)
psetR.addPrimitive(gpprim.modExp, 1)
psetR.addPrimitive(gpprim.Sin, 1)
psetR.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    psetR.addTerminal("randR{}".format(i), round(random.uniform(-10, 10), 6))

psetR.renameArguments(ARG0='errR')
psetR.renameArguments(ARG1='errVr')

psetT = gp.PrimitiveSet("Tangential", 2)
psetT.addPrimitive(operator.add, 2, name="Add")
psetT.addPrimitive(operator.sub, 2, name="Sub")
psetT.addPrimitive(operator.mul, 2, name='Mul')
psetT.addPrimitive(gpprim.TriAdd, 3)
psetT.addPrimitive(np.tanh, 1, name="Tanh")
psetT.addPrimitive(gpprim.Sqrt, 1)
psetT.addPrimitive(gpprim.Log, 1)
psetT.addPrimitive(gpprim.modExp, 1)
psetT.addPrimitive(gpprim.Sin, 1)
psetT.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    psetT.addTerminal("randT{}".format(i), round(random.uniform(-10, 10), 6))

psetT.renameArguments(ARG0='errTheta')
psetT.renameArguments(ARG1='errVt')

Ngenes = 1
d = np.ones((Ngenes+1))*1.0  # weights for linear combination of genes
d[0] = 0
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", list, w=list(d), height=1)
creator.create("Trees", gp.PrimitiveTree)
#################### RETRIEVE COEFFICIENTS  ##########################

w1 = {"name": "w1", "value": 1}
w2 = {"name": "w2", "value": 1}
w3 = {"name": "w3", "value": 1}
w4 = {"name": "w4", "value": 1}

arguments = {"errR": w1,
             "errTheta": w2,
             "errVr": w3,
             "errVt": w4}

'''Retrieve the GP tree through pickle'''
objects = []
with (open("hof_G2C_forNN.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(cPickle.load(openfile))
        except EOFError:
            break


##################  TRAINING DATA  ################################
tref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/time.npy")
total_time_simulation = tref[-1]
tfin = tref[-1]

Rref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/R.npy")
Thetaref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Theta.npy")
Vrref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Vr.npy")
Vtref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Vt.npy")
mref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/m.npy")
Ttref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Tt.npy")
Trref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Tr.npy")

Rfun = PchipInterpolator(tref, Rref)
Thetafun = PchipInterpolator(tref, Thetaref)
Vrfun = PchipInterpolator(tref, Vrref)
Vtfun = PchipInterpolator(tref, Vtref)
mfun = PchipInterpolator(tref, mref)
Ttfun = PchipInterpolator(tref, Ttref)
Trfun = PchipInterpolator(tref, Trref)

dataset = np.load("dataset_forNN_MGGP_50samples.npy", allow_pickle=True)

np.random.shuffle(dataset)
dataset[:,0] = dataset[:,0] - obj.Re
scale_units_x = [max(abs(dataset[:, 0])), max(abs(dataset[:, 1])),max(abs(dataset[:, 2])),max(abs(dataset[:, 3])),
                 max(abs(dataset[:, 4])), max(abs(dataset[:, 5])), max(abs(dataset[:, 6])), max(abs(dataset[:, 7]))]
scale_units_y = []
for i in range(8, dataset.shape[1]):
    scale_units_y.append(max(abs(dataset[:, i])))


datax = np.vstack((dataset[:, 0:8]/scale_units_x))#, dataset2[:, 0:4]/scale_units_x))
datay = np.vstack((dataset[:, 8:]/scale_units_y))#, dataset2[:, 4:]/scale_units_y))


save = True
if save is True:
    ###################### BUILD MODEL ##################################
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(13, activation='relu', kernel_initializer='RandomNormal', bias_initializer='Zeros'),
        #tf.keras.layers.Dense(20, activation='relu'),
        #tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(dataset.shape[1]-8)])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss='mean_squared_error')
    model.summary()
    history = model.fit(datax, datay, epochs=500, validation_split=0.1)
    plt.figure()
    plt.plot(history.history['loss'], label="Loss")
    plt.plot(history.history['val_loss'], linestyle='-', label='Validation Loss')
    plt.legend(loc='best')
    plt.show(block=True)
    model.save("mode_NN_2Controls_10samples.h5")
else:
    model = tf.keras.models.load_model('mode_NN_2Controls_10samples.h5')


###################  PREDICTION WITH NEW VALUES #################################


def sys2GP(t, x, expr1, expr2, new_Cd, change_t, Control, model):

    if t >= change_t:
        Cd = new_Cd
    else:
        Cd = obj.Cd

    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt

    if Control is True:
        input = np.array(([[R - obj.Re, theta, Vr, Vt, er, et, evr, evt]])) / scale_units_x
        prediction = model.predict(input)[0]*scale_units_y
        ex1, s = update_eq(expr1, prediction, 'R', 0)
        ex2, e = update_eq(expr2, prediction, 'T', s)
        fTr = gp.compile(ex1, pset=psetR)
        fTt = gp.compile(ex2, pset=psetT)
        er = (r - R)*prediction[0]
        et = (th - theta)*prediction[1]
        evr = (vr - Vr)*prediction[2]
        evt = (vt - Vt)*prediction[3]
    else:
        fTr = gp.compile(expr1, pset=psetR)
        fTt = gp.compile(expr2, pset=psetT)

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
    rho = obj.air_density(R - obj.Re)
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

individual = objects[0][-1]

ws = 1
eqR = str(individual[0].w[0]) + "+"
while ws < len(individual[0].w):
    eqR = eqR + str(individual[0].w[ws]) + "*" + str(individual[0][ws - 1]) + "+"
    ws += 1
eqR = list(eqR)
del eqR[-1]
eqR = "".join(eqR)

eqT = str(individual[1].w[0]) + "+"
ws = 1
while ws < len(individual[1].w):
    eqT = eqT + str(individual[1].w[ws]) + "*" + str(individual[1][ws - 1]) + "+"
    ws += 1
eqT = list(eqT)
del eqT[-1]
eqT = "".join(eqT)

expr1 = eqR
expr2 = eqT

Nvar = 4

def update_eq(individual, prediction, which, r):
    count = 0
    for j in range(len(individual)):
        if type(individual[j]) == gp.Terminal and individual[j].name[0] != "A":
            individual[j] = deepcopy(individual[j])
            individual[j].value = float(prediction[r+Nvar])
            individual[j].name = str(prediction[r+Nvar])
            count += 1
            r += 1
    return individual, r

n_tot = 10
n = 0
nn_sc = 0
gp_sc = 0
nn_over_gp = 0
gp_over_nn = 0
nn_prec_gp = 0
gp_prec_nn = 0
data_nn = []
data_gp = []
while n < n_tot:
    print("---------------- ITER {} -----------------".format(n))
    t_test = random.uniform(20, 200)
    Cd_test = random.uniform(0.61, 2.0)
    if t_test < 100 and Cd_test > 1:
        Cd_test = random.uniform(0.61, 1)
    print("Cd {}, t {}".format(round(Cd_test,2), round(t_test,2)))
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]
    t_eval_stand = np.linspace(0.0, tfin, 500)
    res_stand = solve_ivp(partial(sys2GP, expr1=expr1, expr2=expr2, new_Cd=Cd_test, change_t=t_test, Control=False, model=[]), [0, tfin], x_ini, t_eval=t_eval_stand)
    r_stand = res_stand.y[0, :]
    theta_stand = res_stand.y[1, :]
    vr_stand = res_stand.y[2, :]
    vt_stand = res_stand.y[3, :]
    m_stand = res_stand.y[4, :]
    t_stand = res_stand.t

    res2 = solve_ivp(partial(sys2GP, expr1=expr1, expr2=expr2, new_Cd=Cd_test, change_t=t_test, Control=True, model=model), [0.0, tfin], x_ini, t_eval=t_eval_stand)
    r_cont = res2.y[0, :]
    theta_cont = res2.y[1, :]
    vr_cont = res2.y[2, :]
    vt_cont = res2.y[3, :]
    m_cont = res2.y[4, :]
    t_cont = res2.t

    if (Rref[-1]-obj.Re)*0.99 < (r_cont[-1]-obj.Re) < (Rref[-1]-obj.Re)*1.01 and Thetaref[-1]*0.99 < theta_cont[-1] < Thetaref[-1]*1.01:  # tolerance of 1%
        print("NN success")
        nn_sc += 1
    if (Rref[-1]-obj.Re)*0.99 < (r_stand[-1]-obj.Re) < (Rref[-1]-obj.Re)*1.01 and Thetaref[-1]*0.99 < theta_stand[-1] < Thetaref[-1]*1.01:  # tolerance of 1%
        print("GP success")
        gp_sc += 1
    if (Rref[-1]-obj.Re)*0.99 < (r_cont[-1]-obj.Re) < (Rref[-1]-obj.Re)*1.01 and Thetaref[-1]*0.99 < theta_cont[-1] < Thetaref[-1]*1.01 and not ((Rref[-1]-obj.Re)*0.99 < (r_stand[-1]-obj.Re) < (Rref[-1]-obj.Re)*1.01 and Thetaref[-1]*0.99 < theta_stand[-1] < Thetaref[-1]*1.01):  # tolerance of 1%
        print("NN better than GP")
        nn_over_gp += 1
        data_nn.append([Cd_test, t_test])
    if (Rref[-1]-obj.Re)*0.99 < (r_stand[-1]-obj.Re) < (Rref[-1]-obj.Re)*1.01 and Thetaref[-1]*0.99 < theta_stand[-1] < Thetaref[-1]*1.01 and not ((Rref[-1]-obj.Re) < (r_cont[-1]-obj.Re) < (Rref[-1]-obj.Re)*1.01 and Thetaref[-1]*0.99 < theta_cont[-1] < Thetaref[-1]*1.01):  # tolerance of 1%
        print("GP better than NN")
        gp_over_nn += 1
        data_gp.append([Cd_test, t_test])
    if (abs(Rref[-1] - r_cont[-1]) < abs(Rref[-1] - r_stand[-1])) and (abs(Thetaref[-1]-theta_cont[-1]) < abs(Thetaref[-1] - theta_stand[-1])):
        print("NN more precise than GP")
        nn_prec_gp += 1
    elif (abs(Rref[-1] - r_cont[-1]) > abs(Rref[-1] - r_stand[-1])) and (abs(Thetaref[-1]-theta_cont[-1]) > abs(Thetaref[-1] - theta_stand[-1])):
        print("GP more precise than NN")
        gp_prec_nn += 1
    print("\n")

    plt.ion()
    plt.figure(2)
    plt.plot(tref, (Rref - obj.Re) / 1e3, 'r--', linewidth=3, label="SET POINT")
    plt.plot(t_stand, (r_stand - obj.Re)/1e3, label="Standard")
    plt.plot(t_cont, (r_cont - obj.Re)/1e3, label="NN control")
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.legend(loc='best')
    plt.grid()
    
    plt.figure(3)
    plt.plot(tref, Vtref, 'r--', label="SET POINT")
    plt.plot(t_stand, vt_stand, label="Standard")
    plt.plot(t_cont, vt_cont, label="NN control")
    plt.xlabel("time [s]")
    plt.ylabel("Tangential Velocity [m/s]")
    plt.legend(loc='best')
    plt.grid()
    
    plt.figure(4)
    plt.plot(tref, mref, 'r--', label="SET POINT")
    plt.plot(t_stand, m_stand, label="Standard")
    plt.plot(t_cont, m_cont, label="NN control")
    plt.axhline(obj.M0 - obj.Mp, 0, tfin, color='r')
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc='best')
    plt.grid()
    
    plt.figure(5)
    plt.plot(tref, Vrref, 'r--', label="SET POINT")
    plt.plot(t_stand, vr_stand, label="Standard")
    plt.plot(t_cont, vr_cont, label="NN control")
    plt.xlabel("time [s]")
    plt.ylabel("Radial Velocity [m/s]")
    plt.legend(loc='best')
    plt.grid()
    
    plt.figure(6)
    plt.plot(tref, np.rad2deg(Thetaref), 'r--', linewidth=3, label="SET POINT")
    plt.plot(t_stand, np.rad2deg(theta_stand), label="Standard")
    plt.plot(t_cont, np.rad2deg(theta_cont), label="NN control")
    plt.xlabel("time [s]")
    plt.ylabel("Angle [deg]")
    plt.legend(loc='best')
    plt.grid()

    n += 1

print("NN success {}%".format(round(nn_sc/n_tot*100,2)))
print("GP success {}%".format(round(gp_sc/n_tot*100,2)))
print("NN better than GP {}%".format(round(nn_over_gp/n_tot*100,2)))
print("GP better than NN {}%".format(round(gp_over_nn/n_tot*100,2)))
print("NN more precise than GP {}%".format(round(nn_prec_gp/n_tot*100,2)))
print("GP more precise than NN {}%".format(round(gp_prec_nn/n_tot*100,2)))
plt.figure(0)
plt.title("Number of successes {}/{}".format(nn_over_gp+gp_over_nn, n_tot))
for i in range(len(data_nn)):
    plt.plot(data_nn[i][1], data_nn[i][0], marker='.', color='k')
for i in range(len(data_gp)):
    plt.plot(data_gp[i][1], data_gp[i][0], marker='x', color='r')
plt.show(block=True)


