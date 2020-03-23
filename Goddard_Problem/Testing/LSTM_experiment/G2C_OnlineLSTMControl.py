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


'''definition of GP parameters used'''
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


Ngenes = 1
limit_height = 10
limit_size = 15
nCost = 2
nVars = 2
obj = Rocket()

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

for i in range(nCost):
    psetR.addTerminal("randR{}".format(i), round(random.uniform(-10, 10), 6))

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
psetT.addPrimitive(gpprim.modExp, 1)
psetT.addPrimitive(gpprim.Sin, 1)
psetT.addPrimitive(gpprim.Cos, 1)

for i in range(nCost):
    psetT.addTerminal("randT{}".format(i), round(random.uniform(-10, 10), 6))

psetT.renameArguments(ARG0='errTheta')
psetT.renameArguments(ARG1='errVt')
# psetT.renameArguments(ARG2='errm')


################################################## TOOLBOX #############################################################

d = np.ones((Ngenes+1))*1.0  # weights for linear combination of genes
d[0] = 0

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", list, w=list(d), height=1)
creator.create("Trees", gp.PrimitiveTree)

toolbox = base.Toolbox()

toolbox.register("exprR", gp.genFull, pset=psetR, type_=psetR.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("exprT", gp.genFull, pset=psetT, type_=psetT.ret, min_=1, max_=4)  ### NEW ###


##################  TRAINING DATA  ################################

full_dataset = np.load("Dataset_GoddardForLSTM_withStates_manySamples.npy", allow_pickle=True)
l0 = full_dataset.shape[1]
l1 = full_dataset.shape[2]
dataset = np.load("Dataset_GoddardForLSTM_withStates_manySamples.npy", allow_pickle=True)[0]
dataset = dataset.reshape(l0, l1)
trainset = np.load("Dataset_GoddardForLSTM_withStates_manySamples.npy", allow_pickle=True)[-1]
#dataset = np.delete(dataset, (0), axis=0) # delete first row to have even number of data (950/190=5 -> use 190 samples)
dataset[:,0] = dataset[:,0]-obj.Re
#trainset = np.delete(trainset, (0), axis=0) # delete first row to have even number of data (950/190=5 -> use 190 samples)
trainset[:,0] = trainset[:,0] - obj.Re
#dataset2 = np.load("Dataset_GoddardForLSTM2.npy", allow_pickle=True)[0]
#dataset2 = np.delete(dataset2, (0), axis=0)

samples = list()
samplesy = list()
samples2 = list()
samplesy2 = list()

length = 200
scale_units_x = [max(abs(dataset[:, 0])), max(abs(dataset[:, 1])),max(abs(dataset[:, 2])),max(abs(dataset[:, 3])),
                 max(abs(dataset[:, 4])), max(abs(dataset[:, 5])), max(abs(dataset[:, 6])), max(abs(dataset[:, 7]))]
#scale_units_x = [max(max(abs(dataset[:, 0])), max(abs(dataset2[:, 0]))), max(max(abs(dataset[:, 1])), max(abs(dataset2[:, 1]))),
#                 max(max(abs(dataset[:, 2])), max(abs(dataset2[:, 2]))), max(max(abs(dataset[:, 3])), max(abs(dataset2[:, 3])))]
scale_units_y = [obj.Tmax, obj.Tmax]
# step over the 5,000 in jumps of 200
for i in range(0, len(dataset)-1, length):
    # grab from i to i + 200
    if i+length+1 > len(dataset)-1:
        break
    else:
        sampley = dataset[i+length+1, 8:]/scale_units_y
        sample = dataset[i:i + length, 0:8]/scale_units_x
        samples.append(sample)
        samplesy.append(sampley)

for i in range(0, len(trainset)-1, length):
    # grab from i to i + 200
    if i+length+1 > len(trainset)-1:
        break
    else:
        sampley2 = trainset[i + length + 1, 8:] / scale_units_y
        sample2 = trainset[i:i + length, 0:8] / scale_units_x
        samples2.append(sample2)
        samplesy2.append(sampley2)

datax = np.array((samples))
datay = np.array((samplesy))
trainx = np.array((samples2))
trainy = np.array((samplesy2))
datax = datax.reshape((len(samples), length, 8))
datay = datay.reshape((len(samplesy), 2))
trainx = trainx.reshape((len(samples2), length, 8))
trainy = trainy.reshape((len(samplesy2), 2))

save = True
if save is True:
    ###################### BUILD MODEL ##################################
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(2, return_sequences=True, input_shape=(datax.shape[1], datax.shape[2]), activation="relu"),#, kernel_initializer='Orthogonal'),
        tf.keras.layers.LSTM(5, return_sequences=False, activation='relu'),#,kernel_initializer='Orthogonal'),
        #tf.keras.layers.Dropout(0.05),
        #tf.keras.layers.LSTM(6, return_sequences=False, activation='relu', kernel_initializer='Orthogonal'),
        #tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(2, activation = "relu")])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    history = model.fit(datax, datay, epochs=500, validation_data=(trainx, trainy))
    #history = model.fit(datax2, datay2, epochs=500)
    #plt.figure()
    #plt.plot(history.history['loss'], label="Loss")
    #plt.plot(history.history['accuracy'], linestyle='-', label='Accuracy')
    #plt.legend(loc='best')
    #plt.show(block=True)
    model.save("model_LSTM_2Controls.h5")
    for j in range(1, len(full_dataset)-1):
        dataset = np.load("Dataset_GoddardForLSTM_withStates_manySamples.npy", allow_pickle=True)[j]
        dataset = dataset.reshape(l0, l1)
        dataset[:, 0] = dataset[:, 0] - obj.Re
        samples = list()
        samplesy = list()

        # step over the 5,000 in jumps of 200
        for i in range(0, len(dataset) - 1, length):
            # grab from i to i + 200
            if i + length + 1 > len(dataset) - 1:
                break
            else:
                sampley = dataset[i + length + 1, 8:] / scale_units_y
                sample = dataset[i:i + length, 0:8] / scale_units_x
                samples.append(sample)
                samplesy.append(sampley)

        datax = np.array((samples))
        datay = np.array((samplesy))
        datax = datax.reshape((len(samples), length, 8))
        datay = datay.reshape((len(samplesy), 2))

        model = tf.keras.models.load_model('model_LSTM_2Controls.h5')
        history = model.fit(datax, datay, epochs=500, validation_data=(trainx, trainy))
        model.save("model_LSTM_2Controls.h5")
else:
    model = tf.keras.models.load_model('model_LSTM_2Controls.h5')



###################  PREDICTION WITH NEW VALUES #################################


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




def sys2GP(t, x, model, new_Cd, change_t):
    global Tr_next, Tt_next, predicted, count, errors

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

    if not predicted:
        if count < length-1:
            errors[count] = np.array((R-obj.Re, theta, Vr, Vt, er, et, evr, evt))/scale_units_x
            Tr, Tt = Trfun(t), Ttfun(t)
            count += 1
        else:
            errors[-1] = np.array((R-obj.Re, theta, Vr, Vt, er, et, evr, evt))/scale_units_x
            Tr, Tt = Trfun(t), Ttfun(t)
            errors = errors.reshape(1, length, 8)
            Tr_next, Tt_next = model.predict(errors)[0] * scale_units_y
            predicted = True
    else:
        Tr, Tt = deepcopy(Tr_next), deepcopy(Tt_next)
        errors[:-1] = errors[1:]
        errors[-1] = np.array((R-obj.Re, theta, Vr, Vt, er, et, evr, evt))/scale_units_x
        errors = errors.reshape(1, length, 8)
        Tr_next, Tt_next = model.predict(errors)[0] * scale_units_y
        predicted = True

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
    predicted = False
    count = 1
    print("---------------- ITER {} -----------------".format(n))
    t_test = random.uniform(20, 250)
    Cd_test = random.uniform(0.61, 2.0)
    print("Cd {}, t {}".format(round(Cd_test, 2), round(t_test, 2)))
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]
    t_eval_stand = np.linspace(0.0, tfin, 2000)
    errors = np.zeros((length, 8))
    res_stand = solve_ivp(partial(sys2GP, model=model, new_Cd=Cd_test, change_t=t_test), [0, tfin], x_ini, t_eval=t_eval_stand)
    r_stand = res_stand.y[0, :]
    theta_stand = res_stand.y[1, :]
    vr_stand = res_stand.y[2, :]
    vt_stand = res_stand.y[3, :]
    m_stand = res_stand.y[4, :]
    t_stand = res_stand.t

    plt.ion()
    plt.figure(2)
    plt.plot(t_stand, (r_stand - obj.Re)/1e3, label="LSTM Control Cd {}, t change {}".format(round(Cd_test, 2), round(t_test, 2)))

    plt.figure(3)
    plt.plot(t_stand, vt_stand, label="LSTM Control Cd {}, t change {}".format(round(Cd_test, 2), round(t_test, 2)))

    plt.figure(4)
    plt.plot(t_stand, m_stand, label="LSTM Control Cd {}, t change {}".format(round(Cd_test, 2), round(t_test, 2)))

    plt.figure(5)
    plt.plot(t_stand, vr_stand, label="LSTM Control Cd {}, t change {}".format(round(Cd_test, 2), round(t_test, 2)))

    plt.figure(6)
    plt.plot(t_stand, np.rad2deg(theta_stand), label="LSTM Control Cd {}, t change {}".format(round(Cd_test, 2), round(t_test, 2)))
    n += 1

plt.ion()
plt.figure(2)
plt.plot(tref, (Rref - obj.Re) / 1e3, 'r--', linewidth=3, label="SET POINT")
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc='best')
plt.grid()

plt.figure(3)
plt.plot(tref, Vtref, 'r--', label="SET POINT")
plt.xlabel("time [s]")
plt.ylabel("Tangential Velocity [m/s]")
plt.legend(loc='best')
plt.grid()

plt.figure(4)
plt.plot(tref, mref, 'r--', label="SET POINT")
plt.axhline(obj.M0 - obj.Mp, 0, tfin, color='r')
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc='best')
plt.grid()

plt.figure(5)
plt.plot(tref, Vrref, 'r--', label="SET POINT")
plt.xlabel("time [s]")
plt.ylabel("Radial Velocity [m/s]")
plt.legend(loc='best')
plt.grid()

plt.figure(6)
plt.plot(tref, np.rad2deg(Thetaref), 'r--', linewidth=3, label="SET POINT")
plt.xlabel("time [s]")
plt.ylabel("Angle [deg]")
plt.legend(loc='best')
plt.grid()

plt.show(block=True)


