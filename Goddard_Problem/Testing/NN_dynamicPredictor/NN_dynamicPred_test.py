import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import time

laptop = True
if laptop:
    initial_path = "/home/francesco/Desktop/PhD/Git_workspace/IC4A2S"
else:
    initial_path = "/home/francesco/Desktop/Git_workspace/IC4A2S"

Rref = np.load(initial_path + "/Goddard_Problem/2Controls/R.npy")
Thetaref = np.load(initial_path + "/Goddard_Problem/2Controls/Theta.npy")
Vrref = np.load(initial_path + "/Goddard_Problem/2Controls/Vr.npy")
Vtref = np.load(initial_path + "/Goddard_Problem/2Controls/Vt.npy")
mref = np.load(initial_path + "/Goddard_Problem/2Controls/m.npy")
Ttref = np.load(initial_path + "/Goddard_Problem/2Controls/Tt.npy")
Trref = np.load(initial_path + "/Goddard_Problem/2Controls/Tr.npy")
tref = np.load(initial_path + "/Goddard_Problem/2Controls/time.npy")
tfin = tref[-1]

Rfun = PchipInterpolator(tref, Rref)
Thetafun = PchipInterpolator(tref, Thetaref)
Vrfun = PchipInterpolator(tref, Vrref)
Vtfun = PchipInterpolator(tref, Vtref)
mfun = PchipInterpolator(tref, mref)
Trfun = PchipInterpolator(tref, Trref)
Ttfun = PchipInterpolator(tref, Ttref)


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


def sys2GP(t, x, model):
    global er, et, evr, evt, Rstat, thetastat, vrstat, vtstat, Trstat, Ttstat, tstat
    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)

    Cd = obj.Cd
    if model:
        input = np.array(([[t, R - obj.Re, theta, Vr, Vt]])) / scale_units_x
        prediction = model.predict(input)[0]*scale_units_y
        Tr = prediction[0]
        Tt = prediction[1]
    else:
        Rstat.append(R)
        thetastat.append(theta)
        vrstat.append(Vr)
        vtstat.append(Vt)
        Tr = Trfun(t)
        Tt = Ttfun(t)
        Trstat.append(Tr)
        Ttstat.append(Tt)
        er.append(r - R)
        et.append(th - theta)
        evr.append(vr - Vr)
        evt.append(vt - Vt)
        tstat.append(t)

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

x_start = [obj.Re, 0.0, 0.0, 0.0, obj.M0]
tev = np.linspace(0.0, tfin, 200000)
create_dataset = True
if create_dataset:
    er = []
    et = []
    evr = []
    evt = []
    Rstat = []
    thetastat = []
    vrstat = []
    vtstat = []
    Trstat = []
    Ttstat = []
    tstat = []

    solgp = solve_ivp(partial(sys2GP, model=[]), [0.0, tfin], x_start, t_eval=tev, method='BDF')

    er = np.array(er)
    et = np.array(et)
    evr = np.array(evr)
    evt = np.array(evt)
    Rstat = np.array(Rstat)
    thetastat = np.array(thetastat)
    vrstat = np.array(vrstat)
    vtstat = np.array(vtstat)
    Trstat = np.array(Trstat)
    Ttstat = np.array(Ttstat)
    tstat = np.array(tstat)
    dataset = np.column_stack((solgp.t.reshape(len(solgp.t),1),
                               solgp.y[0,:].reshape(len(solgp.t),1),
                               solgp.y[1,:].reshape(len(solgp.t),1),
                               solgp.y[2,:].reshape(len(solgp.t),1),
                               solgp.y[3,:].reshape(len(solgp.t),1),
                               Trfun(solgp.t).reshape(len(solgp.t),1),
                               Ttfun(solgp.t).reshape(len(solgp.t),1)))
    dataset2 = np.column_stack((tstat.reshape(len(tstat), 1),
                               Rstat.reshape(len(tstat), 1),
                               thetastat.reshape(len(tstat), 1),
                               vrstat.reshape(len(tstat), 1),
                               vtstat.reshape(len(tstat), 1),
                               Trstat.reshape(len(tstat), 1),
                               Ttstat.reshape(len(tstat), 1)))
    np.save("dataset_dynPred.npy", dataset2)

dataset = np.load("dataset_dynPred.npy")
np.random.shuffle(dataset)
#np.random.shuffle(dataset)
dataset[:,1] = dataset[:,1] - obj.Re
scale_units_x = [max(abs(dataset[:, 0])), max(abs(dataset[:, 1])),max(abs(dataset[:, 2])),max(abs(dataset[:, 3])),
                 max(abs(dataset[:, 4]))]
scale_units_y = []
for i in range(5, dataset.shape[1]):
    scale_units_y.append(max(abs(dataset[:, i])))


datax = np.vstack((dataset[:, 0:5]/scale_units_x))#, dataset2[:, 0:4]/scale_units_x))
datay = np.vstack((dataset[:, 5:]/scale_units_y))#, dataset2[:, 4:]/scale_units_y))

test_perc = 0.1

testx = datax[int(len(datax)*(1-test_perc)):]
testy = datay[int(len(datay)*(1-test_perc)):]
datax = datax[:int(len(datax)*(1-test_perc))]
datay = datay[:int(len(datay)*(1-test_perc))]

save = True
if save is True:
    ###################### BUILD MODEL ##################################
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(10, activation='relu', kernel_initializer='RandomNormal', bias_initializer='Zeros'),
        tf.keras.layers.Dropout(0.1),
        #tf.keras.layers.Dense(150, activation='relu', kernel_initializer='RandomNormal', bias_initializer='Zeros'),
        #tf.keras.layers.Dropout(0.1),
        #tf.keras.layers.Dense(150, activation='relu', kernel_initializer='RandomNormal', bias_initializer='Zeros'),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(dataset.shape[1]-5)])
    model.compile(optimizer='adam',
                  loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    history = model.fit(datax, datay, epochs=100, validation_data=[testx, testy])

    #plt.figure()
    #plt.plot(history.history['loss'], label="Loss")
    #plt.plot(history.history['val_loss'], linestyle='-', label='Validation Loss')
    #plt.legend(loc='best')
    #plt.show(block=True)
    model.save("mode_NN_2Controls_dynamics.h5")
else:
    model = tf.keras.models.load_model('mode_NN_2Controls_dynamics.h5')

tev2 = np.linspace(0.0, tfin, 1000)
res2 = solve_ivp(partial(sys2GP, model=model), [0.0, tfin], x_start, method='BDF')#, t_eval=tev2)
t_cont = res2.t
r_cont = res2.y[0,:]
theta_cont = res2.y[1,:]
vr_cont = res2.y[2,:]
vt_cont = res2.y[3,:]
m_cont = res2.y[4,:]

if (Rref[-1] - obj.Re) * 0.99 < (r_cont[-1] - obj.Re) < (Rref[-1] - obj.Re) * 1.01 and Thetaref[-1] * 0.99 < theta_cont[-1] < Thetaref[-1] * 1.01:  # tolerance of 1%
    print("NN success")


Tr = []
Tt = []
start = time.time()
for i in range(len(t_cont)):
    input = np.array((t_cont[i], r_cont[i] - obj.Re, theta_cont[i], vr_cont[i], vt_cont[i])) / scale_units_x
    prediction = model.predict(input.reshape(1, 5))[0]*scale_units_y
    Tr.append(prediction[0])
    Tt.append(prediction[1])
end = time.time()
print("NN propagation took {} s".format(round(end-start, 2)))
plt.ion()
plt.figure(2)
plt.plot(tref, (Rref - obj.Re) / 1e3, 'r--', linewidth=3, label="SET POINT")
plt.plot(t_cont, (r_cont - obj.Re) / 1e3, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(3)
plt.plot(tref, Vtref, 'r--', label="SET POINT")
plt.plot(t_cont, vt_cont, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Tangential Velocity [m/s]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(4)
plt.plot(tref, mref, 'r--', label="SET POINT")
plt.axhline(obj.M0 - obj.Mp, 0, tfin, color='r')
plt.plot(t_cont, m_cont, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(5)
plt.plot(tref, Vrref, 'r--', label="SET POINT")
plt.plot(t_cont, vr_cont, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Radial Velocity [m/s]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(6)
plt.plot(tref, np.rad2deg(Thetaref), 'r--', linewidth=3, label="SET POINT")
plt.plot(t_cont, np.rad2deg(theta_cont), label="NN")
plt.xlabel("time [s]")
plt.ylabel("Angle [deg]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(7)
plt.plot(tref, Trref, 'r--', linewidth=3, label="SET POINT")
plt.plot(t_cont, Tr, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Thrust [kN]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(8)
plt.plot(tref, Ttref, 'r--', linewidth=3, label="SET POINT")
plt.plot(t_cont, Tt, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Thrust [kN]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.show(block=True)