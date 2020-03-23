import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import statistics

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


def sys2GP(t, x):
    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    Cd = obj.Cd

    Tr = Trfun(t)
    Tt = Ttfun(t)

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
tev = np.linspace(0.0, tfin, 50000)
ttest = np.linspace(0.0, tfin, 2000)
create_dataset = False
if create_dataset:
    solgp = solve_ivp(sys2GP, [0.0, tfin], x_start, t_eval=tev, method='BDF')

    dataset = np.column_stack((solgp.t.reshape(len(solgp.t),1),
                               solgp.y[0,:].reshape(len(solgp.t),1),
                               solgp.y[1,:].reshape(len(solgp.t),1),
                               solgp.y[2,:].reshape(len(solgp.t),1),
                               solgp.y[3,:].reshape(len(solgp.t),1),
                               solgp.y[4,:].reshape(len(solgp.t),1)))
    np.save("datasetLSTM.npy", dataset)

dataset = np.column_stack((tev.reshape(len(tev),1),
                           Rfun(tev).reshape(len(tev),1),
                           Thetafun(tev).reshape(len(tev),1),
                           Vrfun(tev).reshape(len(tev),1),
                           Vtfun(tev).reshape(len(tev),1),
                           mfun(tev).reshape(len(tev),1)))
testset = np.column_stack((ttest.reshape(len(ttest),1),
                           Rfun(ttest).reshape(len(ttest),1),
                           Thetafun(ttest).reshape(len(ttest),1),
                           Vrfun(ttest).reshape(len(ttest),1),
                           Vtfun(ttest).reshape(len(ttest),1),
                           mfun(ttest).reshape(len(ttest),1)))
#dataset = np.load("datasetLSTM.npy")

dataset[:,1] = dataset[:,1] - obj.Re
testset[:,1] = testset[:,1] - obj.Re
'''def scale(data):
    mean = np.mean(data)
    variance = statistics.variance(data)
    data_scaled = (data - mean)/variance
    return data_scaled'''

scale_units_x = [max(abs(dataset[:, 0])), max(abs(dataset[:, 1])),max(abs(dataset[:, 2])),max(abs(dataset[:, 3])),
                max(abs(dataset[:, 4])), max(abs(dataset[:, 5]))]
scale_units_y = [max(abs(dataset[:, 1])),max(abs(dataset[:, 2])),max(abs(dataset[:, 3])),
                 max(abs(dataset[:, 4])), max(abs(dataset[:, 5]))]
'''orig_mean = []
orig_variance = []
for i in range(dataset.shape[1]):
    mean = np.mean(dataset[:, i])
    orig_mean.append(mean)
    variance = statistics.variance(dataset[:, i])
    orig_variance.append(variance)
    dataset[:, i] = (dataset[:, i] - mean)/variance'''

length = 10
samples = list()
samplesy = list()
for i in range(0, len(dataset)-1, length):
    # grab from i to i + 200
    if i+length+1 > len(dataset)-1:
        break
    else:
        sample = dataset[i:i + length, :]
        sampley = dataset[i+length+1, 1:]
        samples.append(sample)
        samplesy.append(sampley)

datax = np.array((samples))/scale_units_x
datay = np.array((samplesy))/scale_units_y
datax = datax.reshape((len(samples), length, 6))
datay = datay.reshape((len(samplesy), 5))

test_samp = []
test_sampy = []
for i in range(0, len(testset)-1, length):
    # grab from i to i + 200
    if i+length+1 > len(testset)-1:
        break
    else:
        test_s = testset[i:i + length, :]
        test_sy = testset[i+length+1, 1:]
        test_samp.append(test_s)
        test_sampy.append(test_sy)

testx = np.array((test_samp))/scale_units_x
testy = np.array((test_sampy))/scale_units_y
testx = testx.reshape((len(test_samp), length, 6))
testy = testy.reshape((len(test_sampy), 5))

save = True
if save is True:
    ###################### BUILD MODEL ##################################
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(12, return_sequences=True, input_shape=(datax.shape[1], datax.shape[2])), #, kernel_initializer='Orthogonal'),
        #tf.keras.layers.Dropout(0.01),
        tf.keras.layers.LSTM(12, return_sequences=False),#, kernel_initializer='Orthogonal'),
        #tf.keras.layers.Dropout(0.0001),
        #tf.keras.layers.LSTM(5, return_sequences=True),  # , kernel_initializer='Orthogonal'),
        #tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.LSTM(5, return_sequences=True),  # , kernel_initializer='Orthogonal'),
        #tf.keras.layers.Dropout(0.05),
        #tf.keras.layers.LSTM(5, return_sequences=False),  # , kernel_initializer='Orthogonal'),
        #tf.keras.layers.Dropout(0.05),
        #tf.keras.layers.LSTM(20, return_sequences=False),  # , kernel_initializer='Orthogonal'),
        #tf.keras.layers.Dropout(0.1),
        #tf.keras.layers.LSTM(15, return_sequences=False, activation='relu', kernel_initializer='Orthogonal'),
        # tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(5, activation="relu")])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    history = model.fit(datax, datay, epochs=500, validation_data=[testx, testy])
    # history = model.fit(datax2, datay2, epochs=500)
    plt.figure()
    plt.plot(history.history['loss'], label="Loss")
    plt.plot(history.history['val_loss'], linestyle='-', label='Validation Loss')
    plt.legend(loc='best')
    plt.show(block=True)
    model.save("model_LSTM_2Controls_dynamics.h5")
else:
    model = tf.keras.models.load_model('model_LSTM_2Controls_dynamics.h5')

n = 0
init_cond = [obj.Re, 0.0, 0.0, 0.0, obj.M0]
tev2 = np.linspace(0.0, 50, int((50/tfin)*2e3))

r_cont = Rfun(tev2)-obj.Re
theta_cont = Thetafun(tev2)
vr_cont = Vrfun(tev2)
vt_cont = Vtfun(tev2)
m_cont = mfun(tev2)
r_tot = list(r_cont)
theta_tot = list(theta_cont)
vr_tot = list(vr_cont)
vt_tot = list(vt_cont)
m_tot = list(m_cont)
t_tot = list(tev2)
disc = 0.1
while tev2[-1]<tfin:

    input = np.column_stack((tev2[-length:].reshape(length, 1), r_cont[-length:].reshape(length, 1), theta_cont[-length:].reshape(length, 1), vr_cont[-length:].reshape(length, 1),
                             vt_cont[-length:].reshape(length, 1), m_cont[-length:].reshape(length, 1)))/scale_units_x
    #for i in range(input.shape[1]):
    #    input[:, i] = (input[:, i] - orig_mean[i]) / orig_variance[i]
    input = input.reshape(1, length, 6)
    prediction = model.predict(input)[0]*scale_units_y
    #for i in range(prediction.shape[0]):
    #    prediction[i] = prediction[i] * orig_variance[i+1] + orig_mean[i+1]

    tev2 = np.append(tev2, tev2[-1]+disc)
    t_tot.append(tev2[-1]+disc)
    tev2 = np.delete(tev2, 0)
    r_cont = np.append(r_cont, prediction[0])
    r_cont = np.delete(r_cont, 0)
    r_tot.append(prediction[0])
    theta_cont = np.append(theta_cont, prediction[1])
    theta_cont = np.delete(theta_cont, 0)
    theta_tot.append(prediction[1])
    vr_cont = np.append(vr_cont, prediction[2])
    vr_cont = np.delete(vr_cont, 0)
    vr_tot.append(prediction[2])
    vt_cont = np.append(vt_cont, prediction[3])
    vt_cont = np.delete(vt_cont, 0)
    vt_tot.append(prediction[3])
    m_cont = np.append(m_cont, prediction[4])
    m_cont = np.delete(m_cont, 0)
    m_tot.append(prediction[4])


plt.ion()
plt.figure(2)
plt.plot(tref, (Rref - obj.Re) / 1e3, 'r--', linewidth=3, label="SET POINT")
plt.plot(t_tot, np.array(r_tot) / 1e3, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(3)
plt.plot(tref, Vtref, 'r--', label="SET POINT")
plt.plot(t_tot, vt_tot, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Tangential Velocity [m/s]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(4)
plt.plot(tref, mref, 'r--', label="SET POINT")
plt.axhline(obj.M0 - obj.Mp, 0, tfin, color='r')
plt.plot(t_tot, m_tot, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(5)
plt.plot(tref, Vrref, 'r--', label="SET POINT")
plt.plot(t_tot, vr_tot, label="NN")
plt.xlabel("time [s]")
plt.ylabel("Radial Velocity [m/s]")
plt.legend(loc='best', ncol=2)
plt.grid()

plt.figure(6)
plt.plot(tref, np.rad2deg(Thetaref), 'r--', linewidth=3, label="SET POINT")
plt.plot(t_tot, np.rad2deg(theta_tot), label="NN")
plt.xlabel("time [s]")
plt.ylabel("Angle [deg]")
plt.legend(loc='best', ncol=2)
plt.grid()

'''plt.figure(7)
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
plt.grid()'''

plt.show(block=True)