import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, splrep
from scipy.integrate import solve_ivp
from functools import partial
from copy import deepcopy



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

obj = Rocket()
rho_newmodel = []

def sys2GP(t, x):

    Cd = obj.Cd

    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    Tr = Trfun(t)
    Tt = Ttfun(t)
    if m <= obj.M0-obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp
    if Tr>obj.Tmax:
        Tr = obj.Tmax
    elif Tr < 0:
        Tr = 0.0
    if Tt>obj.Tmax:
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

def sys_rho(t, x):

    Cd = obj.Cd

    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    Tr = Trfun(t)
    Tt = Ttfun(t)
    if m <= obj.M0-obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp
    if Tr>obj.Tmax:
        Tr = obj.Tmax
    elif Tr < 0:
        Tr = 0.0
    if Tt>obj.Tmax:
        Tt = obj.Tmax
    elif Tt < 0:
        Tt = 0.0

    rho = rho_newmodel(R-obj.Re)

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

alt = np.linspace(0, 400000, 1e3)
plt.figure(1)
plt.plot(alt, air_density(alt), marker='.',label="Density simple")
plt.plot(alt, isa(alt, 0)[1], marker='.',label="ISA")
plt.legend(loc='best')

xini = [obj.Re, 0, 0, 0, obj.M0]
tt = np.linspace(0, tfin, 1e3)
res_ini = solve_ivp(sys2GP, [0, tfin], xini, t_eval=tt)  # integrate with standard thrust and ISA atm model
plt.figure(2)
plt.plot(tref, Rref, 'r--', label='Reference')
plt.plot(res_ini.t, res_ini.y[0,:], label='With ISA')
plt.legend(loc='best')
R_true = PchipInterpolator(res_ini.t, res_ini.y[0, :])
Th_true = PchipInterpolator(res_ini.t, res_ini.y[1, :])
Vr_true = PchipInterpolator(res_ini.t, res_ini.y[2, :])
Vt_true = PchipInterpolator(res_ini.t, res_ini.y[3, :])
m_true = PchipInterpolator(res_ini.t, res_ini.y[4, :])


def find_diff(t, x):
    return 50 - abs(x[0] - Rfun(t))
find_diff.terminal = True

res = solve_ivp(sys2GP, [0, tfin], xini, events=find_diff, t_eval=tt)  # integrate with standard thrust and ISA atm model until condition is broke

r = res.y[0,:]  # values up to the difference in height
t = res.t
new_alt = np.linspace(r[-1], 400000+obj.Re, 1e3)
rho_new = isa(r - obj.Re, 0)[1] # new measured values
rho_old = air_density(new_alt-obj.Re)
rho_newdata = np.hstack((rho_new, rho_old))
r_comp = np.hstack((r, new_alt))
i=1
to_remove = []
while i < len(r_comp):
    if r_comp[i] <= r_comp[i-1]:
        to_remove.append(i)
    i += 1
r_comp_i = np.delete(r_comp, to_remove)
rho_newdata_i = np.delete(rho_newdata, to_remove)
rho_newmodel = PchipInterpolator(r_comp_i-obj.Re, rho_newdata_i) # this atm model uses atm data (ISA) until the decision point and then uses standard model
n = 0
plt.figure(1)
plt.plot(alt, rho_newmodel(alt), label="Model {}".format(n))
plt.legend(loc='best')

xini_test = [R_true(t[-1]), Th_true(t[-1]), Vr_true(t[-1]), Vt_true(t[-1]), m_true(t[-1])]
ttt = np.linspace(t[-1], tfin, 1e3)
test = solve_ivp(sys_rho, [t[-1], tfin], xini_test, t_eval=ttt)  # integrate to see where I end up with the new model

plt.figure(2)
plt.plot(test.t, test.y[0, :], label="Model {}".format(n))
plt.legend(loc='best')

while t[-1] < tfin and n < 5:
    xini = [Rfun(t[-1]+10), Thetafun(t[-1]+10), Vrfun(t[-1]+10), Vtfun(t[-1]+10), mfun(t[-1]+10)]
    tt = np.linspace(t[-1]+10, tfin, 1e3)
    find_diff.terminal = True
    init_t = t[-1] + 10
    res = solve_ivp(sys_rho, [t[-1]+10, tfin], xini, events=find_diff, t_eval=tt)  # necessary to decide when the new model starts

    r = res.y[0,:]
    t = res.t

    new_alt = np.linspace(r[-1], 400000+obj.Re, 1e3)
    r_isa = np.linspace(0, r[-1], 1e3)
    rho_new = isa(r_isa - obj.Re, 0)[1]  # new measured values
    rho_old = rho_newmodel(new_alt-obj.Re)
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
    rho_newmodel = PchipInterpolator(r_comp_i-obj.Re, rho_newdata_i)

    plt.figure(1)
    plt.plot(alt, rho_newmodel(alt), label="Model {}".format(n))
    plt.legend(loc='best')

    test = solve_ivp(sys_rho, [init_t, tfin], xini, t_eval=tt)

    plt.figure(2)
    plt.plot(test.t, test.y[0, :], label="Model {}".format(n))
    plt.legend(loc='best')
    print(t[-1], tfin)
    n += 1



plt.show(block=True)




