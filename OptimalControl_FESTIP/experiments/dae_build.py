from casadi import *
import numpy as np

# Example on how to use the DaeBuilder class
# Joel Andersson, UW Madison 2017

# Start with an empty DaeBuilder instance
dae = DaeBuilder()


isa = external('isa', '')


def thrust(presamb, mass, presv, spimpv, delta, tau, npoint, slpres, wlo, we, lref, xcgf, xcg0):
    nimp = 17
    nmot = 1
    Thrust = []
    Deps = []
    Simp = []
    Mom = []
    # thrmax = nmot * (5.8E+6 + 14.89 * slpres - 11.16 * presamb)
    for j in range(npoint):
        thrx = nmot * (5.8e6 + 14.89 * slpres - 11.16 * presamb[j]) * delta[j]
        if presamb[j] >= slpres:
            spimp = spimpv[-1]
        elif presamb[j] < slpres:
            for i in range(nimp):
                if presv[i] >= presamb[j]:
                    spimp = np.interp(presamb[j], [presv[i - 1], presv[i]], [spimpv[i - 1], spimpv[i]])
                    break
        xcg = ((xcgf - xcg0) / (we - wlo) * (mass[j] - wlo) + xcg0) * lref

        dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * slpres - presamb[j])

        mommot = tau[j] * dthr

        thrz = -tau[j] * (2.5e6 - 22 * slpres + 9.92 * presamb[j])
        thrust = np.sqrt(thrx ** 2 + thrz ** 2)
        deps = np.arctan(thrz / thrx)
        Thrust.append(thrust)
        Deps.append(deps)
        Simp.append(spimp)
        Mom.append(mommot)
    return Thrust, Deps, Simp, Mom

# Constants
GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
Re = 6371000  # Earth Radius [m]
psl = 101325  # ambient pressure at sea level [Pa]
latstart = np.deg2rad(5.2) # deg latitude
longstart = np.deg2rad(-52.775)  # deg longitude
chistart = np.deg2rad(113)  # deg flight direction
incl = np.deg2rad(51.6)  # deg orbit inclination
gammastart = np.deg2rad(89.9)  # deg
M0 = 450400  # kg  starting mass
g0 = 9.80665  # m/s2
gIsp = g0 * 455 # g0 * Isp max
omega = 7.2921159e-5
MaxQ = 40000  # Pa
MaxAx = 30  # m/s2
MaxAz = 15  # m/s2
Htarget = 400000  # m target height after hohmann transfer
wingSurf = 500.0  # m2
lRef = 34.0  # m
k = 5000  # [Nm] livello di precisione per trimmaggio
m10 = M0 * 0.1
xcgf = 0.37  # cg position with empty vehicle
xcg0 = 0.65  # cg position at take-off
pref = 21.25
Hini = 180000
r2 = Re + Htarget
Rtarget = Re + Hini  # m/s
Vtarget = np.sqrt(GMe / Rtarget)  # m/s forse da modificare con velocita' assoluta
chi_fin = 0.5 * np.pi + np.arcsin(np.cos(incl) / np.cos(latstart))


# Add input expressions
# states
v = dae.add_x('v')
chi = dae.add_x('chi')
gamma = dae.add_x('gamma')
teta = dae.add_x('teta')
lam = dae.add_x('lam')
h = dae.add_x('h')
m = dae.add_x('m')

# controls
alfa = dae.add_u('alfa')
delta = dae.add_u('delta')
deltaf = dae.add_u('deltaf')
tau = dae.add_u('tau')

# algebric parameters
T = dae.add_z('T')
L = dae.add_z('L')
D = dae.add_z('D')
isp = dae.add_z('Isp')
g = dae.add_z('g')

# algebric equations
Press, rho, c = dae.add_alg(isa(h[:], psl, g0, Re))
T, Deps, isp, MomT = dae.add_alg(thrust(Press, m, presv, spimpv, delta, tau, n, psl, M0, m10, lRef, xcgf, xcg0))
eps = dae.add_alg(Deps + alfa)




# Add output expressions
vdot = ((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (omega**2) * (Re + h) * np.cos(lam) * \
            (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi))

chidot = ((T * np.sin(eps) + L) / (m * v * np.cos(gamma))) - np.cos(gamma) * np.cos(chi) * np.tan(lam) \
            * (v / (Re + h)) + 2 * omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam))\
            - (omega ** 2) * ((Re + h)/(v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi)

gamdot = ((T * np.sin(eps) + L) / (m * v)) - (g / v - v / (Re + h)) * np.cos(gamma) + 2 * omega \
            * np.cos(lam) * np.cos(chi) + (omega ** 2) * ((Re + h) / v) * np.cos(lam) * \
            (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma))

tetadot = -np.cos(gamma) * np.cos(chi) * (v / ((Re + h) * np.cos(lam)))
lamdot = np.cos(gamma) * np.sin(chi) * (v / (Re + h))
hdot = v * np.sin(gamma)
mdot = -T / (g0 * isp)
dae.add_ode('vdot', vdot)
dae.add_ode('chidot', chidot)
dae.add_ode('gamdot', gamdot)
dae.add_ode('tetadot', tetadot)
dae.add_ode('lamdot', lamdot)
dae.add_ode('hdot', hdot)
dae.add_ode('mdot', mdot)

# Specify initial conditions
dae.set_start('v', 0.01)
dae.set_start('chi', chistart)
dae.set_start('gamma', gammastart)
dae.set_start('teta', longstart)
dae.set_start('lam', latstart)
dae.set_start('h', 0.01)
dae.set_start('m', M0)

# Add meta information
#dae.set_unit('h','m')
#dae.set_unit('v','m/s')
#dae.set_unit('m','kg')

# Print DAE

#dae.make_explicit()
dae.disp(True)