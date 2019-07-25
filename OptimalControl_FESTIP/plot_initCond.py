import matplotlib.pyplot as plt
from OpenGoddard.optimize import Guess
import numpy as np


latstart = 5.2  # deg latitude
longstart = -52.775  # deg longitude
chistart = 113  # deg flight direction
incl = np.deg2rad(51.6)  # deg orbit inclination
gammastart = 89  # deg
M0 = 450400  # kg  starting mass
Htarget = 400000  # m target height after hohmann transfer
m10 = M0 * 0.1
Hini = 100000
GMe = 3.986004418e14
Re = 6371000
Rtarget = Re + Hini
Vtarget = np.sqrt(GMe / Rtarget)  # m/s forse da modificare con velocita' assoluta
chi_fin = np.rad2deg(0.5 * np.pi + np.arcsin(np.cos(incl) / np.cos(np.deg2rad(latstart))))
time_tot = 350
tnew = np.linspace(0, time_tot, 200)

v_init = Guess.cubic(tnew, 1.0, 0.0, Vtarget, 0.0)
chi_init = Guess.linear(tnew, chistart, chi_fin)
gamma_init = Guess.linear(tnew, gammastart, 0.0)
teta_init = Guess.constant(tnew, longstart)
lam_init = Guess.constant(tnew, latstart)
h_init = Guess.cubic(tnew, 1.0, 0.0, Hini, 0.0)
m_init = Guess.cubic(tnew, M0, 0.0, m10, 0.0)
alfa_init = Guess.constant(tnew, 0.0)
delta_init = Guess.cubic(tnew, 1.0, 0.0, 0.001, 0.0)
deltaf_init = Guess.constant(tnew, 0.0)
tau_init = Guess.constant(tnew, 0.0)
mu_init = Guess.constant(tnew, 0.0)


plt.figure(0)
plt.title("Velocity")
plt.plot(tnew, v_init)
plt.grid()
plt.ylabel("m/s")
plt.xlabel("time [s]")

plt.figure(1)
plt.title("Flight path angle \u03C7")
plt.plot(tnew, chi_init)
plt.grid()
plt.ylabel("deg")
plt.xlabel("time [s]")

plt.figure(2)
plt.title("Angle of climb \u03B3")
plt.plot(tnew, gamma_init)
plt.grid()
plt.ylabel("deg")
plt.xlabel("time [s]")

plt.figure(3)
plt.title("Longitude \u03B8")
plt.plot(tnew, teta_init)
plt.grid()
plt.ylabel("deg")
plt.xlabel("time [s]")

plt.figure(4)
plt.title("Latitude \u03BB")
plt.plot(tnew, lam_init)
plt.grid()
plt.ylabel("deg")
plt.xlabel("time [s]")

plt.figure(6)
plt.title("Altitude")
plt.plot(tnew, h_init / 1000)
plt.grid()
plt.ylabel("km")
plt.xlabel("time [s]")

plt.figure(7)
plt.title("Mass")
plt.plot(tnew, m_init)
plt.grid()
plt.ylabel("kg")
plt.xlabel("time [s]")

plt.figure(8)
plt.title("Angle of attack \u03B1")
plt.plot(tnew, alfa_init)
plt.grid()
plt.ylabel("deg")
plt.xlabel("time [s]")

plt.figure(9)
plt.title("Throttles")
plt.plot(tnew, delta_init * 100, color='r')
plt.plot(tnew, tau_init * 100, color='k')
plt.grid()
plt.ylabel("%")
plt.xlabel("time [s]")


plt.figure(10)
plt.title("Body Flap deflection \u03B4")
plt.plot(tnew, deltaf_init)
plt.grid()
plt.ylabel("deg")
plt.xlabel("time [s]")
plt.legend(["Control points"], loc="best")

plt.figure(11)
plt.title("Bank angle profile \u03BC")
plt.plot(tnew, mu_init)
plt.grid()
plt.ylabel("deg")
plt.xlabel("time [s]")

plt.show()
plt.close()