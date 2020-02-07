import matplotlib.pyplot as plt
from OpenGoddard.optimize import Guess
from scipy import interpolate, optimize
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import Bounds
from models import *
import time
import datetime
import os
'''old TestMS2.py'''
'''modified initial guess for controls so it has more points than states. States points are only in the conjunction of legs while controls are also inside'''
'''simplified version of problem. All controls but equatorial trajectory and no earth rotation'''
'''definition of vehicle and mission parameters. Angles are set in degrees'''
'''maybe best to set initial values of angles in radians'''

'''DIFFERENT METHOD FOR INTEGRATION'''

'''set initial conditions constraints on all parameters'''

start = time.time()
timestr = time.strftime("%Y%m%d-%H%M%S")
savefig_file = "MultiShooting_{}_{}_".format(os.path.basename(__file__), timestr)

'''vehicle parameters'''


class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
        self.chistart = np.deg2rad(113)  # deg flight direction
        self.incl = np.deg2rad(51.6)  # deg orbit inclination
        self.gammastart = np.deg2rad(89.9)  # deg
        self.M0 = 450400  # kg  starting mass
        self.g0 = 9.80665  # m/s2
        self.gIsp = self.g0 * 455  # g0 * Isp max
        self.omega = 7.2921159e-5
        self.MaxQ = 40000  # Pa
        self.MaxAx = 30  # m/s2
        self.MaxAz = 15  # m/s2
        self.Htarget = 400000  # m target height after hohmann transfer
        self.wingSurf = 500.0  # m2
        self.lRef = 34.0  # m
        self.k = 5e3  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 100000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))


obj = Spaceplane()

'''reading of aerodynamic coefficients and specific impulse from file'''

cl = fileReadOr("clfile.txt")
cd = fileReadOr("cdfile.txt")
cm = fileReadOr("cmfile.txt")
cl = np.asarray(cl)
cd = np.asarray(cd)
cm = np.asarray(cm)

with open("impulse.dat") as f:
    impulse = []
    for line in f:
        line = line.split()
        if line:
            line = [float(i) for i in line]
            impulse.append(line)

f.close()

presv = []
spimpv = []

for i in range(len(impulse)):
    presv.append(impulse[i][0])
    spimpv.append(impulse[i][1])

presv = np.asarray(presv)
spimpv = np.asarray(spimpv)

'''set problem parameters'''

time_tot = 350  # initial time
Nbar = 5 # number of conjunction points
Nleg = Nbar - 1  # number of multiple shooting sub intervals
NContPoints = 11  # number of control points for interpolation inside each interval
Nint = 500 # number of points for each single shooting integration
maxiter = 20  # max number of iterations for nlp solver
ftol = 1e-12  # numeric tolerance of nlp solver
eps = 1e-9  # increment of the derivative?
Nstates = 7  # number of states
Ncontrols = 4  # number of controls
maxIterator = 10  # max number of optimization iterations
varStates = Nstates * Nleg  # total number of optimization variables for states
varControls = Ncontrols * (Nleg * NContPoints - Nbar + 2)   # total number of optimization variables for controls
varTot = varStates + varControls  # total number of optimization variables for states and controls
NineqCond = Nint # Nleg * NContPoints - Nbar + 2
tnew = np.linspace(0, time_tot, Nbar)  # time vector used for interpolation of states initial guess
tcontr = np.linspace(0, time_tot, int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess

'''definition of values used for scaling'''

unit_v = 9000*100
unit_chi = np.deg2rad(270)
unit_gamma = np.deg2rad(90)
unit_teta = np.deg2rad(90)
unit_lam = obj.incl
unit_h = obj.Hini*100
unit_m = obj.M0
unit_alfa = np.deg2rad(40)
unit_delta = 1
unit_deltaf = np.deg2rad(30)
unit_tau = 1
unit_t = 2000
states_unit = np.array((unit_v, unit_chi, unit_gamma, unit_teta, unit_lam, unit_h, unit_m))
controls_unit = np.array((unit_alfa, unit_delta, unit_deltaf, unit_tau))

'''definiton of initial conditions'''

# set vector of initial conditions of states and controls
X = np.zeros((0))
U = np.zeros((0))

# set vector of initial guesses for state and controls in all intervals taken from OpenGoddard
'''
Nguess = 30  # number of points for initial guess from OpenGoddard
tsec = np.linspace(0, time_tot, Nguess) # time vector used to interpolate data from OpenGoddard optimization (10 points)

v_guess = [1.0e-09, 8.42365240e+00, 3.02780338e+01, 6.70071187e+01, 1.48836123e+02, 2.40170803e+02, 3.18666373e+02,
           4.57624105e+02, 6.98401714e+02, 1.13201602e+03, 1.71953780e+03, 2.40766415e+03, 3.12826061e+03,
           3.89947130e+03, 4.67543002e+03, 5.48235411e+03, 6.26271635e+03, 7.02503542e+03, 7.33743041e+03,
           7.29309118e+03, 7.31458573e+03, 7.29364241e+03, 7.30740197e+03, 7.29359174e+03, 7.30648182e+03,
           7.29438240e+03, 7.30894806e+03, 7.29263195e+03, 7.33675411e+03, 7.36116125e+03]
chi_guess = np.deg2rad([113.0, 117.30667129, 124.0298524, 126.33624543, 126.19178325, 126.36287331, 126.53099384,
                        126.80720038,126.93345433, 127.04606505, 127.09787061, 127.15252937, 127.18077606, 127.21684868,
                        127.24535739, 127.29008228, 127.34400289, 127.42411093, 127.52309828, 127.63934945,
                        127.76207244, 127.89213559, 128.02138782, 128.14822772, 128.26572351, 128.37113463, 128.4590911,
                        128.5269148, 128.5713432, 128.5907579])
gamma_guess = np.deg2rad([8.99e+01, 8.97732513e+01, 8.94996702e+01, 6.54943445e+01, 3.78044052e+01, 5.33573510e+01,
                          6.78705457e+01, 5.62198702e+01, 4.03498301e+01, 2.69545622e+01, 1.82742948e+01,
                          1.25515378e+01, 8.73630527e+00, 6.04579115e+00, 4.35248485e+00, 3.27152337e+00,
                          2.69879074e+00, 2.47439232e+00, 2.28806770e+00, 1.90439338e+00, 1.59769600e+00,
                          1.27268032e+00, 1.00435916e+00, 7.41809601e-01, 5.30562949e-01, 3.38585731e-01,
                          1.96478781e-01, 8.05183649e-02, 2.23195860e-02, -1.62466060e-26])
teta_guess = np.deg2rad([-52.768, -52.76799646, -52.76801126, -52.76757814, -52.76356382, -52.75355109, -52.74325584,
                         -52.72641137, -52.68539222, -52.59631748, -52.43047445, -52.16527029, -51.78764624,
                         -51.29026369, -50.6704522, -49.93068787, -49.07842855, -48.12693918, -47.11876153,
                         -46.13663476, -45.20390655, -44.32993438, -43.52539154, -42.79997448, -42.16307383,
                         -41.62321101, -41.18790068, -40.86345843, -40.65452219, -40.56459687])
lam_guess = np.deg2rad([-3.10998691e-10, 4.05603492e-06, -1.21105065e-05, 5.69328922e-04, 6.06165768e-03,
                        1.96940620e-02, 3.36506629e-02, 5.62343607e-02, 1.10917154e-01, 2.29102751e-01,
                        4.48589050e-01, 7.98861351e-01, 1.29685774e+00, 1.95188203e+00, 2.76682123e+00, 3.73749376e+00,
                        4.85241021e+00, 6.09201631e+00, 7.39806214e+00, 8.66179921e+00, 9.85286069e+00, 1.09600158e+01,
                        1.19708011e+01, 1.28748265e+01, 1.36623725e+01, 1.43252498e+01, 1.48564679e+01, 1.52504566e+01,
                        1.55032746e+01, 1.56118694e+01])
h_guess = [1.0e-09, 1.02594800e+01, 1.01669017e+02, 4.79410476e+02, 1.24901429e+03, 3.07738808e+03, 6.95249072e+03,
           1.30182532e+04, 2.13732011e+04, 3.19296457e+04, 4.43565755e+04, 5.75591488e+04, 7.05099709e+04,
           8.22592852e+04, 9.25842704e+04, 1.01604660e+05, 1.09760804e+05, 1.17659165e+05, 1.25473403e+05,
           1.32111194e+05, 1.37324429e+05, 1.41325415e+05, 1.44212725e+05, 1.46210830e+05, 1.47470715e+05,
           1.48204477e+05, 1.48561255e+05, 1.48702040e+05, 1.48732637e+05, 1.48736155e+05]
m_guess = [450400.0, 446501.88114705, 437341.04636429, 423150.98828608, 403787.52237251, 381131.1410853,
           355680.85186168, 324359.52686723, 290142.3120519, 253897.56451206, 216969.90474864, 183247.87420845,
           154009.95119844, 128717.65590982, 107413.25548964, 89365.09741087, 74656.60552944, 62817.08425109,
           58388.25559038, 58855.89169975, 58489.4807071, 58691.21886462, 58464.06837688, 58607.52813238,
           58418.26363457, 58561.50880645, 58365.43363299, 58574.25132824, 57999.5035841, 57681.80932478]
alfa_guess = np.deg2rad([0.07124792, 0.43865414, -2.0, -1.70652002, 8.09111491, 9.99999976, 1.99999967, -0.57524248,
                         -0.93728157, -0.99126429, 0.96934777, 1.99999949, 2.99944644, 3.88791914, 4.66423286,
                         4.74730228, 4.52390215, 4.5405197, 3.83157713, 4.69750322, 3.64059788, 4.95615688, 3.4412657,
                         5.15899686, 3.14604604, 5.19774418, 2.63747854, 4.586136, 1.86106885, 1.04934187])
delta_guess = [1.0, 1.0, 1.0, 1.00000000e+00, 1.00000000e+00, 8.94621339e-01, 9.35205020e-01,
               1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 9.05821691e-01, 7.57605237e-01, 6.33808712e-01,
               5.28705865e-01, 4.40986763e-01, 3.66839807e-01, 3.06450306e-01, 2.10669818e-01, 7.89407797e-03,
               1.00000000e-03, 1.00000000e-03, 1.00000000e-03, 1.00000000e-03, 1.00000000e-03, 1.00000000e-03,
               1.00000000e-03, 1.00000000e-03, 1.00000000e-03, 1.28617617e-01, 1.00000000e-03]
deltaf_guess = np.deg2rad([2.70368530e-06, -4.53196198e-02, -1.06303860e+01, -2.58958601e-07, -5.02212003e-01,
                           -9.94730080e+00, -3.04974911e-07, -9.81936573e-07, 1.39735226e-07, -3.76683790e-10,
                           -9.04738283e-07, -2.00000000e+01, -2.00000000e+01, -1.76780576e+01, -7.62673659e+00,
                           -4.54796010e-01, -1.70858367e-01, -7.97549606e-02, -2.60267400e-02, -1.41015790e-02,
                           -5.68894823e-03, -6.11867150e-03, -2.71903095e-03, -3.39230242e-03, -1.04887512e-03,
                           -1.66914155e-03, -2.74874582e-04, -3.37624279e-04, -5.29526647e-05, -1.65563466e-10])
tau_guess = [0.00000000e+00, 9.26321485e-18, -3.11614679e-18, 1.00301731e-02, -3.14268401e-19, -1.56065307e-18,
             3.88151264e-01, 5.01879303e-01, 3.15093422e-01, 1.56525964e-01, 1.67431405e-01, 2.03078671e-16,
             -8.04829119e-18, 3.12206596e-10, 1.16172499e-06, 1.39546659e-06, 1.69069378e-06, 9.21939729e-05,
             5.34539741e-05, 5.26179915e-18, 1.28379075e-08, 1.35049516e-17, 9.76933853e-18, 1.54831365e-17,
             9.43658685e-18, 7.77294698e-18, 1.19511253e-17, 1.02343101e-19, 7.70659449e-05, -2.81734426e-19]

# interpolation over OpenGoddard data to use more than 10 points

v_interpOG = interpolate.interp1d(tsec, v_guess)
chi_interpOG = interpolate.interp1d(tsec, chi_guess)
gamma_interpOG = interpolate.interp1d(tsec, gamma_guess)
teta_interpOG = interpolate.interp1d(tsec, teta_guess)
lam_interpOG = interpolate.interp1d(tsec, lam_guess)
h_interpOG = interpolate.interp1d(tsec, h_guess)
m_interpOG = interpolate.interp1d(tsec, m_guess)
alfa_interpOG = interpolate.interp1d(tsec, alfa_guess)
delta_interpOG = interpolate.interp1d(tsec, delta_guess)
deltaf_interpOG = interpolate.interp1d(tsec, deltaf_guess)
tau_interpOG = interpolate.interp1d(tsec, tau_guess)

# creation of vectors for states and controls of initial guesses from open goddard

XGuessOG = np.array((v_interpOG(tnew) / unit_v, chi_interpOG(tnew) / unit_chi, gamma_interpOG(tnew) / unit_gamma,
                     teta_interpOG(tnew) / unit_teta, lam_interpOG(tnew) / unit_lam, h_interpOG(tnew) / unit_h,
                     m_interpOG(tnew) / unit_m))

UGuessOG = np.array((alfa_interpOG(tcontr) / unit_alfa, delta_interpOG(tcontr) / unit_delta, deltaf_interpOG(tcontr) / unit_deltaf,
     tau_interpOG(tcontr) / unit_tau))  # controls initial guesses from OpenGoddard data
'''
'''definition of another set of initial guesses, more random but reasonable'''

v_init = Guess.cubic(tnew, 1, 0.0, obj.Vtarget, 0.0)
chi_init = Guess.cubic(tnew, obj.chistart, 0.0, obj.chi_fin, 0.0)
gamma_init = Guess.linear(tnew, obj.gammastart, 0.0)
teta_init = Guess.constant(tnew, obj.longstart+np.deg2rad(90))
lam_init = Guess.constant(tnew, (obj.latstart+obj.incl)/2)
h_init = Guess.cubic(tnew, 1, 0.0, obj.Hini, 0.0)
m_init = Guess.cubic(tnew, obj.M0, 0.0, obj.m10, 0.0)
alfa_init = Guess.zeros(tcontr)
delta_init = Guess.cubic(tcontr, 1.0, 0.0, 0.001, 0.0)
deltaf_init = Guess.zeros(tcontr)
tau_init = Guess.constant(tcontr, 0.5)

XGuess = np.array(
    (v_init / unit_v, chi_init / unit_chi, gamma_init / unit_gamma, teta_init / unit_teta, lam_init / unit_lam,
     h_init / unit_h, m_init / unit_m))  # states initial guesses

UGuess = np.array((alfa_init / unit_alfa, delta_init / unit_delta, deltaf_init / unit_deltaf,
                   tau_init / unit_tau))  # states initial guesses

for i in range(Nleg):
    '''creation of vector of states initial guesses'''
    for j in range(Nstates):
        X = np.hstack((X, XGuess[j][i]))

for i in range(int(varControls / Ncontrols)):
    '''creation of vector of controls initial guesses'''
    for j in range(Ncontrols):
        U = np.hstack((U, UGuess[j][i]))

dt = np.zeros((0))
for i in range(len(tnew) - 1):
    '''creation of vector of time intervals'''
    dt = np.hstack((dt, tnew[i + 1] - tnew[i]))

X0 = np.hstack((X, U, dt / unit_t))  # vector of initial conditions here all the angles are in degrees!!!!!

# X0 has first all X guesses and then all U guesses
# at this point the vectors of initial guesses for states and controls for every time interval are defined


'''set upper and lower bounds for states, controls and time, scaled'''
'''major issues with bounds!!!'''
#bndX = ((1e-9 / unit_v, 1.0), (np.deg2rad(90) / unit_chi, np.deg2rad(270) / unit_chi),
 #       (0.0, np.deg2rad(90.0) / unit_gamma), (None, None),
  #      (-obj.incl/unit_lam, obj.incl / unit_lam),
   #     (1e-9 / unit_h, obj.Hini / unit_h), (obj.m10 / unit_m, obj.M0 / unit_m))
#bndU = ((np.deg2rad(-2) / unit_alfa, np.deg2rad(40) / unit_alfa), (0.001, 1.0),
 #       (np.deg2rad(-20) / unit_deltaf, np.deg2rad(30) / unit_deltaf), (0.0, 1.0))
#bndT = ((5/unit_t, time_tot/unit_t),)

#bnds = (bndX) * Nleg + (bndU) * (Nleg * NContPoints - Nbar + 2)  + (bndT) * Nleg


Xlb = ([1e-5/unit_v, np.deg2rad(90)/unit_chi, 0.0, 0.0, 0.0, 1e-5/unit_h, obj.m10/unit_m]) # states lower bounds
Xub = ([20000/unit_v, np.deg2rad(270)/unit_chi, np.deg2rad(89.9)/unit_gamma, np.deg2rad(90)/unit_teta, obj.incl/unit_lam, 200000/unit_h, obj.M0/unit_m]) # states upper bounds

Ulb = ([np.deg2rad(-2)/unit_alfa, 0.001/unit_delta, np.deg2rad(-20)/unit_deltaf, 0.0/unit_tau]) # controls lower bounds
Uub = ([np.deg2rad(40)/unit_alfa, 1.0/unit_delta, np.deg2rad(30)/unit_deltaf, 1.0/unit_tau]) # controls upper bounds

Tlb = ([1/unit_t,]) # time lower bounds
Tub = ([time_tot/unit_t,]) # time upper bounds

lb = Xlb*Nleg + Ulb *(Nleg * NContPoints - Nbar + 2) + Tlb*Nleg
ub = Xub*Nleg + Uub *(Nleg * NContPoints - Nbar + 2) + Tub*Nleg
bnds = Bounds(lb, ub)

'''definition of initial and final conditions for states and controls for equality constraints'''

#init_condX = np.array((1 / unit_v, obj.chistart / unit_chi, obj.gammastart / unit_gamma, obj.longstart / unit_teta,
 #                      obj.latstart / unit_lam, 1 / unit_h, obj.M0 / unit_m))
#init_condU = np.array((0.0, 1.0, 0.0, 0.0))
#final_cond = np.array((0.0))  # final conditions on gamma

'''function definitions'''

def dynamicsInt(t, states, alfa_Int, delta_Int, deltaf_Int, tau_Int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3] - np.deg2rad(90)
    lam = states[4] * 2 - obj.incl
    h = states[5]
    m = states[6]
    alfa = alfa_Int(t)
    delta = delta_Int(t)
    deltaf = deltaf_Int(t)
    tau = tau_Int(t) * 2 - 1


    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)
    # Press = np.asarray(Press, dtype=np.float64)
    # rho = np.asarray(rho, dtype=np.float64)
    # c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)
    # L = np.asarray(L, dtype=np.float64)
    # D = np.asarray(D, dtype=np.float64)
    # MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

    # T = np.asarray(T, dtype=np.float64)
    # isp = np.asarray(isp, dtype=np.float64)
    # Deps = np.asarray(Deps, dtype=np.float64)
    # MomT = np.asarray(MomT, dtype=np.float64)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
                   ((T * np.sin(eps) + L) / (m * v)) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (g0 * isp)))

    return dx


def dynamicsVel(states, contr):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''

    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3] - np.deg2rad(90)
    lam = states[4] * 2 - obj.incl
    h = states[5]
    m = states[6]
    alfa = contr[0]
    delta = contr[1]
    deltaf = contr[2]
    tau = contr[3] * 2 - 1

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)
    # Press = np.asarray(Press, dtype=np.float64)
    # rho = np.asarray(rho, dtype=np.float64)
    # c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)
    # L = np.asarray(L, dtype=np.float64)
    # D = np.asarray(D, dtype=np.float64)
    # MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

    # T = np.asarray(T, dtype=np.float64)
    # isp = np.asarray(isp, dtype=np.float64)
    # Deps = np.asarray(Deps, dtype=np.float64)
    # MomT = np.asarray(MomT, dtype=np.float64)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
                   ((T * np.sin(eps) + L) / (m * v)) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (g0 * isp)))

    return dx


def inequalityAll(states, controls, rep):
    '''this function takes states and controls unscaled'''
    v = states[0, :]
    chi = states[1, :]  # it's not used for functions, can be used in degree
    gamma = states[2, :]  # it's not used for functions, can be used in degree
    teta = states[3, :] - np.deg2rad(90)# it's not used for functions, can be used in degree
    lam = states[4, :] * 2 - obj.incl  # it's not used for functions, can be used in degree
    h = states[5, :]
    m = states[6, :]
    alfa = controls[0, :]
    delta = controls[1, :]
    deltaf = controls[2, :]
    tau = controls[3, :] * 2 - 1

    # print("v ", v)
    # print("chi ", chi)
    # print("gamma ", gamma)
    # print("teta ", teta)
    # print("lam ", lam)
    # print("h ", h)
    # print("m ", m)
    # print("alfa ", alfa)
    # print("delta ", delta)
    # print("deltaf ", deltaf)
    # print("tau ", tau)

    Press, rho, c = isaMulti(h, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    M = v / c

    L, D, MomA = aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                 obj.xcg0, obj.xcgf, obj.pref, rep)

    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrustMulti(Press, m, presv, spimpv, delta, tau, rep, obj.psl, obj.M0, obj.m10, obj.lRef,
                                     obj.xcgf, obj.xcg0)

    T = np.asarray(T, dtype=np.float64)
    # isp = np.asarray(isp, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)
    MomT = np.asarray(MomT, dtype=np.float64)

    MomTot = MomA + MomT
    MomTotA = abs(MomTot)

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)

    # accelerations
    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    return np.hstack(((obj.MaxAx - ax)/(obj.MaxAx), (obj.MaxAz - az)/(obj.MaxAz), (obj.MaxQ - q)/(obj.MaxQ),
                      (obj.k - MomTotA)/(obj.k*1e4), (h[-1] - 80000)/unit_h))  # when with bounds

    # return np.hstack(((v - 1e-9) / unit_v, (obj.Vtarget - v) / unit_v, (chi - np.deg2rad(90)) / unit_chi,
    #                 (np.deg2rad(270) - chi) / unit_chi, gamma / unit_gamma, (np.deg2rad(89.9) - gamma) / unit_gamma,
    #                (lam + obj.incl) / unit_lam, (obj.incl - lam) / unit_lam, np.deg2rad(-90)/unit_teta, np.deg2rad(90)/unit_teta, (h - 1e-9) / unit_h,
    #               (obj.Hini - h) / unit_h, (m - obj.m10) / unit_m,
    #              (obj.M0 - m) / unit_m, (delta - 0.001), (1.0 - delta), (alfa + np.deg2rad(2)) / unit_alfa,
    #             (np.deg2rad(40) - alfa) / unit_alfa,
    #            (deltaf + np.deg2rad(20)) / unit_deltaf, (np.deg2rad(30) - deltaf) / unit_deltaf, tau, (1 - tau),
    #           (obj.MaxAx - ax) / obj.MaxAx, (obj.MaxAz - az) / obj.MaxAz, (obj.MaxQ - q) / obj.MaxQ,
    #          (obj.k - MomTotA) / obj.k))


def ineqCond(var):
    '''this functions repeats the inequality conditions for every time step'''
    #print("ineq cond")
    # print("Inside IneqCond")
    # h = var[varStates - 2]
    # m = var[varStates - 1] * states_unit[6]

    # r1 = h + obj.Re
    # Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    # Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    # mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))

    iC = MultiShootIneq(var, dynamicsInt)
    # iC = np.concatenate((iC, (h - 80000 / unit_h,)))
    # ineq_cond = np.hstack((ineq_cond, var[varTot:-1]))
    return iC


def SingleShooting(states, controls, dyn, tstart, tfin, Nint):
    '''this function integrates the dynamics equation over time.'''
    '''INPUT: states: states vector
              controls: controls matrix
              dyn: dynamic equations
              tstart: initial time
              tfin: final time
              Nint: unmber of integration steps'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''in this function the states are not scaled anymore'''
    '''tstart and tfin are the initial and final time of the considered leg'''
    #print("single shooting")
    Time = np.linspace(tstart, tfin, NContPoints)

    x = np.zeros((Nint, Nstates))
    x[0,:]= states[0:Nstates] * states_unit  # vector of intial states ready

    #print("Single: ", Time)
    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(Time, controls[0, :] * unit_alfa)
    delta_Int = interpolate.PchipInterpolator(Time, controls[1, :] * unit_delta)
    deltaf_Int = interpolate.PchipInterpolator(Time, controls[2, :] * unit_deltaf)
    tau_Int = interpolate.PchipInterpolator(Time, controls[3, :] * unit_tau * 2 - 1)

    #VAR = np.concatenate((states[0:Nstates] * states_unit, controls[0, :] * unit_alfa, controls[1, :], controls[2, :] * unit_deltaf, controls[3, :], Time))

    time_new = np.linspace(tstart, tfin, Nint)


    #plt.figure()
    # plt.plot(time_new, np.rad2deg(alfa_interp(time_new)))
    # plt.figure()
    #plt.plot(time_new, delta_Int(time_new), 'x')
    #plt.plot(time_new, delta_Int2(time_new), '.')
    #plt.figure()
    # plt.plot(time_new, np.rad2deg(deltaf_interp(time_new)))
    # plt.figure()
    # plt.plot(time_new, tau_interp(time_new))
    #plt.show()
    # alfa_u = np.ones((np.shape(time_new)))*unit_alfa
    # deltaf_u = np.ones((np.shape(time_new))) * unit_deltaf
    # u[:, 0] = alfa_interp(time_new)
    # u[:, 1] = delta_interp(time_new)
    # u[:, 2] = deltaf_interp(time_new)
    # u[:, 3] = tau_interp(time_new)

    dt = (time_new[1] - time_new[0])

    t = time_new
    #print(time_new)
    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int), t_span=[tstart, tfin], y0=x, t_eval=time_new, method='RK45')


    #print(sol)


    for i in range(Nint-1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt*dyn(t[i], x[i, :], alfa_Int, delta_Int, deltaf_Int, tau_Int)
        # print("k1: ", k1)
        k2 = dt*dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int)
        # print("k2: ", k2)
        k3 = dt*dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int)
        # print("k3: ", k3)
        k4 = dt*dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    #vres = sol.y[0, :]
    #chires = sol.y[1, :]
    #gammares = sol.y[2, :]
    #tetares = sol.y[3, :]
    #lamres = sol.y[4, :]
    #hres = sol.y[5, :]
    #mres = sol.y[6, :]

    vres = x[:, 0]
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]

    alfares = alfa_Int(time_new)
    deltares = delta_Int(time_new)
    deltafres = deltaf_Int(time_new)
    taures = (tau_Int(time_new) + 1 )/2

    return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, \
           alfa_Int, delta_Int, deltaf_Int, tau_Int


def MultiShooting(var, dyn):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in degrees'''
    #print("multi shooting")
    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    deltaf = np.zeros((NContPoints))
    tau = np.zeros((NContPoints))
    states_atNode = np.zeros((0))
    timestart = 0
    for i in range(Nleg):
        # print("Multiple shooting Leg: ", i)
        states = var[i * Nstates:(i + 1) * Nstates]
        #print(i, range(Nleg))
        if i == 0.0:
            timestart = 0.0
        timeend = timestart + var[varTot+i] * unit_t
        #print("var: ", var[varTot:])
        #print("used: ", var[varTot+i])
        #print("start: ", timestart)
        #print("end: ", timeend)
        for k in range(NContPoints):
            alfa[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + Ncontrols * k]
            delta[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 1 + Ncontrols * k]
            deltaf[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 2 + Ncontrols * k]
            tau[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 3 + Ncontrols * k]
        controls = np.vstack((alfa, delta, deltaf, tau))
        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, alfa_I, delta_I, deltaf_I, tau_I = SingleShooting(
            states, controls, dyn, timestart, timeend, Nint)


        # print(i, vres[-1], chires[-1], gammares[-1], lamres[-1], tetares[-1], hres[-1], mres[-1])
        # print(i, alfares[-1], deltares[-1], deltafres[-1], taures[-1])
        states_atNode = np.concatenate((states_atNode, (
                    (vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]) / states_unit)))
        timestart = timeend
        # if i == (Nleg-1):
        #   var[varStates-Nstates:varStates] = np.array(((vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]) / states_unit))
    return states_atNode


def MultiShootIneq(var, dyn):
    '''in this function the states and controls are scaled'''
    #print("multi shooting ineq")
    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    deltaf = np.zeros((NContPoints))
    tau = np.zeros((NContPoints))
    #states_atNode = np.zeros((0))
    #tres = np.zeros((0))
    #vres = np.zeros((0))
    #chires = np.zeros((0))
    #gammares = np.zeros((0))
    #lamres = np.zeros((0))
    #tetares = np.zeros((0))
    #hres = np.zeros((0))
    #mres = np.zeros((0))
    #time = np.zeros((1))
    timestart = 0
    ineq_cond = np.zeros((0))
    for i in range(Nleg):
        controls = np.zeros((NContPoints,))
        # print("Multiple shooting Leg: ", i)
        states = var[i * Nstates:(i + 1) * Nstates]

        if i == 0:
            timestart = 0
        timeend = timestart + var[i + varTot] * unit_t

        for k in range(NContPoints):
            alfa[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + Ncontrols * k]
            delta[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 1 + Ncontrols * k]
            deltaf[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 2 + Ncontrols * k]
            tau[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 3 + Ncontrols * k]
        controls = np.vstack((alfa, delta, deltaf, tau))

        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, alfainterp, deltainterp, deltafinterp, tauinterp = SingleShooting(
            states, controls, dyn, timestart, timeend, Nint)

        #time_ineq = np.linspace(tres[0], tres[-1], NineqCond)
        #print(np.shape(tres), np.shape(vres))
        #vinterp = interpolate.PchipInterpolator(tres, vres)
        #chiinterp = interpolate.PchipInterpolator(tres, chires)
        #gammainterp = interpolate.PchipInterpolator(tres, gammares)
        #tetainterp = interpolate.PchipInterpolator(tres, tetares)
        #laminterp = interpolate.PchipInterpolator(tres, lamres)
        #hinterp = interpolate.PchipInterpolator(tres, hres)
        #minterp = interpolate.PchipInterpolator(tres, mres)

        #vineq = vinterp(time_ineq)
        #chiineq = chiinterp(time_ineq)
        #gammaineq = gammainterp(time_ineq)
        #tetaineq = tetainterp(time_ineq)
        #lamineq = laminterp(time_ineq)
        #hineq = hinterp(time_ineq)
        #mineq = minterp(time_ineq)
        #alfaineq = alfainterp(time_ineq)
        #deltaineq = deltainterp(time_ineq)
        #deltafineq = deltafinterp(time_ineq)
        #tauineq = tauinterp(time_ineq)

        #states_after = np.vstack((vineq, chiineq, gammaineq, tetaineq, lamineq, hineq, mineq))
        #controls_after = np.vstack((alfaineq, deltaineq, deltafineq, tauineq))

        states_after = np.vstack((vres, chires, gammares, tetares, lamres, hres, mres))
        controls_after = np.vstack((alfares, deltares, deltares, taures))

        ineq_cond = np.hstack((ineq_cond, inequalityAll(states_after, controls_after, len(tres))))

        timestart = timeend
        '''the values output of this function are not scaled'''

    return ineq_cond


def MultiPlot(var, dyn):
    #print("multi plot")
    # alfa = np.zeros((NContPoints))
    # delta = np.zeros((NContPoints))
    # deltaf = np.zeros((NContPoints))
    # tau = np.zeros((NContPoints))
    # states_atNode = np.zeros((0))
    # tres = np.zeros((0))
    # vres = np.zeros((0))
    # chires = np.zeros((0))
    # gammares = np.zeros((0))
    # lamres = np.zeros((0))
    # tetares = np.zeros((0))
    # hres = np.zeros((0))
    # mres = np.zeros((0))
    time = np.zeros((1))
    alfaCP = np.zeros((Nleg, NContPoints))
    deltaCP = np.zeros((Nleg, NContPoints))
    deltafCP = np.zeros((Nleg, NContPoints))
    tauCP = np.zeros((Nleg, NContPoints))
    res = open("res_{}_{}.txt".format(os.path.basename(__file__), timestr), "w")
    for i in range(Nleg):
        alfa = np.zeros((NContPoints))
        delta = np.zeros((NContPoints))
        deltaf = np.zeros((NContPoints))
        tau = np.zeros((NContPoints))
        controls = np.zeros((NContPoints,))
        states = var[i * Nstates:(i + 1) * Nstates]

        if i == 0:
            timestart = 0
        timeend = timestart + var[i + varTot] * unit_t
        time = np.concatenate((time, (timeend,)))
        tC = np.linspace(timestart, timeend, NContPoints)
        for k in range(NContPoints):
            alfa[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + Ncontrols * k]
            alfaCP[i, k] = alfa[k] * unit_alfa
            delta[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 1 + Ncontrols * k]
            deltaCP[i, k] = delta[k] * unit_delta
            deltaf[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 2 + Ncontrols * k]
            deltafCP[i, k] = deltaf[k] * unit_deltaf
            tau[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 3 + Ncontrols * k]
            tauCP[i, k] = tau[k] * unit_tau * 2 - 1
        controls = np.vstack((alfa, delta, deltaf, tau))

        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, alfaInt, deltaInt, deltafInt, tauInt = SingleShooting(
            states, controls, dyn, timestart, timeend, Nint)

        tetares = tetares - np.deg2rad(90)
        lamres = lamres * 2 - obj.incl
        taures = taures * 2 - 1


        Press, rho, c = isaMulti(hres, obj.psl, obj.g0, obj.Re)
        Press = np.asarray(Press, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        M = vres / c

        L, D, MomA = aeroForcesMulti(M, alfares, deltafres, cd, cl, cm, vres, obj.wingSurf, rho, obj.lRef, obj.M0, mres,
                                     obj.m10, obj.xcg0, obj.xcgf, obj.pref, len(tres))
        L = np.asarray(L, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        MomA = np.asarray(MomA, dtype=np.float64)

        T, Deps, isp, MomT = thrustMulti(Press, mres, presv, spimpv, deltares, taures, len(tres), obj.psl, obj.M0, obj.m10,
                                         obj.lRef, obj.xcgf, obj.xcg0)
        T = np.asarray(T, dtype=np.float64)
        Deps = np.asarray(Deps, dtype=np.float64)
        MomT = np.asarray(MomT, dtype=np.float64)
        isp = np.asarray(isp, dtype=np.float64)

        MomTot = MomA + MomT

        r1 = hres + obj.Re
        Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
        Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
        mf = mres / np.exp((Dv1 + Dv2) / (obj.g0*isp))

        g0 = obj.g0
        eps = Deps + alfares
        g = []
        for alt in hres:
            if alt == 0:
                g.append(g0)
            else:
                g.append(obj.g0 * (obj.Re / (obj.Re + alt)) ** 2)
        g = np.asarray(g, dtype=np.float64)
        # dynamic pressure

        q = 0.5 * rho * (vres ** 2)

        # accelerations

        ax = (T*np.cos(Deps) - D * np.cos(alfares) + L * np.sin(alfares)) / mres
        az = (T*np.sin(Deps) + D * np.sin(alfares) + L * np.cos(alfares)) / mres

        res.write(
            "Number of leg: " + str(Nleg) + "\n" + "Max number Optimization iterations: " + str(maxIterator) + "\n"
            + "Number of NLP iterations: " + str(maxiter) + "\n" + "Leg Number:" + str(i) + "\n" + "v: " + str(
                vres) + "\n" + "Chi: " + str(np.rad2deg(chires))
            + "\n" + "Gamma: " + str(np.rad2deg(gammares)) + "\n" + "Teta: " + str(
                np.rad2deg(tetares)) + "\n" + "Lambda: "
            + str(np.rad2deg(lamres)) + "\n" + "Height: " + str(hres) + "\n" + "Mass: " + str(
                mres) + "\n" + "mf: " + str(mf) + "\n"
            + "Objective Function: " + str(-mf / unit_m) + "\n" + "Alfa: "
            + str(np.rad2deg(alfares)) + "\n" + "Delta: " + str(deltares) + "\n" + "Delta f: " + str(
                np.rad2deg(deltafres)) + "\n"
            + "Tau: " + str(taures) + "\n" + "Eps: " + str(np.rad2deg(eps)) + "\n" + "Lift: "
            + str(L) + "\n" + "Drag: " + str(D) + "\n" + "Thrust: " + str(T) + "\n" + "Spimp: " + str(
                isp) + "\n" + "c: "
            + str(c) + "\n" + "Mach: " + str(M) + "\n" + "Time vector: " + str(tres) + "\n" + "Press: " + str(
                Press) + "\n" + "Dens: " + str(rho) + "\n" + "Time elapsed during optimization: " + tformat)

        downrange = - (vres ** 2) / g * np.sin(2 * chires)
        timestart = timeend

        plt.figure(0)
        plt.title("Velocity")
        plt.plot(tres, vres)
        plt.ylabel("m/s")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "velocity" + ".png")

        plt.figure(1)
        plt.title("Flight path angle \u03C7")
        plt.plot(tres, np.rad2deg(chires))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "chi" + ".png")

        plt.figure(2)
        plt.title("Angle of climb \u03B3")
        plt.plot(tres, np.rad2deg(gammares))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "gamma" + ".png")

        plt.figure(3)
        plt.title("Longitude \u03B8")
        plt.plot(tres, np.rad2deg(tetares))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "theta" + ".png")

        plt.figure(4)
        plt.title("Latitude \u03BB")
        plt.plot(tres, np.rad2deg(lamres))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "lambda" + ".png")

        plt.figure(5)
        plt.title("Flight angles")
        plt.plot(tres, np.rad2deg(chires), color="g")
        plt.plot(tres, np.rad2deg(gammares), color="b")
        plt.plot(tres, np.rad2deg(tetares), color="r")
        plt.plot(tres, np.rad2deg(lamres), color="k")
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Chi", "Gamma", "Theta", "Lambda"], loc="best")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "angles" + ".png")

        plt.figure(6)
        plt.title("Altitude")
        plt.plot(tres, hres / 1000)
        plt.ylabel("km")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "altitude" + ".png")

        plt.figure(7)
        plt.title("Mass")
        plt.plot(tres, mres)
        plt.ylabel("kg")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "mass" + ".png")

        plt.figure(8)
        plt.title("Angle of attack \u03B1")
        plt.plot(tC, np.rad2deg(alfaCP[i, :]), 'ro')
        plt.plot(tres, np.rad2deg(alfares))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "alpha" + ".png")

        plt.figure(9)
        plt.title("Throttles")
        plt.plot(tres, deltares * 100, color='r')
        plt.plot(tres, taures * 100, color='k')
        plt.plot(tC, deltaCP[i, :] * 100, 'ro')
        plt.plot(tC, tauCP[i, :] * 100, 'ro')
        plt.ylabel("%")
        plt.xlabel("time [s]")
        plt.legend(["Delta", "Tau", "Control points"], loc="best")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "throttles" + ".png")

        plt.figure(10)
        plt.title("Body Flap deflection \u03B4")
        plt.plot(tC, np.rad2deg(deltafCP[i, :]), "ro")
        plt.plot(tres, np.rad2deg(deltafres))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "deltaf" + ".png")

        plt.figure(11)
        plt.title("Dynamic Pressure")
        plt.plot(tres, q / 1000)
        plt.ylabel("kPa")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "dynPress" + ".png")

        plt.figure(12)
        plt.title("Accelerations")
        plt.plot(tres, ax, color='b')
        plt.plot(tres, az, color='r')
        plt.ylabel("m/s^2")
        plt.xlabel("time [s]")
        plt.legend(["ax", "az"], loc="best")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "accelerations" + ".png")

        plt.figure(13)
        plt.title("Downrange")
        plt.plot(downrange / 1000, hres / 1000)
        plt.ylabel("km")
        plt.xlabel("km")
        plt.savefig(savefig_file + "downrange" + ".png")

        plt.figure(14)
        plt.title("Forces")
        plt.plot(tres, T / 1000, color='r')
        plt.plot(tres, L / 1000, color='b')
        plt.plot(tres, D / 1000, color='k')
        plt.ylabel("kN")
        plt.xlabel("time [s]")
        plt.legend(["Thrust", "Lift", "Drag"], loc="best")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "forces" + ".png")

        plt.figure(15)
        plt.title("Mach")
        plt.plot(tres, M)
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "mach" + ".png")

        plt.figure(16)
        plt.title("Total pitching Moment")
        plt.plot(tres, MomTot / 1000, color='k')
        plt.axhline(5, 0, tres[-1], color='r')
        plt.axhline(-5, 0, tres[-1], color='r')
        plt.ylabel("kNm")
        plt.xlabel("time [s]")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "moment" + ".png")

    res.close()
    plt.show()


def equality(var):
    '''this functions applies the equality conditions, in the knotting points, plus final states
    conditions and controls initial conditions'''
    #print("Equality counter: ", obj.eqcounter)

    conj = MultiShooting(var, dynamicsInt)
    #v = conj[varStates - 7] * unit_v
    h = conj[varStates - 2] * unit_h
    lam = conj[varStates - 3] * unit_lam *2 - obj.incl
    #var[varStates - Nstates:varStates] = conj[Nstates * (Nleg - 1):]
    #vt = np.sqrt(obj.GMe / (obj.Re + h)) - obj.omega * np.cos(lam) * (obj.Re + h)
    vtAbs, chiass, vtAbs2 = vass(conj[varStates-7:varStates]*states_unit, var[varTot-4:varTot]*controls_unit, dynamicsVel, obj.omega)
    vvv = np.sqrt(obj.GMe / (obj.Re + h))
    #print("1. Speed: ", v)
    #print("2. Orbital velocity aim: ", vvv)
    #print("3. Orbital velocity corrected: ", vt)
    #print("4. Vela: ", vtAbs)
    #print("5. Vela2: ", vtAbs2)
    if np.cos(obj.incl)/np.cos(lam) > 1:
        chifin = np.pi
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam))
        #print(lam, chifin)
    #print(np.rad2deg(obj.chi_fin), np.rad2deg(chiass), np.rad2deg(chifin))
    eq_cond = np.zeros((0))
    #eq_cond = np.concatenate((eq_cond, abs(var[0:4] - (v_init[0] / unit_v, chi_init[0] / unit_chi, gamma_init[0] / unit_gamma, teta_init[0]/unit_teta))))
    #eq_cond = np.concatenate((eq_cond, abs(var[5:7] - (h_init[0] / unit_h, m_init[0] / unit_m))))
    eq_cond = np.concatenate((eq_cond, abs(var[0:7] - (v_init[0] / unit_v, chi_init[0] / unit_chi, gamma_init[0] / unit_gamma, teta_init[0]/unit_teta,
                                                       lam_init[0]/unit_lam, h_init[0] / unit_h, m_init[0] / unit_m))))
    eq_cond = np.concatenate((eq_cond, abs(var[Nstates:varStates] - conj[:Nstates*(Nleg-1)]))) # knotting conditions
    eq_cond = np.concatenate((eq_cond, (abs(var[varStates]),)))  # init cond on alpha
    eq_cond = np.concatenate((eq_cond, (abs(var[varStates + 1] - 1.0/unit_delta),)))  # init cond on delta
    eq_cond = np.concatenate((eq_cond, (abs(var[varStates + 2]),)))  # init cond on deltaf
    eq_cond = np.concatenate((eq_cond, (abs(var[varStates + 3] - 0.5/unit_tau),)))  # init cond on tau
    #eq_cond = np.concatenate((eq_cond, (abs(conj[Nstates*(Nbar-1) - 7] - vtAbs2 / unit_v),)))  # equality condition on orbital velocity for insertion
    eq_cond = np.concatenate((eq_cond, (abs((vvv - vtAbs) / unit_v),)))
    eq_cond = np.concatenate((eq_cond, (abs((chifin - chiass) / unit_chi),)))
    #eq_cond = np.concatenate((eq_cond, (abs(conj[Nstates*(Nbar-1) - 6] - chiass / unit_chi),)))
    eq_cond = np.concatenate((eq_cond, (abs(conj[varStates - 5]),)))  # final condition on gamma
    #print(eq_cond)

    return eq_cond


def cost_fun(var):
    '''this is the cost function of the problem, which is the propellant mass maximization'''

    con = MultiShooting(var, dynamicsInt)
    m = con[varStates - 1] * unit_m
    h = con[varStates - 2] * unit_h
    delta = var[varStates+1] * unit_delta
    tau = var[varStates+3] * unit_tau * 2 - 1


    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                     obj.lRef, obj.xcgf, obj.xcg0)

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / (obj.g0*isp))

    return - mf / obj.M0


def display_func(var):
    '''function to display data at the end of each iteration'''
    #print("display func")
    con = MultiShooting(var, dynamicsInt)
    m = con[varStates - 1] * unit_m
    h = con[varStates - 2] * unit_h
    delta = var[varStates + 1] * unit_delta
    tau = var[varStates + 3]*unit_tau*2-1

    time_vec = np.zeros((1))
    for i in range(Nleg):
        new = time_vec[-1] + var[i + varTot] * unit_t
        time_vec = np.hstack((time_vec, new))

    # Hohmann transfer mass calculation
    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / (obj.g0*isp))

    print("m before Ho : {0:.5f}".format(m))
    print("mf          : {0:.5f}".format(mf))
    print("altitude Hohmann starts: {0:.5f}".format(h))
    print("final time  : {}".format(time_vec))


'''definition of constraints dictionaries'''

cons = ({'type': 'eq',
         'fun': equality},
        {'type': 'ineq',
         'fun': ineqCond})  # equality and inequality constraints

cons2 = {'type': 'eq',
         'fun': equality}  # only equality constraints

'''NLP SOLVER'''

iterator = 0

while iterator < maxIterator:
    print("---- iteration : {0} ----".format(iterator + 1))

    opt = optimize.minimize(cost_fun,
                            X0,
                            constraints=cons,
                            bounds=bnds,
                            method='SLSQP',
                            options={"ftol": ftol,
                                     "eps":eps,
                                     "maxiter": maxiter,
                                     "disp": True})

    X0 = opt.x
    display_func(X0)
    if not (opt.status):
        break
    iterator += 1

end = time.time()
time_elapsed = end-start
tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
print("Time elapsed:for total optimization ", tformat)

MultiPlot(X0, dynamicsInt)

