from scipy.optimize import minimize
from OpenGoddard.optimize import Guess
import time
import datetime
import os
from multiprocessing import Pool
import numpy as np
import ShootingFunctions as shoot
import dynamics as dyns
import plot as plt
import models as mod

timestr = time.strftime("%Y%m%d-%H%M%S")
flag_save = True
laptop = False
if flag_save and laptop:
    os.makedirs("/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm"
               "/Python/SLSQP_solver/Results/MultiShooting_SLSQP_{}_{}_".format(os.path.basename(__file__), timestr))
    savefig_file = "/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm/Python/SLSQP_solver/Results/MultiShooting_SLSQP_{}_{}_/Plot_".format(os.path.basename(__file__), timestr)
    savedata_file ="/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm/Python/SLSQP_solver/Results/MultiShooting_SLSQP_{}_{}_/Data_".format(os.path.basename(__file__), timestr)
elif flag_save and not laptop:
    os.makedirs("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/SLSQP_solver/Results/MultiShooting_SLSQP_{}_{}_".format(os.path.basename(__file__), timestr))
    savefig_file = "/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/SLSQP_solver/Results/MultiShooting_SLSQP_{}_{}_/Plot_".format(os.path.basename(__file__), timestr)
    savedata_file ="/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/SLSQP_solver/Results/MultiShooting_SLSQP_{}_{}_/Data_".format(os.path.basename(__file__), timestr)

'''vehicle parameters'''

class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.lamstart = np.deg2rad(5.2)  # deg latitude
        self.thetastart = np.deg2rad(-52.775)  # deg longitude
        self.chistart = np.deg2rad(125)  # deg flight direction
        self.incl = np.deg2rad(51.6)  # deg orbit inclination
        self.gammastart = np.deg2rad(89)  # deg
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
        self.tvert = 2 # [s]
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.lamstart))
        self.varOld = np.zeros((0))
        self.costOld = np.zeros((0))
        self.eqOld = np.zeros((0))
        self.ineqOld = np.zeros((0))
        self.States = np.zeros((0))
        self.Controls = np.zeros((0))
        self.gammamax = np.deg2rad(89)
        self.gammamin = np.deg2rad(-89)
        self.chimax = np.deg2rad(150)
        self.chimin = np.deg2rad(90)
        self.lammax = np.deg2rad(30)
        self.lammin = np.deg2rad(2)
        self.thetamax = 0.0
        self.thetamin = np.deg2rad(-70)
        self.hmax = 2e5
        self.hmin = 1.0
        self.vmax = 1e4
        self.vmin = 1.0
        self.alphamax = np.deg2rad(40)
        self.alphamin = np.deg2rad(-2)
        self.deltamax = 1.0
        self.deltamin = 0.0


def constraints(var, type):
    if (var==obj.varOld).all():
        if type == "eq":
            return obj.eqOld
        else:
            return obj.ineqOld
    else:
        eq, ineq, cost = shoot.MultiShooting(var, dyns.dynamicsInt, obj, p, Nint, Nleg, presv, spimpv, NContPoints, Nstates, varTot, varStates, Ncontrols, cl, cd, cm, states_init, cont_init)
        if type == "eq":
            return eq
        else:
            return ineq


def cost_fun(var):
    if (var==obj.varOld).all():
        return obj.costOld
    else:
        eq, ineq, cost = shoot.MultiShooting(var, dyns.dynamicsInt, obj, p, Nint, Nleg, presv, spimpv, NContPoints, Nstates, varTot, varStates, Ncontrols, cl, cd, cm, states_init, cont_init)
        return cost


cons = ({'type': 'eq',
         'fun': constraints,
         'args':("eq",)},
        {'type': 'ineq',
         'fun': constraints,
         'args': ("ineq",)})  # equality and inequality constraints


if __name__ == '__main__':

    obj = Spaceplane()

    cl = mod.fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/clfile.txt")
    cd = mod.fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cdfile.txt")
    cm = mod.fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cmfile.txt")
    cl = np.asarray(cl)
    cd = np.asarray(cd)
    cm = np.asarray(cm)

    with open("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/impulse.dat") as f:
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
    Nbar = 6  # number of conjunction points
    Nleg = Nbar - 1  # number of multiple shooting sub intervals
    NContPoints = 5  # number of control points for interpolation inside each interval
    Nint = 250  # number of points for each single shooting integration
    Nstates = 7  # number of states
    Ncontrols = 2  # number of controls
    varStates = Nstates * Nleg  # total number of optimization variables for states
    varControls = Ncontrols * Nleg * NContPoints  # total number of optimization variables for controls
    varTot = varStates + varControls  # total number of optimization variables for states and controls
    NineqCond = Nint
    tnew = np.linspace(0, time_tot, Nbar)  # time vector used for interpolation of states initial guess
    tcontr = np.linspace(0, time_tot, int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess
    tstat = np.linspace(0, time_tot, Nleg)
    unit_t = 1000
    Nintplot = 1000

    '''NLP solver parameters'''
    maxiter = 30  # max number of iterations for nlp solver
    ftol = 1e-8  # numeric tolerance of nlp solver
    eps = 1e-09  # increment of the derivative
    maxIterator = 5  # max number of optimization iterations

    '''definiton of initial conditions'''

    # set vector of initial conditions of states and controls
    X = np.zeros((0))
    U = np.zeros((0))

    v_init = Guess.linear(tstat, obj.vmin, obj.Vtarget)
    chi_init = Guess.linear(tstat, obj.chistart, obj.chi_fin)
    gamma_init = Guess.linear(tstat, obj.gammastart, 0.0)
    teta_init = Guess.constant(tstat, obj.thetastart)
    lam_init = Guess.constant(tstat, obj.lamstart)
    h_init = Guess.linear(tstat, obj.hmin, obj.Hini)
    m_init = Guess.linear(tstat, obj.M0, obj.m10)

    alfa_init = Guess.zeros(tcontr)
    part1 = np.repeat(1.0, int(len(tcontr) / 3))
    part2 = Guess.linear(tcontr[int(len(tcontr) / 3):], obj.deltamax, obj.deltamin)
    delta_init = np.hstack((part1, part2))
    deltaf_init = Guess.zeros(tcontr)
    tau_init = Guess.zeros(tcontr)
    mu_init = Guess.zeros(tcontr)

    states_init = np.array((v_init[0], chi_init[0], gamma_init[0], teta_init[0], lam_init[0], h_init[0], m_init[0]))
    cont_init = np.array((alfa_init[0], delta_init[0])) #, deltaf_init[0], tau_init[0], mu_init[0]))

    XGuess = np.array((v_init, chi_init, gamma_init, teta_init, lam_init, h_init, m_init))  # states initial guesses

    UGuess = np.array((alfa_init, delta_init)) #, deltaf_init, tau_init, mu_init))  # states initial guesses

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

    uplimx = np.tile([obj.vmax, obj.chimax, obj.gammamax, obj.thetamax, obj.lammax, obj.hmax, obj.M0], Nleg)
    inflimx = np.tile([obj.vmin, obj.chimin, obj.gammamin, obj.thetamin, obj.lammin, obj.hmin, obj.m10], Nleg)
    uplimu = np.tile([obj.alphamax, obj.deltamax], Nleg * NContPoints)
    inflimu = np.tile([obj.alphamin, obj.deltamin], Nleg * NContPoints)


    X0d = np.hstack((X, U, dt))  # vector of initial conditions here all the angles are in degrees!!!!!
    obj.varOld = np.zeros((len(X0d)))


    LbS = [obj.vmin, np.deg2rad(110), np.deg2rad(88), np.deg2rad(-53), np.deg2rad(4.8), obj.hmin, obj.M0 - 10,
           10, np.deg2rad(100), np.deg2rad(50), np.deg2rad(-60), np.deg2rad(2.0), 1e3, 2.5e5,
           100, np.deg2rad(100), np.deg2rad(0), np.deg2rad(-60), np.deg2rad(2.0), 1e4, 1.5e5,
           500, np.deg2rad(100), np.deg2rad(-50), np.deg2rad(-60), np.deg2rad(2.0), 2e4, 1e5,
           1000, np.deg2rad(100), np.deg2rad(-20), np.deg2rad(-60), np.deg2rad(2.0), 5e4, 5e4]

    UbS = [1.5, np.deg2rad(115), np.deg2rad(89.99), np.deg2rad(-51), np.deg2rad(5.8), 1.5, obj.M0,
           1500, np.deg2rad(150), np.deg2rad(70), np.deg2rad(-45), np.deg2rad(8.0), 3e4, 3.5e5,
           3500, np.deg2rad(150), np.deg2rad(30), np.deg2rad(-45), np.deg2rad(15.0), 8e4, 3e5,
           5500, np.deg2rad(150), np.deg2rad(30), np.deg2rad(-45), np.deg2rad(25.0), 1e5, 2e5,
           6500, np.deg2rad(150), np.deg2rad(30), np.deg2rad(-45), np.deg2rad(25.0), 1.2e5, 1e5]

    LbC = [np.deg2rad(-1.0), 0.9, #np.deg2rad(-20.0), -0.1, np.deg2rad(-1),
           np.deg2rad(-1.0), 0.9, #np.deg2rad(-20.0), -0.2, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -0.2, np.deg2rad(-5),
           #np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -0.5, np.deg2rad(-10),
           #np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -0.5, np.deg2rad(-15),
           np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -0.6, np.deg2rad(-20),
           np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -0.6, np.deg2rad(-25), # leg1
           np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -0.8, np.deg2rad(-30),
           np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -1, np.deg2rad(-35),
           #np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -1, np.deg2rad(-60),
           #np.deg2rad(-2.0), 0.9, #np.deg2rad(-20.0), -1, np.deg2rad(-70),
           np.deg2rad(-2.0), 0.8, #np.deg2rad(-20.0), -1, np.deg2rad(-70),
           np.deg2rad(-2.0), 0.7, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.7, #np.deg2rad(-20.0), -1, np.deg2rad(-90), # leg2
           np.deg2rad(-2.0), 0.6, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           #np.deg2rad(-2.0), 0.6, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           #np.deg2rad(-2.0), 0.5, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.5, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.4, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.4, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.3, #np.deg2rad(-20.0), -1, np.deg2rad(-90), # leg3
           np.deg2rad(-2.0), 0.3, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.2, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           #np.deg2rad(-2.0), 0.2, #np.deg2rad(-20.0), -1, np.deg2rad(-90),
           #np.deg2rad(-2.0), 0.1, #np.deg2rad(-20.0), -1, np.deg2rad(-80),
           np.deg2rad(-2.0), 0.1, #np.deg2rad(-20.0), -0.9, np.deg2rad(-50),
           np.deg2rad(-2.0), 0.05,# np.deg2rad(-20.0), -0.8, np.deg2rad(-20),
           np.deg2rad(-2.0), 0.01,# np.deg2rad(-20.0), -0.5, np.deg2rad(-10), # leg4
           np.deg2rad(-2.0), 0.001,# np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.001,# np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           #np.deg2rad(-2.0), 0.001,# np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           #np.deg2rad(-2.0), 0.001,# np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.001,# np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.001,# np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.001]# np.deg2rad(-20.0), -0.01, np.deg2rad(-1)] # leg5

    UbC = [np.deg2rad(1.0), 1.0, #np.deg2rad(30), 0.1, np.deg2rad(1),
           np.deg2rad(1.0), 1.0, #np.deg2rad(30.0), 0.2, np.deg2rad(1),
           np.deg2rad(3.0), 1.0, #np.deg2rad(30.0), 0.2, np.deg2rad(5),
           #np.deg2rad(5.0), 1.0, #np.deg2rad(30.0), 0.5, np.deg2rad(10),
           np.deg2rad(10.0), 1.0,# np.deg2rad(30.0), 0.5, np.deg2rad(15),
           #np.deg2rad(15.0), 1.0,# np.deg2rad(30.0), 0.6, np.deg2rad(20),
           np.deg2rad(20.0), 1.0,# np.deg2rad(30.0), 0.6, np.deg2rad(25), # leg1
           np.deg2rad(25.0), 1.0,# np.deg2rad(30.0), 0.8, np.deg2rad(30),
           np.deg2rad(30.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(35),
          # np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(60),
           #np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(70),
           np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(70),
           np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(90),
           np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(90), # leg2
           np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(90),
           #np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(90),
           #np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 1.0,# np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.7,# np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.6,# np.deg2rad(30.0), 1, np.deg2rad(90),
           np.deg2rad(40.0), 0.6,# np.deg2rad(30.0), 1, np.deg2rad(90.0), # leg3
           #np.deg2rad(40.0), 0.5,# np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.5,# np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.5,# np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.4,# np.deg2rad(30.0), 1, np.deg2rad(80.0),
           #np.deg2rad(40.0), 0.4,# np.deg2rad(30.0), 0.9, np.deg2rad(50.0),
           np.deg2rad(40.0), 0.3,# np.deg2rad(30.0), 0.8, np.deg2rad(20.0),
           np.deg2rad(40.0), 0.25, #np.deg2rad(30.0), 0.5, np.deg2rad(10.0), # leg4
           np.deg2rad(40.0), 0.25, #np.deg2rad(30.0), 0.01, np.deg2rad(1),
           np.deg2rad(40.0), 0.2,# np.deg2rad(30.0), 0.01, np.deg2rad(1),
           #np.deg2rad(40.0), 0.2, #np.deg2rad(30.0), 0.01, np.deg2rad(1),
           #np.deg2rad(40.0), 0.2, #np.deg2rad(30.0), 0.01, np.deg2rad(1),
           np.deg2rad(40.0), 0.2, #np.deg2rad(30.0), 0.01, np.deg2rad(1),
           np.deg2rad(40.0), 0.2, #np.deg2rad(30.0), 0.01, np.deg2rad(1),
           np.deg2rad(40.0), 0.2] #np.deg2rad(30.0), 0.01, np.deg2rad(1)] #leg5

    Tlb = [3.0]
    Tub = [200]

    obj.LBV = np.hstack((LbS, LbC, np.repeat(Tlb, Nleg)))
    obj.UBV = np.hstack((UbS, UbC, np.repeat(Tub, Nleg)))
    X0a = (X0d - obj.LBV)/(obj.UBV - obj.LBV)
    # X0 has first all X guesses and then all U guesses
    # at this point the vectors of initial guesses for states and controls for every time interval are defined

    '''set upper and lower bounds for states, controls and time, scaled'''
    '''major issues with bounds!!!'''
    bndX = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    bndU = ((0.0, 1.0), (0.0, 1.0))
    bndT = ((0.0, 1.0),)
    bnds = bndX * Nleg + bndU * Nleg * NContPoints + bndT * Nleg

    iterator = 0
    p = Pool(processes=Nleg)
    start = time.time()
    while iterator < maxIterator:
        print("---- iteration : {0} ----".format(iterator + 1))
        opt = minimize(cost_fun,
                            X0a,
                            constraints=cons,
                            bounds=bnds,
                            method='SLSQP',
                            options={"maxiter": maxiter,
                                        "ftol": ftol,
                                        "iprint":2,
                                        "disp":True})

        X0a = opt.x

        if not (opt.status):
            break
        iterator += 1

    p.close()
    p.join()

    end = time.time()
    time_elapsed = end-start
    tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
    print("Time elapsed for total optimization ", tformat)
    plt.plot(X0a, Nintplot, flag_save, Nstates, obj, varTot, Nint, Nleg, varStates, Ncontrols, cl, cd, cm, presv, spimpv, maxiter, savefig_file)



