from OpenGoddard.optimize import Guess
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import time
import datetime
import sys
from smooth_fun import *
import os
import dynamics as dyns
import ShootingFunctions as shoot
import plot as plt
from scipy.sparse import csc_matrix, save_npz, load_npz

sys.path.insert(0, 'home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP')
import models as mod

timestr = time.strftime("%Y%m%d-%H%M%S")
flag_save = True
laptop = False
if flag_save and laptop:
    os.makedirs("/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm"
               "/Python/trust_constr_solver/Results/MultiShooting_trust-constr_{}_{}_".format(os.path.basename(__file__), timestr))
    savefig_file = "/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm/Python/trust_constr_solver/Results/MultiShooting_trust-constr_{}_{}/Plot_".format(os.path.basename(__file__), timestr)
    savedata_file ="/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm/Python/trust_constr_solver/Results/MultiShooting_trust-constr_{}_{}_/Data_".format(os.path.basename(__file__), timestr)
elif flag_save and not laptop:
    os.makedirs("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/trust_constr_solver/Results/MultiShooting_trust-constr_{}_{}".format(os.path.basename(__file__), timestr))
    savefig_file = "/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/trust_constr_solver/Results/MultiShooting_trust-constr_{}_{}/Plot_".format(os.path.basename(__file__), timestr)
    savedata_file ="/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/trust_constr_solver/Results/MultiShooting_trust-constr_{}_{}/Data_".format(os.path.basename(__file__), timestr)

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
        self.gammamax = np.deg2rad(89.9)
        self.gammamin = np.deg2rad(-50)
        self.chimax = np.deg2rad(150)
        self.chimin = np.deg2rad(100)
        self.lammax = np.deg2rad(30)
        self.lammin = np.deg2rad(2)
        self.tetamax = np.deg2rad(-10)
        self.tetamin = np.deg2rad(-70)
        self.hmax = 2e5
        self.hmin = 0.5
        self.vmax = 1e4
        self.vmin = 0.5
        self.alphamax = np.deg2rad(40)
        self.alphamin = np.deg2rad(-2)
        self.deltamax = 1.0
        self.deltamin = 0.0
        self.vstart = 0.1
        self.hstart = 0.1


def constraints(var):
    if (var==obj.varOld).all():
        return np.concatenate((obj.eqOld, obj.ineqOld))

    else:
        ineq_c, eq_c, cost = shoot.SingleShooting(var, dyns.dynamicsInt, Nint, Nstates, NContPoints, obj, varStates, Ncontrols, cl, cd, cm, presv, spimpv, NineqCond, cont_init)
        return np.concatenate((eq_c, ineq_c))

def constraints_slsqp(var, type):
    if (var==obj.varOld).all():
        if type == 'eq':
            return obj.eqOld
        else:
            return obj.ineqOld
    else:
        ineq, eq, cost = shoot.SingleShooting(var, dyns.dynamicsInt, Nint, Nstates, NContPoints, obj, varStates, Ncontrols, cl, cd, cm, presv, spimpv, NineqCond, cont_init)
        if type == 'eq':
            return eq
        else:
            return ineq

cons_slsqp = ({'type': 'eq',
               'fun': constraints_slsqp,
               'args': ("eq",)},  # equality and inequality constraints
              {'type': 'ineq',
               'fun': constraints_slsqp,
               'args': ("ineq",)})  # equality and inequality constraints

def cost_fun(var):
    if (var==obj.varOld).all():
        return obj.costOld
    else:
        ineq, eq, cost = shoot.SingleShooting(var, dyns.dynamicsInt, Nint, Nstates, NContPoints, obj, varStates, Ncontrols, cl, cd, cm, presv, spimpv, NineqCond, cont_init)
        return cost


if __name__ == '__main__':
    obj = Spaceplane()

    '''reading of aerodynamic coefficients and specific impulse from file'''

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

    tfin = 350 # initial time
    NContPoints = 20  # number of control points for interpolation inside each interval
    Nint = 300 # number of points for each single shooting integration
    Nstates = 1  # number of states
    Ncontrols = 2  # number of controls
    varStates = Nstates # total number of optimization variables for states
    varControls = Ncontrols * NContPoints   # total number of optimization variables for controls
    varTot = varStates + varControls  # total number of optimization variables for states and controls
    NineqCond = 10 # Nleg * NContPoints - Nbar + 2
    tcontr = np.linspace(0, tfin, int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess
    solver='SLSQP'
    save_matrix = False
    '''NLP solver parameters'''
    general_tol = 1e-8
    tr_radius = 1
    constr_penalty = 10
    maxiter = 1 # max number of iterations for nlp solver

    '''definiton of initial conditions'''

    # set vector of initial conditions of states and controls. Angles are in radians
    X = np.zeros((0))
    U = np.zeros((0))

    ####### INITIAL CONDITIONS  ON CONTROLS #######
    t_contr_init = np.linspace(0, tfin, NContPoints)
    alfa_init = np.ones(len(t_contr_init))*np.deg2rad(5) #Guess.linear(t_contr_init, np.deg2rad(2), np.deg2rad(8))
    part1 = np.repeat(1.0, int(len(t_contr_init)/3))
    part2 = Guess.linear(t_contr_init[int(len(t_contr_init)/3):], obj.deltamax, obj.deltamin)
    delta_init = np.hstack((part1, part2))
    cont_init = np.array((alfa_init[0], delta_init[0]))  # , deltaf_init[0], tau_init[0], mu_init[0]))
    UGuess = np.array((alfa_init, delta_init))  # , deltaf_init, tau_init, mu_init))  # states initial guesses

    ###### INITIAL CONDITIONS ON STATES  #########
    states_init = np.array((1.0, obj.chistart, obj.gammastart, obj.thetastart, obj.lamstart, 1.0, obj.M0))  # initial conditions on states


    for i in range(int(varControls / Ncontrols)):
        '''creation of vector of controls initial guesses'''
        for j in range(Ncontrols):
            U = np.hstack((U, UGuess[j][i]))

    UbS = np.array([obj.chimax])
    LbS = np.array([obj.chimin])
    UbC = np.tile([obj.alphamax, obj.deltamax], NContPoints)
    LbC = np.tile([obj.alphamin, obj.deltamin], NContPoints)

    X0d = np.hstack((states_init[1], U, tfin))  # array of initial conditions here all the angles are in radians!!!!!
    obj.varOld = np.zeros((len(X0d)))

    Tlb = [250]
    Tub = [700]
    obj.LBV = np.hstack((LbS, LbC, Tlb))
    obj.UBV = np.hstack((UbS, UbC, Tub))

    X0a = (X0d - obj.LBV)/(obj.UBV - obj.LBV)  # adimensional array of variables

    if solver == "SLSQP":
        bndX_slsqp = ((0.0, 1.0),)
        bndU_slsqp = ((0.0, 1.0), (0.0, 1.0))#, (0.0, 1.0), (0.0, 1.0))
        bndT_slsqp = ((0.0, 1.0),)
        bnds_slsqp = bndX_slsqp + bndU_slsqp * NContPoints + bndT_slsqp
    else:
        '''constratints for trust-constr solver'''

        lbeq = ([0.0])
        ubeq = ([0.0])

        lbineq = ([0.0])  # lower bound for inequality constraints
        ubineq = ([np.inf])  # upper bound for inequality constraints

        lb = lbeq * 4 + lbineq * (5 * NineqCond + 4)  # all lower bounds
        ub = ubeq * 4 + ubineq * (5 * NineqCond + 4)  # all upper bounds


        if save_matrix:
            cons = NonlinearConstraint(constraints, lb, ub, finite_diff_jac_sparsity=None)
        else:
            sparseJac = load_npz("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/SingleShooting_InitCond/FestipSparsity.npz")
            sp = sparseJac.todense()
            row = np.shape(sp)[0]
            column = np.shape(sp)[1]
            for i in range(row):
                for j in range(column):
                    if sp[i, j] != 0:
                        sp[i, j] = 1
            cons = NonlinearConstraint(constraints, lb, ub, finite_diff_jac_sparsity=sp)
        '''bounds for trust-constr solver'''
        Tlb = ([0.0])  # time lower bounds
        Tub = ([1.0])  # time upper bounds
        Xlb = ([0.0])
        Xub = ([1.0])
        Ulb = ([0.0])
        Uub = ([1.0])
        Vlb = Xlb * Nstates + Ulb * Ncontrols * NContPoints + Tlb
        Vub = Xub * Nstates + Uub * Ncontrols * NContPoints + Tub
        bnds = Bounds(Vlb, Vub)


    '''NLP SOLVER'''
    iterator = 0
    tot_it = 1
    start = time.time()
    while iterator < tot_it:
        if solver == "SLSQP":
            opt = minimize(cost_fun, X0a, constraints=cons_slsqp, bounds=bnds_slsqp, method='SLSQP', options={"disp":True, "iprint":2})
        elif save_matrix:
            opt = minimize(cost_fun, X0a, constraints=cons, bounds=bnds, method='trust-constr', options={"verbose":2, 'maxiter':maxiter})
        else:
            opt = minimize(cost_fun, X0a, constraints=cons, bounds=bnds, method='trust-constr', options={"verbose": 2, "initial_tr_radius":tr_radius, "initial_constr_penalty":constr_penalty})

        X0a = opt.x
        iterator += 1
    end = time.time()
    #p.close()
    #p.join()
    time_elapsed = end-start
    tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
    print("Time elapsed for total optimization ", tformat)
    if save_matrix:
        sparse = csc_matrix(opt.jac[0])
        save_npz("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/SingleShooting_InitCond/FestipSparsity.npz", sparse)
    plt.plot(opt.x, Nint, NContPoints, obj, Nstates, varStates, Ncontrols, cl, cm ,cd, presv, spimpv, flag_save, savedata_file, savefig_file, maxiter, timestr, tformat)