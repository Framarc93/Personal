from scipy import optimize
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint, basinhopping, shgo
import modelsMS as mods
import time
import datetime
from scipy.sparse import csc_matrix, save_npz, load_npz
import multiprocessing
import os
import dynamicsMS as dyns
import constraintsMS as const
import ShootingFunctionsMS as shot
import PlotMS as Plot
from functools import partial
import numpy as np
from scipy.interpolate import PchipInterpolator
from import_initialCond_MS import init_conds
import scipy.io as sio

'''Multiple Shooting Algorithm'''

start = time.time()
timestr = time.strftime("%Y%m%d-%H%M%S")
flag_save = True
laptop = False

if laptop:
    initial_path = "/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP"
else:
    initial_path = "/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP"

if flag_save and laptop:
    os.makedirs(initial_path + "/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}".format(os.path.basename(__file__), timestr))
    savefig_file = initial_path + "/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}/Plot_".format(os.path.basename(__file__), timestr)
    savedata_file = initial_path + "/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}/Data_".format(os.path.basename(__file__), timestr)
elif flag_save and not laptop:
    os.makedirs(initial_path + "/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}".format(os.path.basename(__file__), timestr))
    savefig_file = initial_path + "/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}/Plot_".format(os.path.basename(__file__), timestr)
    savedata_file = initial_path + "/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}/Data_".format(os.path.basename(__file__), timestr)

'''vehicle parameters'''


class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
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
        self.tvert = 2 #[s]
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))
        self.varOld = np.zeros((0))
        self.costOld = np.zeros((0))
        self.eqOld = np.zeros((0))
        self.ineqOld = np.zeros((0))
        self.States = np.zeros((0))
        self.Controls = np.zeros((0))
        self.gammamax = np.deg2rad(89)
        self.gammamin = np.deg2rad(-40)
        self.chimax = np.deg2rad(170)
        self.chimin = np.deg2rad(100)
        self.lammax = np.deg2rad(30)
        self.lammin = np.deg2rad(2)
        self.tetamax = np.deg2rad(-10)
        self.tetamin = np.deg2rad(-70)
        self.hmax = 2e5
        self.hmin = 1.0
        self.vmax = 1e4
        self.vmin = 1.0
        self.alfamax = np.deg2rad(40)
        self.alfamin = np.deg2rad(-2)
        self.deltamax = 1.0
        self.deltamin = 0.0
        self.vstart = 0.1
        self.hstart = 0.1
        self.mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
        self.angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0]
        self.bodyFlap = [-20, -10, 0, 10, 20, 30]
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]


def constraints(var):
    if (var==obj.varOld).all():
        return np.concatenate((obj.eqOld, obj.ineqOld))

    else:
        eq_c, ineq_c, cost = MultiShooting(var, dyns.dynamicsInt, obj, Nleg, Nint, NineqCond, presv, spimpv, NContPoints, Nstates, varTot, Ncontrols, varStates, cl, cd, cm, states_init, cont_init)
        return np.concatenate((eq_c, ineq_c))


def constraints_slsqp(var, type):
    if (var==obj.varOld).all():
        if type == "eq":
            return obj.eqOld
        else:
            return obj.ineqOld
    else:
        eq, ineq, cost = MultiShooting(var, dyns.dynamicsInt, obj, Nleg, Nint, NineqCond, presv, spimpv, NContPoints, Nstates, varTot, Ncontrols, varStates, cl, cd, cm, states_init, cont_init)
        if type == "eq":
            return eq
        else:
            return ineq


def cost_fun(var):
    if (var==obj.varOld).all():
        return obj.costOld
    else:
        eq, ineq, cost = MultiShooting(var, dyns.dynamicsInt, obj, Nleg, Nint, NineqCond, presv, spimpv, NContPoints, Nstates, varTot, Ncontrols, varStates, cl, cd, cm, states_init, cont_init)
        return cost

def MultiShooting(var, dyn, obj, Nleg, Nint, NineqCond, presv, spimpv, NContPoints, Nstates, varTot, Ncontrols, varStates, cl, cd, cm, states_init, cont_init):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in radians'''
    global penalty, pen
    penalty = False
    pen = []
    states_atNode = np.zeros((0))
    ineq_c = np.zeros((0))
    varD = var * (obj.UBV - obj.LBV) + obj.LBV

    res = p.map(partial(shot.SingleShootingMulti, var=varD, dyn=dyn, Nint=Nint, NContPoints=NContPoints, Nstates=Nstates,
                           varTot=varTot, Ncontrols=Ncontrols, varStates=varStates, obj=obj, cl=cl, cd=cd, cm=cm, presv=presv, spimpv=spimpv, states_init=states_init), range(Nleg))

    for i in range(Nleg):
        leg = res[i]
        vres = leg[0] #np.hstack((vres, leg[0]))
        chires = leg[1] #np.hstack((chires, leg[1]))
        gammares = leg[2] #np.hstack((gammares, leg[2]))
        tetares = leg[3] #np.hstack((tetares, leg[3]))
        lamres = leg[4] #np.hstack((lamres, leg[4]))
        hres = leg[5] #np.hstack((hres, leg[5]))
        mres = leg[6] #np.hstack((mres, leg[6]))
        tres = leg[7] #np.hstack((tres, leg[7]))
        alfares = leg[8] #np.hstack((alfares, leg[8]))
        deltares = leg[9] #np.hstack((deltares, leg[9]))
        #deltafres = np.hstack((deltafres, leg[10]))
        #taures = np.hstack((taures, leg[11]))
        #mures = np.hstack((mures, leg[12]))
        if i < Nleg-1:
            states_atNode = np.hstack((states_atNode, ((vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]))))  # new intervals

        vrescol = np.reshape(vres, (Nint, 1))
        chirescol = np.reshape(chires, (Nint, 1))
        gammarescol = np.reshape(gammares, (Nint, 1))
        tetarescol = np.reshape(tetares, (Nint, 1))
        lamrescol = np.reshape(lamres, (Nint, 1))
        hrescol = np.reshape(hres, (Nint, 1))
        mrescol = np.reshape(mres, (Nint, 1))
        alfarescol = np.reshape(alfares, (Nint, 1))
        deltarescol = np.reshape(deltares, (Nint, 1))
        #deltafrescol = np.reshape(deltafres, (Nint*Nleg, 1))
        #murescol = np.reshape(mures, (Nint*Nleg, 1))
        #taurescol = np.reshape(taures, (Nint*Nleg, 1))
        t_ineq = np.linspace(tres[0], tres[-1], NineqCond)

        vrescol = np.nan_to_num(vrescol)
        chirescol = np.nan_to_num(chirescol)
        gammarescol = np.nan_to_num(gammarescol)
        tetarescol = np.nan_to_num(tetarescol)
        lamrescol = np.nan_to_num(lamrescol)
        hrescol = np.nan_to_num(hrescol)
        mrescol = np.nan_to_num(mrescol)
        alfarescol = np.nan_to_num(alfarescol)
        deltarescol = np.nan_to_num(deltarescol)

        v_Int = PchipInterpolator(tres, vrescol)
        v_ineq = v_Int(t_ineq)
        chi_Int = PchipInterpolator(tres, chirescol)
        chi_ineq = chi_Int(t_ineq)
        gamma_Int = PchipInterpolator(tres, gammarescol)
        gamma_ineq = gamma_Int(t_ineq)
        teta_Int = PchipInterpolator(tres, tetarescol)
        teta_ineq = teta_Int(t_ineq)
        lam_Int = PchipInterpolator(tres, lamrescol)
        lam_ineq = lam_Int(t_ineq)
        h_Int = PchipInterpolator(tres, hrescol)
        h_ineq = h_Int(t_ineq)
        m_Int = PchipInterpolator(tres, mrescol)
        m_ineq = m_Int(t_ineq)
        alfa_Int_post = PchipInterpolator(tres, alfarescol)
        alfa_ineq = alfa_Int_post(t_ineq)
        delta_Int_post = PchipInterpolator(tres, deltarescol)
        delta_ineq = delta_Int_post(t_ineq)
        # deltafres = np.zeros(len(t))  # deltaf_Int(time_old)
        # taures = np.zeros(len(t))  # tau_Int(time_old)
        # mures = np.zeros(len(t))  # mu_Int(time_old)

        states_ineq = np.column_stack((v_ineq, chi_ineq, gamma_ineq, teta_ineq, lam_ineq, h_ineq, m_ineq))
        controls_ineq = np.column_stack((alfa_ineq, delta_ineq))  # , deltafrescol, taurescol, murescol))
        states_after = np.column_stack((vrescol, chirescol, gammarescol, tetarescol, lamrescol, hrescol, mrescol))
        controls_after = np.column_stack((alfarescol, deltarescol))#, deltafrescol, taurescol, murescol))

        obj.States = states_after
        obj.Controls = controls_after
        if i == 0:
            ineq_c = np.hstack((ineq_c, gamma_ineq.T[0]/obj.gammamax, h_ineq.T[0]/obj.hmax))
        ineq_c = np.hstack((ineq_c, const.inequalityAll(states_ineq, controls_ineq, NineqCond, obj, cl, cd, cm, presv, spimpv)))

    eq_c = const.equality(varD, states_atNode, varStates, NContPoints, obj, Ncontrols, Nleg, cl, cd, cm, presv, spimpv)
    v = states_after[-1, 0]
    h = states_after[-1, 5]
    m = states_after[-1, 6]
    delta = controls_after[-1, 1]
    tau = 0.0 #controls_after[-1, 2]

    Press, rho, c = isa(h, obj)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj)

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / (obj.g0 * isp))

    cost = -mf / obj.M0
    ineq_c = np.hstack((ineq_c, (mf-obj.m10)/obj.M0, (h-6e4)/obj.hmax, (v-6e3)/obj.vmax))

    '''max_i = np.nan_to_num(max(ineq_c))
    min_i = np.nan_to_num(min(ineq_c))
    #bnd_i = max(abs(min_i), max_i)
    max_e = np.nan_to_num(max(eq_c))
    min_e = np.nan_to_num(min(eq_c))
    #bnd_e = max(abs(min_e), max_e)
    exp_i = int(np.log10(max(abs(min_i), max_i)))
    red_i = 10 ** exp_i
    exp_e = int(np.log10(max(abs(min_e), max_e)))
    red_e = 10 ** exp_e
    new_sup_i = max_i/red_i
    new_inf_i = min_i/red_i
    new_sup_e = max_e/red_e
    new_inf_e = min_e/red_e
    norm_ineq = new_inf_i + ((new_sup_i - new_inf_i)/(max_i - min_i))*(ineq_c - min_i) # maps inequality between -1 and 1 using min and max of the current array
    norm_eq = new_inf_e + ((new_sup_e - new_inf_e)/(max_e - min_e))*(eq_c - min_e) # maps equality between -1 and 1 using min and max of the current array
    #print("Max norm ineq: {}, Min norm ineq: {}".format(max(norm_ineq), min(norm_ineq)))
    #print("Max norm eq: {}, Min norm eq: {}".format(max(norm_eq), min(norm_eq)))'''
    obj.eqOld = eq_c
    obj.costOld = cost
    obj.ineqOld = ineq_c
    obj.varOld = var

    return eq_c, ineq_c, cost


if __name__ == '__main__':
    obj = Spaceplane()

    '''reading of aerodynamic coefficients and specific impulse from file'''

    cl = mods.fileReadOrMS(initial_path + "/coeff_files/clfile.txt")
    cd = mods.fileReadOrMS(initial_path + "/coeff_files/cdfile.txt")
    cm = mods.fileReadOrMS(initial_path + "/coeff_files/cmfile.txt")
    with open(initial_path + "/coeff_files/impulse.dat") as f:
        impulse = []
        for line in f:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                impulse.append(line)

    f.close()

    cl = np.asarray(cl)
    cd = np.asarray(cd)
    cm = np.asarray(cm)

    presv = []
    spimpv = []

    for i in range(len(impulse)):
        presv.append(impulse[i][0])
        spimpv.append(impulse[i][1])

    presv = np.asarray(presv)
    spimpv = np.asarray(spimpv)

    '''set problem parameters'''
    source = 'matlab'
    if source == 'matlab':
        mat_contents = sio.loadmat(initial_path + "/workspace_init_cond.mat")
        time_tot = mat_contents['t'][0][-1]
    else:
        time_tot = np.load(initial_path + "/Collocation_Algorithm/nice_initCond/Data_timeTot.npy")[-1]

    discretization = 1 # [s]  how close are the propagation points in the legs
    Nbar = 6 # number of conjunction points
    Nleg = Nbar - 1  # number of multiple shooting sub intervals
    NContPoints = 7  # number of control points for interpolation inside each interval
    Nint = int((time_tot/Nleg)/discretization)# number of points for each single shooting integration
    Nstates = 7  # number of states
    Ncontrols = 2  # number of controls
    varStates = 1 + Nstates * (Nleg-1)  # total number of optimization variables for states
    varControls = Ncontrols * Nleg * NContPoints   # total number of optimization variables for controls
    varTot = varStates + varControls  # total number of optimization variables for states and controls
    NineqCond = 7 # Nleg * NContPoints - Nbar + 2
    tnew = np.linspace(0, time_tot, Nbar)  # time vector used for interpolation of states initial guess
    tcontr = np.linspace(0, time_tot, int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess
    unit_t = 1000
    save_matrix = False
    if save_matrix:
        flag_save = False
    '''NLP solver parameters'''
    solver = 'SLSQP'
    OptGlobal = False
    general_tol = 1e-8
    tr_radius = 10
    constr_penalty = 10
    maxiter = 20 # max number of iterations for nlp solver
    nbCPU = multiprocessing.cpu_count()
    p = multiprocessing.Pool(nbCPU)
    '''definiton of initial conditions'''

    # set vector of initial conditions of states and controls. Angles are in radians

    U = np.zeros((0))

    t_stat = np.linspace(0, time_tot, Nbar)
    t_cont_vects = []
    for i in range(Nleg):
        t_cont_vects.append(np.linspace(t_stat[i], t_stat[i + 1], NContPoints))

    states_init, controls_init = init_conds(t_stat, t_cont_vects, source, initial_path)

    v_init = states_init[0]
    chi_init = states_init[1]
    gamma_init = states_init[2]
    teta_init = states_init[3]
    lam_init = states_init[4]
    h_init = states_init[5]
    m_init = states_init[6]
    alfa_init = controls_init[0]
    delta_init = controls_init[1]

    states_init = np.array((v_init[0], chi_init[0], gamma_init[0], teta_init[0], lam_init[0], h_init[0], m_init[0]))
    cont_init = np.array((alfa_init[0], delta_init[0]))#, deltaf_init[0], tau_init[0], mu_init[0]))
    X = chi_init[0]
    XGuess = np.array((v_init[1:], chi_init[1:], gamma_init[1:], teta_init[1:], lam_init[1:], h_init[1:], m_init[1:]))  # states initial guesses

    UGuess = np.array((alfa_init, delta_init))#, deltaf_init, tau_init, mu_init))  # states initial guesses

    for i in range(Nleg-1):
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

    uplimx = np.hstack((obj.chimax, np.tile([obj.vmax, obj.chimax, obj.gammamax, obj.tetamax, obj.lammax, obj.hmax, obj.M0], Nleg-1)))
    inflimx = np.hstack((obj.chimin, np.tile([obj.vmin, obj.chimin, obj.gammamin, obj.tetamin, obj.lammin, obj.hmin, obj.m10], Nleg-1)))
    uplimu = np.tile([obj.alfamax, obj.deltamax], Nleg * NContPoints)
    inflimu = np.tile([obj.alfamin, obj.deltamin], Nleg * NContPoints)

    X0d = np.hstack((X, U, dt))  # vector of initial conditions here all the angles are in degrees!!!!!
    obj.varOld = np.zeros((len(X0d)))

    LbS = inflimx
    UbS = uplimx
    LbC = inflimu
    UbC = uplimu
    tlb = [10]
    tub = [250]

    obj.LBV = np.hstack((LbS, LbC, np.repeat(tlb, Nleg)))
    obj.UBV = np.hstack((UbS, UbC, np.repeat(tub, Nleg)))
    X0a = (X0d - obj.LBV)/(obj.UBV - obj.LBV)

    if solver == 'trust-constr':

        Vlb = np.zeros((len(X0a)))
        Vub = np.ones((len(X0a)))
        bnds = Bounds(Vlb, Vub)

        lbeq = ([0.0])  # lower bound for equality constraints
        ubeq = ([0.0])  # upper bound for equality constraints

        lbineq = ([0.0])  # lower bound for inequality constraints
        ubineq = ([np.inf])  # upper bound for inequality constraints

        lb = lbeq * ((Nstates + Ncontrols) * (Nleg-1) + 4) + lbineq * (4 * NineqCond * Nleg + 3 + 2 * NineqCond)# + NineqCond)  # all lower bounds
        ub = ubeq * ((Nstates + Ncontrols) * (Nleg-1) + 4) + ubineq * (4 * NineqCond * Nleg + 3 + 2 * NineqCond)# + NineqCond)  # all upper bounds
        if save_matrix:
            cons = NonlinearConstraint(constraints, lb, ub, finite_diff_jac_sparsity=None)
        else:
            sparseJac = load_npz(initial_path + "/MultipleShooting_Algorithm/Python/trust-constr_solver/FestipSparsity.npz")
            sp = sparseJac.todense()
            row = np.shape(sp)[0]
            column = np.shape(sp)[1]
            for i in range(row):
                for j in range(column):
                    if sp[i, j] != 0:
                        sp[i, j] = 1

            cons = NonlinearConstraint(constraints, lb, ub, finite_diff_jac_sparsity=sp)
    else:
        bndX_slsqp = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        bndU_slsqp = ((0.0, 1.0), (0.0, 1.0))  # , (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        bndT_slsqp = ((0.0, 1.0),)
        bnds_slsqp = ((0.0, 1.0),) + bndX_slsqp * (Nleg-1) + bndU_slsqp * Nleg * NContPoints + bndT_slsqp * Nleg

        cons_slsqp = ({'type': 'eq',
                       'fun': constraints_slsqp,
                       'args': ("eq",)},
                      {'type': 'ineq',
                       'fun': constraints_slsqp,
                       'args': ("ineq",)})  # equality and inequality constraints

    """ Custom step-function """

    if OptGlobal:
        '''class RandomDisplacementBounds(object):
            """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
                Modified! (dropped acceptance-rejection sampling for a more specialized approach)
            """

            def __init__(self, xmin, xmax, stepsize=0.5):
                self.xmin = xmin
                self.xmax = xmax
                self.stepsize = stepsize

            def __call__(self, x):
                """take a random step but ensure the new position is within the bounds """
                min_step = np.maximum(self.xmin - x, -self.stepsize)
                max_step = np.minimum(self.xmax - x, self.stepsize)

                random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
                xnew = x + random_step

                return xnew
        bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bnds]), np.array([b[1] for b in bnds]))'''

    '''NLP SOLVER'''

    if OptGlobal:
        minimizer_kwargs = {'method':'trust-constr', "constraints":cons, "bounds":bnds}
        opt = basinhopping(cost_fun, X0a, disp=True, minimizer_kwargs=minimizer_kwargs)
    elif solver == 'trust-constr':
        if save_matrix:
            opt = optimize.minimize(cost_fun, X0a,
                                    constraints=cons,
                                    bounds=bnds,
                                    method='trust-constr',
                                    options={"maxiter":1})
        else:
            opt = optimize.minimize(cost_fun, X0a,
                                    constraints=cons,
                                    bounds=bnds,
                                    method='trust-constr',
                                    options={"verbose":2,
                                             "maxiter":1e6})

    else:
        opt = optimize.minimize(cost_fun, X0a,
                                constraints=cons_slsqp,
                                bounds=bnds_slsqp,
                                method='SLSQP',
                                options={"disp": True,
                                         "iprint":2,
                                         "maxiter": 1e6})
    print("Done local search")
    X0a = opt.x
    end = time.time()
    p.close()
    p.join()
    time_elapsed = end-start
    tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
    print("Time elapsed for total optimization ", tformat)
    if save_matrix:
        sparse = csc_matrix(opt.jac[0])
        save_npz(initial_path + "/MultipleShooting_Algorithm/Python/trust-constr_solver/FestipSparsity.npz", sparse)
    if flag_save:
        Plot.plot(opt.x, Nint, Nleg, NContPoints, obj, Nstates, varTot, Ncontrols, varStates, cl, cd, cm, presv, spimpv, flag_save, savedata_file, maxiter, timestr, tformat, savefig_file, states_init)
