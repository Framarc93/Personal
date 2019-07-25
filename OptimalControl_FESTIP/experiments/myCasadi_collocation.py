import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = 5.2  # deg latitude
        self.longstart = -52.775  # deg longitude
        self.chistart = 113  # deg flight direction
        self.incl = 51.6  # deg orbit inclination
        self.gammastart = 89.9  # deg
        self.M0 = 450400  # kg  starting mass
        self.g0 = 9.80665  # m/s2
        self.gIsp = self.g0 * 455 # g0 * Isp max
        self.omega = 7.2921159e-5
        self.MaxQ = 40000  # Pa
        self.MaxAx = 30  # m/s2
        self.MaxAz = 15  # m/s2
        self.Htarget = 400000  # m target height after hohmann transfer
        self.wingSurf = 500.0  # m2
        self.lRef = 34.0  # m
        self.k = 5000  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 180000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(np.deg2rad(self.incl)) / np.cos(np.deg2rad(self.latstart)))

    @staticmethod
    def isa(altitude, pstart, g0, r):
        t0 = 288.15
        p0 = pstart
        prevh = 0.0
        R = 287.00
        m0 = 28.9644
        Rs = 8314.32
        m0 = 28.9644
        temperature = []
        pressure = []
        tempm = []
        density = []
        csound = []

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
                #print("h < 0", alt)
                t = t0
                p = p0
                d = p / (R * t)
                c = np.sqrt(1.4 * R * t)
                density.append(d)
                csound.append(c)
                temperature.append(t)
                pressure.append(p)
                tempm.append(t)
            elif 0 <= alt < 90000:

                for i in range(0, 8):

                    if alt <= hv[i]:
                        t, p = cal(p0, t0, a[i], prevh, alt)
                        d = p / (R * t)
                        c = np.sqrt(1.4 * R * t)
                        density.append(d)
                        csound.append(c)
                        temperature.append(t)
                        pressure.append(p)
                        tempm.append(t)
                        t0 = 288.15
                        p0 = pstart
                        prevh = 0
                        break
                    else:

                        t0, p0 = cal(p0, t0, a[i], prevh, hv[i])
                        prevh = hv[i]

            elif 90000 <= alt <= 190000:
                t, p, tpm = atm90(a90, alt, h90, tcoeff1, pcoeff, tcoeff2, tmcoeff)
                temperature.append(t)
                pressure.append(p)
                tempm.append(tpm)
                d = p / (R * t)
                c = np.sqrt(1.4 * R * tpm)
                density.append(d)
                csound.append(c)
            elif alt > 190000:
                #print("h > 190 km", alt)
                zb = h90[6]
                z = h90[-1]
                b = zb - tcoeff1[6] / a90[6]
                t = tcoeff1[6] + (tcoeff2[6] * (z - zb)) / 1000
                tm = tmcoeff[6] + a90[6] * (z - zb) / 1000
                add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                p = pcoeff[6] * np.exp(-m0 / (a90[6] * Rs) * g0 * r ** 2 * (add1 - add2))
                temperature.append(t)
                pressure.append(p)
                tempm.append(tm)
                d = p / (R * t)
                c = np.sqrt(1.4 * R * tm)
                density.append(d)
                csound.append(c)

        return pressure, density, csound

    @staticmethod
    def aeroForces(M, alfa, deltaf, cd, cl, cm, v, sup, rho, leng, mstart, mass, m10, xcg0, xcgf, pref, npoint):
        def limCalc(array, value):
            j = 0
            lim = array.__len__()
            for num in array:
                if j == lim - 1:
                    sup = num
                    inf = array[j - 1]
                if value < num:
                    sup = num
                    if j == 0:
                        inf = num
                    else:
                        inf = array[j - 1]
                    break
                j += 1
            s = array.index(sup)
            i = array.index(inf)
            return i, s

        def coefCalc(coeff, m, alfa, deltaf):
            im, sm = limCalc(mach, m)  # moments boundaries and determination of the 2 needed tables
            cnew1 = coeff[17 * im: 17 * im + angAttack.__len__()][:]
            cnew2 = coeff[17 * sm: 17 * sm + angAttack.__len__()][:]

            ia, sa = limCalc(angAttack, alfa)  # angle of attack boundaries

            idf, sdf = limCalc(bodyFlap, deltaf)  # deflection angle boundaries

            '''interpolation on the first table between angle of attack and deflection'''
            rowinf1 = cnew1[ia][:]
            rowsup1 = cnew1[sa][:]
            coeffinf = [rowinf1[idf], rowsup1[idf]]
            coeffsup = [rowinf1[sdf], rowsup1[sdf]]
            c1 = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinf)
            c2 = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsup)
            coeffd1 = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])

            '''interpolation on the first table between angle of attack and deflection'''
            rowinf2 = cnew2[ia][:]
            rowsup2 = cnew2[sa][:]
            coeffinf = [rowinf2[idf], rowsup2[idf]]
            coeffsup = [rowinf2[sdf], rowsup2[sdf]]
            c1 = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinf)
            c2 = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsup)
            coeffd2 = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])

            '''interpolation on the moments to obtain final coefficient'''
            coeffFinal = np.interp(m, [mach[im], mach[sm]], [coeffd1, coeffd2])

            return coeffFinal

        alfag = np.rad2deg(alfa)
        deltafg = np.rad2deg(deltaf)
        L = []
        D = []
        Mom = []
        for i in range(npoint):
            cL = coefCalc(cl, M[i], alfag[i], deltafg[i])
            cD = coefCalc(cd, M[i], alfag[i], deltafg[i])
            l = 0.5 * (v[i] ** 2) * sup * rho[i] * cL
            d = 0.5 * (v[i] ** 2) * sup * rho[i] * cD
            xcg = leng * (((xcgf - xcg0) / (m10 - mstart)) * (mass[i] - mstart) + xcg0)
            Dx = xcg - pref
            cM1 = coefCalc(cm, M[i], alfag[i], deltafg[i])
            cM = cM1 + cL * (Dx / leng) * np.cos(alfa[i]) + cD * (Dx / leng) * np.cos(alfa[i])
            mom = 0.5 * (v[i] ** 2) * sup * leng * rho[i] * cM
            L.append(l)
            D.append(d)
            Mom.append(mom)

        return L, D, Mom

    @staticmethod
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

            dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7*slpres - presamb[j])

            mommot = tau[j] * dthr

            thrz = -tau[j] * (2.5e6 - 22 * slpres + 9.92 * presamb[j])
            thrust = np.sqrt(thrx ** 2 + thrz ** 2)
            deps = np.arctan(thrz / thrx)
            Thrust.append(thrust)
            Deps.append(deps)
            Simp.append(spimp)
            Mom.append(mommot)
        return Thrust, Deps, Simp, Mom


obj = Spaceplane()

a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0]
bodyFlap = [-20, -10, 0, 10, 20, 30]

'''function to read data from txt file'''

def fileRead(filename):
    with open(filename) as f:
        table = []
        for line in f:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                table.append(line)

    table = table[:][:]
    par = [[x[0], x[1], x[2], x[3], x[4], x[5]] for x in table]
    f.close()
    return par


def fileReadOr(filename):
    with open(filename) as f:
        table = []
        for line in f:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                table.append(line)

    table = table[1:][:]
    par = [[x[1], x[2], x[3], x[4], x[5], x[6]] for x in table]
    f.close()
    return par

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


# Degree of interpolating polynomial
d = 3

# Get collocation points
tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

# Coefficients of the collocation equation
C = np.zeros((d+1,d+1))

# Coefficients of the continuity equation
D = np.zeros(d+1)

# Coefficients of the quadrature function
B = np.zeros(d+1)

# Construct polynomial basis
for j in range(d+1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    p = np.poly1d([1])
    for r in range(d+1):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(p)
    for r in range(d+1):
        C[j,r] = pder(tau_root[r])

    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = np.polyint(p)
    B[j] = pint(1.0)

# Time horizon
T = 400.

# Declare model variables
v = ca.SX.sym('v')
chi = ca.SX.sym('chi')
gamma = ca.SX.sym('gamma')
teta = ca.SX.sym('teta')
lam = ca.SX.sym('lam')
h = ca.SX.sym('h')
m = ca.SX.sym('m')
x = ca.vertcat(v, chi, gamma, teta, lam, h, m)
alfa = ca.SX.sym('alfa')
delta = ca.SX.sym('delta')
deltaf = ca.SX.sym('deltaf')
tau = ca.SX.sym('tau')
u = ca.vertcat(alfa, delta, deltaf, tau)

Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)

Press = np.asarray(Press, dtype=np.float64)
rho = np.asarray(rho, dtype=np.float64)
c = np.asarray(c, dtype=np.float64)

M = v / c

L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, prob.nodes[0])
L = np.asarray(L, dtype=np.float64)
D = np.asarray(D, dtype=np.float64)
MomA = np.asarray(MomA, dtype=np.float64)

Tt, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)
Tt = np.asarray(Tt, dtype=np.float64)
isp = np.asarray(isp, dtype=np.float64)
Deps = np.asarray(Deps, dtype=np.float64)
MomT = np.asarray(MomT, dtype=np.float64)

MomTot = MomA + MomT

eps = Deps + alfa
g0 = obj.g0
g = []

for alt in h:
    if alt == 0:
        g.append(g0)
    else:
        g.append(obj.g0 * (obj.Re / (obj.Re + alt)) ** 2)  # [m/s2]
g = np.asarray(g, dtype=np.float64)

# Model equations
xdot = ca.vertcat(((Tt * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega**2) * (obj.Re + h) * np.cos(lam) * \
            (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
            ((Tt * np.sin(eps) + L) / (m * v * np.cos(gamma))) - np.cos(gamma) * np.cos(chi) * np.tan(lam) \
            * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam))\
            - (obj.omega ** 2) * ((obj.Re + h)/(v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
            ((Tt * np.sin(eps) + L) / (m * v)) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
            * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
            (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
            -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
            np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
            v * np.sin(gamma),
            -Tt / (g0 * isp))

# Objective term

r1 = h + obj.Re
Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))

L = -mf/obj.M0

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])

# Control discretization
N = 20 # number of control intervals
h = T/N

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# For plotting x and u given w
x_plot = []
u_plot = []

# "Lift" initial conditions
Xk = ca.MX.sym('X0', 11)
w.append(Xk)
lbw.append([0.0, np.deg2rad(90), 0.0, ])
ubw.append([0, 1])
w0.append([0.0, np.deg2rad(obj.chistart), np.deg2rad(obj.gammastart), np.deg2rad(obj.longstart), np.deg2rad(obj.latstart),
           0.0, obj.M0])
x_plot.append(Xk)

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym('U_' + str(k))
    w.append(Uk)
    lbw.append([-1])
    ubw.append([1])
    w0.append([0])
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), 2)
        Xc.append(Xkj)
        w.append(Xkj)
        lbw.append([-0.25, -np.inf])
        ubw.append([np.inf,  np.inf])
        w0.append([0, 0])

    # Loop over collocation points
    Xk_end = D[0]*Xk
    for j in range(1,d+1):
       # Expression for the state derivative at the collocation point
       xp = C[0,j]*Xk
       for r in range(d): xp = xp + C[r+1,j]*Xc[r]

       # Append collocation equations
       fj, qj = f(Xc[j-1],Uk)
       g.append(h*fj - xp)
       lbg.append([0, 0])
       ubg.append([0, 0])

       # Add contribution to the end state
       Xk_end = Xk_end + D[j]*Xc[j-1];

       # Add contribution to quadrature function
       J = J + B[j]*qj*h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym('X_' + str(k+1), 2)
    w.append(Xk)
    lbw.append([-0.25, -np.inf])
    ubw.append([np.inf,  np.inf])
    w0.append([0, 0])
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end-Xk)
    lbg.append([0, 0])
    ubg.append([0, 0])

# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)
w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', prob);

# Function to get x and u trajectories from w
trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

# Plot the result
tgrid = np.linspace(0, T, N+1)
plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt[0], '--')
plt.plot(tgrid, x_opt[1], '-')
plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()