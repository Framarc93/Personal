from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Condition, Dynamics, Problem, Guess
import time
from functools import partial
import os
import datetime
import sys
from multiprocessing import Pool
from scipy import special
from scipy import interpolate
from scipy import optimize
from scipy.interpolate import splev, splrep
import scipy.io as sio
from import_initialCond import init_conds
from models_collocation import *


'''this script is a collocation algorithm on FESTIP model'''


class Spaceplane():
    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
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
        #self.k = 5e4  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 100000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))  # value in radians
        self.chistart = np.deg2rad(125)  # deg flight direction
        self.mach = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
        self.angAttack = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0])
        self.bodyFlap = np.array([-20, -10, 0, 10, 20, 30])
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
        self.hvert = 50
        self.n = prob.nodes
        self.gammamax = np.deg2rad(89)
        self.gammamin = np.deg2rad(-50)
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
        self.deltafmax = np.deg2rad(30)
        self.deltafmin = np.deg2rad(-20)
        self.taumax = 1
        self.taumin = -1


def dynamics(prob, obj, section):
    v = prob.states(0, section)
    chi = prob.states(1, section)
    gamma = prob.states(2, section)
    # teta = prob.states(3, section)
    lam = prob.states(4, section)
    h = prob.states(5, section)
    m = prob.states(6, section)
    #time = prob.time_update()
    alfa = prob.controls(0, section)
    #part1 = np.repeat(1.0, int(len(v) * 0.4))
    #part2 = Guess.linear(time[int(len(v) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls(1, section)
    deltaf = prob.controls(2, section)
    tau = prob.controls(3, section)
    mu = np.zeros(len(v)) #prob.controls(4, section)

    Press, rho, c = isa(h, obj, 0)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, rho, m, prob.nodes[0], obj)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], spimp_interp, obj)

    eps = Deps + alfa
    g = np.zeros(prob.nodes[0])
    k = 0
    for alt in h:
        if alt == 0:
            g[k] = obj.g0
            k += 1
        else:
            g[k] = obj.g0 * (obj.Re / (obj.Re + alt)) ** 2  # [m/s2]
            k += 1
    hi = 0
    for i in range(len(h)):
        if h[i] > obj.hvert:
            hi = i-1
            break
    if hi <= 0:
        hi = 1

    dx = Dynamics(prob, section)

    dx[0] = (T[0:hi]*np.cos(eps[0:hi]) - D[0:hi])/m[0:hi] - g[0:hi]
    dx[1] = np.zeros((hi))
    dx[2] = np.zeros((hi))
    dx[3] = np.zeros((hi))
    dx[4] = np.zeros((hi))
    dx[5] = v[0:hi]
    dx[6] = -T[0:hi] / (obj.g0 * isp[0:hi])


    dx[0] = np.hstack((dx[0], (T[hi:] * np.cos(eps[hi:]) - D[hi:]) / m[hi:] - g[hi:] * np.sin(gamma[hi:]) + (obj.omega ** 2) * (obj.Re + h[hi:]) * np.cos(lam[hi:]) * \
            (np.cos(lam[hi:]) * np.sin(gamma[hi:]) - np.sin(lam[hi:]) * np.cos(gamma[hi:]) * np.sin(chi[hi:]))))
    dx[1] = np.hstack((dx[1], ((T[hi:] * np.sin(eps[hi:]) + L[hi:]) * np.sin(mu[hi:])) / (m[hi:] * v[hi:] * np.cos(gamma[hi:])) - np.cos(gamma[hi:]) * np.cos(chi[hi:]) * np.tan(lam[hi:]) \
            * (v[hi:] / (obj.Re + h[hi:])) + 2 * obj.omega * (np.cos(lam[hi:]) * np.tan(gamma[hi:]) * np.sin(chi[hi:]) - np.sin(lam[hi:])) \
            - (obj.omega ** 2) * ((obj.Re + h[hi:]) / (v[hi:] * np.cos(gamma[hi:]))) * np.cos(lam[hi:]) * np.sin(lam[hi:]) * np.cos(chi[hi:])))
    dx[2] = np.hstack((dx[2], ((T[hi:] * np.sin(eps[hi:]) + L[hi:]) * np.cos(mu[hi:])) / (m[hi:] * v[hi:]) - (g[hi:] / v[hi:] - v[hi:] / (obj.Re + h[hi:])) * np.cos(gamma[hi:]) + 2 * obj.omega \
            * np.cos(lam[hi:]) * np.cos(chi[hi:]) + (obj.omega ** 2) * ((obj.Re + h[hi:]) / v[hi:]) * np.cos(lam[hi:]) * \
            (np.sin(lam[hi:]) * np.sin(gamma[hi:]) * np.sin(chi[hi:]) + np.cos(lam[hi:]) * np.cos(gamma[hi:]))))
    dx[3] = np.hstack((dx[3], -np.cos(gamma[hi:]) * np.cos(chi[hi:]) * (v[hi:] / ((obj.Re + h[hi:]) * np.cos(lam[hi:])))))
    dx[4] = np.hstack((dx[4], np.cos(gamma[hi:]) * np.sin(chi[hi:]) * (v[hi:] / (obj.Re + h[hi:]))))
    dx[5] = np.hstack((dx[5], v[hi:] * np.sin(gamma[hi:])))
    dx[6] = np.hstack((dx[6], -T[hi:] / (obj.g0 * isp[hi:])))
    return dx()


def cost(prob, obj):
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)
    #time = prob.time_update()
    #part1 = np.repeat(1.0, int(len(m) * 0.4))
    #part2 = Guess.linear(time[int(len(m) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls_all_section(1)
    tau = prob.controls_all_section(3)

    Press, rho, c = isa(h, obj, 0)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], spimp_interp, obj)

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * isp[-1])))

    return -mf / obj.M0


def equality(prob, obj):
    v = prob.states_all_section(0)
    chi = prob.states_all_section(1)
    gamma = prob.states_all_section(2)
    teta = prob.states_all_section(3)
    lam = prob.states_all_section(4)
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)

    alfa = prob.controls_all_section(0)
    #part1 = np.repeat(1.0, int(len(v) * 0.4))
    #time = prob.time_update()
    #part2 = Guess.linear(time[int(len(v) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls_all_section(1)
    deltaf = prob.controls_all_section(2)
    tau = prob.controls_all_section(3)
    #mu = np.zeros(len(v)) #prob.controls_all_section(4)

    States = np.array((v[-1], chi[-1], gamma[-1], teta[-1], lam[-1], h[-1], m[-1]))
    Controls = np.array((alfa[-1], delta[-1], deltaf[-1], tau[-1]))
    vt = np.sqrt(obj.GMe / (obj.Re + h[-1]))  # - obj.omega*np.cos(lam[-1])*(obj.Re+h[-1])



    vtAbs, chiass, vtAbs2 = vass(States, Controls, dynamicsVel, obj.omega, obj)

    if np.cos(obj.incl) > np.cos(lam[-1]):
        chifin = np.pi
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam[-1]))
    result = Condition()

    # event condition
    result.equal(to_new_int(v[0], obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(obj.vstart, obj.vmin, obj.vmax, 0.0, 1.0), unit=1)      #########   O L D  ##########
    #result.equal(to_new_int(chi[0], obj.chimin, obj.chimax, 0.0, 1.0), to_new_int(obj.chistart, obj.chimin, obj.chimax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(gamma[0], obj.gammamin, obj.gammamax, 0.0, 1.0), to_new_int(obj.gammastart, obj.gammamin, obj.gammamax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(teta[0], obj.tetamin, obj.tetamax, 0.0, 1.0), to_new_int(obj.longstart, obj.tetamin, obj.tetamax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(lam[0], obj.lammin, obj.lammax, 0.0, 1.0), to_new_int(obj.latstart, obj.lammin, obj.lammax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(h[0], obj.hmin, obj.hmax, 0.0, 1.0), to_new_int(obj.hstart, obj.hmin, obj.hmax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(m[0], obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.M0, obj.m10, obj.M0, 0.0, 1.0), unit=1)
    result.equal(delta[0], 1.0, unit=1)
    # result.equal(to_new_int(alfa[0], np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), to_new_int(0.0, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)
    # result.equal(to_new_int(deltaf[0], np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), to_new_int(0.0, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)
    # result.equal(to_new_int(tau[0], -1, 1, 0.0, 1.0), to_new_int(0.0, -1, 1, 0.0, 1.0), unit=1)
    # result.equal(to_new_int(mu[0], np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), to_new_int(0.0, np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), unit=1)
    #result.equal(to_new_int(mu[-1], np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), to_new_int(0.0, np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), unit=1)
    result.equal(to_new_int(vtAbs, obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(vt, obj.vmin, obj.vmax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(chiass, obj.chimin, obj.chimax, 0.0, 1.0), to_new_int(chifin, obj.chimin, obj.chimax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(gamma[-1], obj.gammamin, obj.gammamax, 0.0, 1.0), to_new_int(0.0, obj.gammamin, obj.gammamax, 0.0, 1.0), unit=1)

    '''result.equal(v[0], obj.vstart, unit=obj.vmax)
    result.equal(chi[0], obj.chistart, unit=obj.chimax)
    result.equal(gamma[0], obj.gammastart, unit=obj.gammastart)
    result.equal(teta[0], obj.longstart, unit=obj.tetamax)
    result.equal(lam[0], obj.latstart, unit=obj.lammax)
    result.equal(h[0], obj.hstart, unit=obj.hmax)
    result.equal(m[0], obj.M0, unit=obj.M0)
    result.equal(delta[0], 1.0, unit=obj.deltamax)
    result.equal(vtAbs, vt, unit=obj.vmax)
    result.equal(chiass, chifin, unit=obj.chimax)
    result.equal(gamma[-1], 0.0, unit=obj.gammamax)'''

    return result()


def inequality(prob, obj):
    v = prob.states_all_section(0)
    chi = prob.states_all_section(1)
    gamma = prob.states_all_section(2)
    teta = prob.states_all_section(3)
    lam = prob.states_all_section(4)
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)
    time = prob.time_update()
    alfa = prob.controls_all_section(0)
    #part1 = np.repeat(1.0, int(len(v) * 0.4))
    #part2 = Guess.linear(time[int(len(v) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls_all_section(1)
    deltaf = prob.controls_all_section(2)
    tau = prob.controls_all_section(3)
    #mu = np.zeros(len(v)) #prob.controls_all_section(4)
    #t = prob.time_update()

    Press, rho, c = isa(h, obj, 0)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, rho, m, prob.nodes[0], obj)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], spimp_interp, obj)

    #MomTot = MomA + MomT

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)
    # accelerations
    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * isp[-1])))

    result = Condition()

    # lower bounds
    result.lower_bound(to_new_int(v, obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(obj.vmin, obj.vmin, obj.vmax, 0.0, 1.0), unit=1)  # v lower bound                                   ############### OLD  ###################
    result.lower_bound(to_new_int(chi, obj.chimin, obj.chimax, 0.0, 1.0), to_new_int(obj.chimin, obj.chimin, obj.chimax, 0.0, 1.0), unit=1)  # chi lower bound
    result.lower_bound(to_new_int(gamma, obj.gammamin, obj.gammamax, 0.0, 1.0), to_new_int(obj.gammamin, obj.gammamin, obj.gammamax, 0.0, 1.0), unit=1)  # gamma lower bound
    result.lower_bound(to_new_int(teta, obj.tetamin, obj.tetamax, 0.0, 1.0), to_new_int(obj.tetamin, obj.tetamin, obj.tetamax, 0.0, 1.0), unit=1)  # teta lower bound
    result.lower_bound(to_new_int(lam, obj.lammin, obj.lammax, 0.0, 1.0), to_new_int(obj.lammin, obj.lammin, obj.lammax, 0.0, 1.0), unit=1)  # lambda lower bound
    result.lower_bound(to_new_int(h, obj.hmin, obj.hmax, 0.0, 1.0), to_new_int(obj.hmin, obj.hmin, obj.hmax, 0.0, 1.0), unit=1)  # h lower bound
    result.lower_bound(to_new_int(m, obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.m10, obj.m10, obj.M0, 0.0, 1.0), unit=1)  # m lower bound
    result.lower_bound(to_new_int(alfa, obj.alfamin, obj.alfamax, 0.0, 1.0), to_new_int(obj.alfamin, obj.alfamin, obj.alfamax, 0.0, 1.0), unit=1)  # alpha lower bound
    result.lower_bound(delta, obj.deltamin, unit=1)  # delta lower bound
    result.lower_bound(to_new_int(deltaf, obj.deltafmin, obj.deltafmax, 0.0, 1.0), to_new_int(obj.deltafmin, obj.deltafmin, obj.deltafmax, 0.0, 1.0), unit=1)  # deltaf lower bound
    result.lower_bound(to_new_int(tau, obj.taumin, obj.taumax, 0, 1), to_new_int(obj.taumin, obj.taumin, obj.taumax, 0, 1), unit=1)  # tau lower bound
    # result.lower_bound(to_new_int(mu, np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0),  to_new_int(np.deg2rad(-60), np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), unit=1)  # mu lower bound
    result.lower_bound(to_new_int(mf, obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.m10, obj.m10, obj.M0, 0.0, 1.0), unit=1)  # mf lower bound
    result.lower_bound(to_new_int(h[-1], obj.hmin, obj.hmax, 0.0, 1.0), to_new_int(6e4, obj.hmin, obj.hmax, 0.0, 1.0), unit=1)  # final h lower bound
    # result.lower_bound(to_new_int(MomTot / 1e4, -1e2, 1e2, 0.0, 1.0), to_new_int(-obj.k / 1e4, -1e2, 1e2, 0.0, 1.0), unit=1)  # total moment lower bound
    result.lower_bound(to_new_int(az, -obj.MaxAz, obj.MaxAz, 0.0, 1.0), to_new_int(-obj.MaxAz, -obj.MaxAz, obj.MaxAz, 0.0, 1.0), unit=1)  # ax lower bound
    result.lower_bound(to_new_int(v[-1], obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(6e3, obj.vmin, obj.vmax, 0.0, 1.0),
                       unit=1)  # v lower bound                                   ############### OLD  ###################

    '''result.lower_bound(v, obj.vmin, unit=obj.vmax)  # v lower bound
    result.lower_bound(chi, obj.chimin, unit=obj.chimax)  # chi lower bound
    result.lower_bound(gamma, obj.gammamin, unit=obj.gammamax)  # gamma lower bound
    result.lower_bound(teta, obj.tetamin, unit=obj.tetamax)  # teta lower bound
    result.lower_bound(lam, obj.lammin, unit=obj.lammax)  # lambda lower bound
    result.lower_bound(h, obj.hmin, unit=obj.hmax)  # h lower bound
    result.lower_bound(m, obj.m10, unit=obj.M0)  # m lower bound
    result.lower_bound(alfa, obj.alfamin, unit=obj.alfamax)  # alpha lower bound
    result.lower_bound(delta, obj.deltamin, unit=obj.deltamax)  # delta lower bound
    result.lower_bound(mf, obj.m10, unit=obj.M0)  # mf lower bound
    result.lower_bound(h[-1], 6e4, unit=obj.hmax)  # final h lower bound
    result.lower_bound(az, -obj.MaxAz, unit=obj.MaxAz)  # ax lower bound'''

    # upper bounds
    result.upper_bound(to_new_int(v, obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(obj.vmax, obj.vmin, obj.vmax, 0.0, 1.0),
                       unit=1)  # v upper bound

    result.upper_bound(to_new_int(chi, obj.chimin, obj.chimax, 0.0, 1.0),  # obj.chi_ub, unit=1)
                       to_new_int(obj.chimax, obj.chimin, obj.chimax, 0.0, 1.0),
                       unit=1)  # chi upper bound

    result.upper_bound(to_new_int(gamma, obj.gammamin, obj.gammamax, 0.0, 1.0),
                       to_new_int(obj.gammamax, obj.gammamin, obj.gammamax, 0.0, 1.0),
                       unit=1)  # gamma upper bound

    result.upper_bound(to_new_int(teta, obj.tetamin, obj.tetamax, 0.0, 1.0),
                       to_new_int(obj.tetamax, obj.tetamin, obj.tetamax, 0.0, 1.0), unit=1)  # teta upper bound

    result.upper_bound(to_new_int(lam, obj.lammin, obj.lammax, 0.0, 1.0),
                       to_new_int(obj.lammax, obj.lammin, obj.lammax, 0.0, 1.0), unit=1)  # lam upper bound

    result.upper_bound(to_new_int(h, obj.hmin, obj.hmax, 0.0, 1.0), to_new_int(obj.hmax, obj.hmin, obj.hmax, 0.0, 1.0),
                       unit=1)  # h upper bound

    result.upper_bound(to_new_int(m, obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.M0, obj.m10, obj.M0, 0.0, 1.0),
                       unit=1)  # m upper bound

    result.upper_bound(to_new_int(alfa, obj.alfamin, obj.alfamax, 0.0, 1.0),  # obj.alfa_ub, unit=1)
                       to_new_int(obj.alfamax, obj.alfamin, obj.alfamax, 0.0, 1.0), unit=1)  # alfa upper bound

    result.upper_bound(delta, obj.deltamax, unit=1)  # delta upper bound

    result.upper_bound(to_new_int(deltaf, obj.deltafmin, obj.deltafmax, 0.0, 1.0), to_new_int(obj.deltafmax, obj.deltafmin, obj.deltafmax, 0.0, 1.0), unit=1)  # deltaf upper bound

    result.upper_bound(to_new_int(tau, obj.taumin, obj.taumax, 0, 1), to_new_int(obj.taumax, obj.taumin, obj.taumax, 0, 1), unit=1)  # tau upper bound

    #result.upper_bound(to_new_int(mu, np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0),  # obj.mu_ub, unit=1)
     #                  to_new_int(np.deg2rad(60), np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), unit=1)  # mu upper bound

    #result.upper_bound(to_new_int(MomTot / 1e5, -1e2, 1e2, 0.0, 1.0), to_new_int(obj.k / 1e5, -1e2, 1e2, 0.0, 1.0),
     #                  unit=1)  # momtot upper bound

    result.upper_bound(to_new_int(az, -obj.MaxAz, obj.MaxAz, 0.0, 1.0), to_new_int(obj.MaxAz, -obj.MaxAz, obj.MaxAz, 0.0, 1.0),
                       unit=1)  # az upper bound

    result.upper_bound(to_new_int(ax, 0.0, obj.MaxAx, 0.0, 1.0), to_new_int(obj.MaxAx, 0.0, obj.MaxAx, 0.0, 1.0),
                       unit=1)  # ax upper bound

    result.upper_bound(to_new_int(q, 0.0, obj.MaxQ, 0.0, 1.0), to_new_int(obj.MaxQ, 0.0, obj.MaxQ, 0.0, 1.0),
                       unit=1)  # q upper bound

    '''result.upper_bound(v, obj.vmax, unit=obj.vmax)  # v upper bound
    result.upper_bound(chi, obj.chimax, unit=obj.chimax)  # chi upper bound
    result.upper_bound(gamma, obj.gammamax, unit=obj.gammamax)  # gamma upper bound
    result.upper_bound(teta, obj.tetamax, unit=obj.tetamax)  # teta upper bound
    result.upper_bound(lam, obj.lammax, unit=obj.lammax)  # lam upper bound
    result.upper_bound(h, obj.hmax, unit=obj.hmax)  # h upper bound
    result.upper_bound(m, obj.M0, unit=obj.M0)  # m upper bound
    result.upper_bound(alfa, obj.alfamax, unit=obj.alfamax)  # alfa upper bound
    result.upper_bound(delta, obj.deltamax, unit=obj.deltamax)  # delta upper bound
    result.upper_bound(az, obj.MaxAz, unit=obj.MaxAz)  # az upper bound
    result.upper_bound(ax, obj.MaxAx, unit=obj.MaxAx)  # ax upper bound
    result.upper_bound(q, obj.MaxQ, unit=obj.MaxQ)  # q upper bound'''

    return result()


def dynamicsVel(states, contr, obj):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    v = states[0]
    chi = states[1]
    gamma = states[2]
    # teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = contr[0]
    delta = contr[1]
    deltaf = contr[2]
    tau = contr[3]
    mu = 0.0 #contr[4]

    Press, rho, c = isa(h, obj, 1)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, rho, m, 1, obj)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, 1, spimp_interp, obj)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                           np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(
                       chi),
                ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(
                       gamma) + 2 * obj.omega \
                * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                v * np.sin(gamma),
                -T / (g0 * isp)))
    return dx


if __name__ == '__main__':

    cl = np.array(fileReadOr("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/clfile.txt"))
    cd = np.array(fileReadOr("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cdfile.txt"))
    cm = np.array(fileReadOr("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cmfile.txt"))
    # cl = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cl_smooth_few.npy")
    # cd = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cd_smooth_few.npy")
    # cm = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cm_smooth_few.npy")
    cl = np.reshape(cl, (13, 17, 6))
    # cd = np.asarray(cd)
    cd = np.reshape(cd, (13, 17, 6))
    # cm = np.asarray(cm)
    cm = np.reshape(cm, (13, 17, 6))


    with open("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/impulse.dat") as f:
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
    spimp_interp = splrep(presv, spimpv, s=2)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    flag_savefig = True
    source = 'matlab'
    if source == 'matlab':
        mat_contents = sio.loadmat('/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/workspace_init_cond.mat')
        tfin = mat_contents['t'][0][-1]
    else:
        tfin = np.load('/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_timeTot.npy')[-1]

    pool = Pool(processes=3)
    plt.ion()
    start = time.time()
    n = [50]
    time_init = [0.0, tfin]
    num_states = [7]
    num_controls = [4]
    max_iteration = 1000
    Ncontrols = num_controls[0]
    Nstates = num_states[0]
    Npoints = n[0]
    varStates = Nstates * Npoints
    varTot = (Nstates + Ncontrols) * Npoints
    Nint = 1000
    maxiter = 50
    ftol = 1e-8

    if flag_savefig:
        os.makedirs("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr))
        savefig_file = "/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/Res_".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr)
    # -------------
    # set OpenGoddard class for algorithm determination
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)

    # -------------
    # create instance of operating object
    obj = Spaceplane()

    unit_v = obj.vmax
    unit_chi = obj.chimax
    unit_gamma = obj.gammamax
    unit_teta = obj.tetamax
    unit_lam = obj.lammax
    unit_h = obj.hmax
    unit_m = obj.M0
    unit_t = 700
    unit_alfa = obj.alfamax
    unit_delta = obj.deltamax
    unit_deltaf = obj.deltafmax
    unit_tau = obj.taumax
    #unit_mu = np.deg2rad(60)
    prob.set_unit_states_all_section(0, unit_v)
    prob.set_unit_states_all_section(1, unit_chi)
    prob.set_unit_states_all_section(2, unit_gamma)
    prob.set_unit_states_all_section(3, unit_teta)
    prob.set_unit_states_all_section(4, unit_lam)
    prob.set_unit_states_all_section(5, unit_h)
    prob.set_unit_states_all_section(6, unit_m)
    prob.set_unit_controls_all_section(0, unit_alfa)
    prob.set_unit_controls_all_section(1, unit_delta)
    prob.set_unit_controls_all_section(2, unit_deltaf)
    prob.set_unit_controls_all_section(3, unit_tau)
    #prob.set_unit_controls_all_section(4, unit_mu)
    prob.set_unit_time(unit_t)

    # =================
    # initial parameters guess
    # velocity

    states_init, controls_init = init_conds(prob.time_update(), source)

    v_init = states_init[0]
    chi_init = states_init[1]
    gamma_init = states_init[2]
    teta_init = states_init[3]
    lam_init = states_init[4]
    h_init = states_init[5]
    m_init = states_init[6]
    alfa_init = controls_init[0]
    delta_init = controls_init[1]
    deltaf_init = Guess.zeros(prob.time_all_section)
    tau_init = Guess.zeros(prob.time_all_section)
    '''v_init = Guess.linear(prob.time_all_section, 1, obj.Vtarget)
    chi_init = Guess.linear(prob.time_all_section, obj.chistart, obj.chi_fin)
    gamma_init = Guess.linear(prob.time_all_section, obj.gammastart, 0.0)
    teta_init = Guess.constant(prob.time_all_section, obj.longstart)
    lam_init = Guess.constant(prob.time_all_section, obj.latstart)
    h_init = Guess.linear(prob.time_all_section, 1, obj.Hini)
    m_init = Guess.linear(prob.time_all_section, obj.M0, obj.m10)

    #alfa_init = Guess.zeros(prob.time_all_section)
    part1 = np.repeat(1.0, obj.n32)
    part2 = Guess.linear(prob.time_all_section[obj.n32:], 1.0, 0.001)
    delta_init = np.hstack((part1, part2))'''

    #mu_init = Guess.zeros(prob.time_all_section)

    # ===========
    # Substitution initial value to parameter vector to be optimized
    # non dimensional values (Divide by scale factor)

    prob.set_states_all_section(0, v_init)
    prob.set_states_all_section(1, chi_init)
    prob.set_states_all_section(2, gamma_init)
    prob.set_states_all_section(3, teta_init)
    prob.set_states_all_section(4, lam_init)
    prob.set_states_all_section(5, h_init)
    prob.set_states_all_section(6, m_init)
    prob.set_controls_all_section(0, alfa_init)
    prob.set_controls_all_section(1, delta_init)
    prob.set_controls_all_section(2, deltaf_init)
    prob.set_controls_all_section(3, tau_init)
    #prob.set_controls_all_section(4, mu_init)

    # ========================
    # Main Process
    # Assign problem to SQP solver
    prob.dynamics = [dynamics]
    # prob.knot_states_smooth = []
    prob.cost = cost
    prob.equality = equality
    prob.inequality = inequality


    # prob.cost_derivative = cost_derivative
    # prob.ineq_derivative = ineq_derivative
    # prob.eq_derivative = eq_derivative

    def display_func():

        m = prob.states_all_section(6)
        h = prob.states_all_section(5)
        #time = prob.time_update()
        #part1 = np.repeat(1.0, int(len(m) * 0.4))
        #part2 = Guess.linear(time[int(len(m) * 0.4):], 1.0, 0.0001)
        #delta = np.hstack((part1, part2))
        delta = prob.controls_all_section(1)
        tau = prob.controls_all_section(3)

        tf = prob.time_final(-1)

        Press, rho, c = isa(h, obj, 0)

        T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], spimp_interp, obj)

        # Hohmann transfer mass calculation
        r1 = h[-1] + obj.Re
        Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
        Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
        mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * isp[-1])))

        print("m0          : {0:.5f}".format(m[0]))
        print("m before Ho : {0:.5f}".format(m[-1]))
        print("mf          : {0:.5f}".format(mf))
        print("altitude Hohmann starts: {0:.5f}".format(h[-1]))
        print("final time  : {0:.3f}".format(tf))


    prob.solve(obj, display_func, ftol=ftol, maxiter=maxiter)

    end = time.time()
    time_elapsed = end - start
    tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
    print("Time elapsed:for total optimization ", tformat)


    def dynamicsInt(t, states, alfa_int, delta_int, deltaf_int, tau_int):
        # this functions receives the states and controls unscaled and calculates the dynamics

        v = states[0]
        chi = states[1]
        gamma = states[2]
        # teta = states[3]
        lam = states[4]
        h = states[5]
        m = states[6]
        alfa = alfa_int(t)
        delta = delta_int(t)
        deltaf = deltaf_int(t)
        tau = tau_int(t)
        mu = 0.0 #mu_int(t)

        # if h<0:
        #   print("h: ", h, "gamma: ", gamma, "v: ", v)

        Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 1)

        M = v / c

        L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                    obj.xcg0, obj.xcgf, obj.pref, 1)

        T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, 1, obj.psl, obj.M0, obj.m10,
                                        obj.lRef, obj.xcgf, obj.xcg0)

        eps = Deps + alfa
        g0 = obj.g0

        if h == 0:
            g = g0
        else:
            g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2

        if h < obj.hvert:
            dx = np.array(((T * np.cos(eps) - D) / m - g, 0, 0, 0, 0, v, -T / (g0 * isp)))
        else:
            dx = np.array(
                (((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                 (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                 ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                 np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                         np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                 - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(
                     chi),
                 ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(
                     gamma) + 2 * obj.omega \
                 * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                 (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                 -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                 np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                 v * np.sin(gamma),
                 -T / (g0 * isp)))
        dx = np.reshape(dx, (7,))
        return dx


    def SingleShooting(states, controls, dyn, time, Nint):

        x = np.zeros((Nint, Nstates))

        x[:Nstates] = states  # vector of intial states ready

        # now interpolation of controls

        alfa_Int = interpolate.PchipInterpolator(time, controls[0, :])
        delta_Int = interpolate.PchipInterpolator(time, controls[1, :])
        deltaf_Int = interpolate.PchipInterpolator(time, controls[2, :])
        tau_Int = interpolate.PchipInterpolator(time, controls[3, :])
        #mu_Int = interpolate.PchipInterpolator(time, controls[4, :])

        time_new = np.linspace(0, time[-1], Nint)

        dt = (time_new[1] - time_new[0])

        t = time_new

        # sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int), t_span=[0, time[-1]], y0=x,
        # t_eval=time_new, method='RK45')

        for i in range(Nint - 1):
            # print(i, x[i,:])
            # print(u[i,:])
            k1 = dt * dyn(t[i], x[i, :], alfa_Int, delta_Int, deltaf_Int, tau_Int)
            # print("k1: ", k1)
            k2 = dt * dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int)
            # print("k2: ", k2)
            k3 = dt * dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int)
            # print("k3: ", k3)
            k4 = dt * dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int)
            # print("k4: ", k4)
            x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # vres = sol.y[0, :]
        # chires = sol.y[1, :]
        # gammares = sol.y[2, :]
        # tetares = sol.y[3, :]
        # lamres = sol.y[4, :]
        # hres = sol.y[5, :]
        # mres = sol.y[6, :]
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
        taures = tau_Int(time_new)
        mures = np.zeros(len(time_new)) #mu_Int(time_new)

        return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, mures


    # ======
    # Post Process
    # ------------------------
    # Convert parameter vector to variable

    # integration of collocation results
    # here angles are in radians and not scaled

    v = prob.states_all_section(0)
    chi = prob.states_all_section(1)
    gamma = prob.states_all_section(2)
    teta = prob.states_all_section(3)
    lam = prob.states_all_section(4)
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)
    time = prob.time_update()
    alfa = prob.controls_all_section(0)
    #part1 = np.repeat(1.0, int(len(v) * 0.4))
    #part2 = Guess.linear(time[int(len(v) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls_all_section(1)
    deltaf = prob.controls_all_section(2)
    tau = prob.controls_all_section(3)
    mu = np.zeros(len(v)) #prob.controls_all_section(4)


    Uval = np.array((alfa, delta, deltaf, tau))

    Xinit = np.array((v[0], chi[0], gamma[0], teta[0], lam[0], h[0], m[0]))

    vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures = \
        SingleShooting(Xinit, Uval, dynamicsInt, time, Nint)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 0)

    Pressres, rhores, cres = obj.isa(hres, obj.psl, obj.g0, obj.Re, 0)

    M = v / c

    Mres = vres / cres

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10, obj.xcg0, obj.xcgf, obj.pref, n[0])

    Lres, Dres, MomAres = obj.aeroForces(Mres, alfares, deltafres, cd, cl, cm, vres, obj.wingSurf, rhores, obj.lRef, obj.M0, mres, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, len(vres))

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, n[0], obj.psl, obj.M0, obj.m10, obj.lRef,
                                    obj.xcgf,
                                    obj.xcg0)

    Tres, Depsres, ispres, MomTres = obj.thrust(Pressres, mres, presv, spimpv, deltares, taures, len(vres), obj.psl, obj.M0, obj.m10, obj.lRef,
                                    obj.xcgf,obj.xcg0)

    MomTot = MomA + MomT
    MomTotres = MomAres + MomTres

    g0 = obj.g0
    eps = Deps + alfa

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)
    qres = 0.5 * rhores * (vres ** 2)

    # accelerations

    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    ax_res = (Tres * np.cos(Depsres) - Dres * np.cos(alfares) + Lres * np.sin(alfares)) / mres
    az_res = (Tres * np.sin(Depsres) + Dres * np.sin(alfares) + Lres * np.cos(alfares)) / mres

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * isp)))

    if flag_savefig:
        res = open("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/resColloc_{}_{}.txt".format(os.path.basename(__file__), n[0], maxiter, max_iteration, timestr, os.path.basename(__file__), timestr), "w")

        res.write("Max number Optimization iterations: " + str(max_iteration) + "\n" + "Number of NLP iterations: "
                  + str(maxiter) + "\n" + "v: " + str(v) + "\n" + "Chi: " + str(np.rad2deg(chi)) + "\n" + "Gamma: "
                  + str(np.rad2deg(gamma)) + "\n" + "Teta: " + str(np.rad2deg(teta)) + "\n" + "Lambda: "
                  + str(np.rad2deg(lam)) + "\n" + "Height: " + str(h) + "\n" + "Mass: " + str(m) + "\n" + "mf: "
                  + str(mf) + "\n" + "Objective Function: " + str(-mf / unit_m) + "\n" + "Alfa: " + str(np.rad2deg(alfa))
                  + "\n" + "Delta: " + str(delta) + "\n" + "Delta f: " + str(np.rad2deg(deltaf)) + "\n" + "Tau: "
                  + str(tau) + "\n" + "Mu: " + str(np.rad2deg(mu)) + "\n" + "Eps: " + str(np.rad2deg(eps)) + "\n" + "Lift: "
                  + str(L) + "\n" + "Drag: " + str(D) + "\n" + "Thrust: " + str(T) + "\n" + "Spimp: " + str(
            isp) + "\n" + "c: "
                  + str(c) + "\n" + "Mach: " + str(M) + "\n" + "Time vector: " + str(time) + "\n" + "Press: " + str(Press)
                  + "\n" + "Dens: " + str(rho) + "\n" + "Time elapsed for optimization: " + tformat + "\n")
        res.close()

        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/v".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), vres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/chi".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), chires)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/gamma".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), gammares)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/teta".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), tetares)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/lambda".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), lamres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/h".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), hres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/m".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), mres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/alfa".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), alfa)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/delta".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), delta)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/deltaf".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), deltaf)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/tau".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), tau)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/mu".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), mu)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/timeTot".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), tres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/timeCol".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), time)

    # ------------------------
    # Visualization

    plt.figure(0)
    plt.title("Altitude profile")
    plt.plot(time, h / 1000, marker=".", label="Altitude")
    plt.plot(tres, hres / 1000, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "altitude" + ".png")

    plt.figure(1)
    plt.title("Velocity")
    plt.plot(time, v, marker=".", label="V")
    plt.plot(tres, vres, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "velocity" + ".png")

    plt.figure(2)
    plt.title("Mass")
    plt.plot(time, m, marker=".", label="Mass")
    plt.plot(tres, mres, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "mass" + ".png")

    '''plt.figure(3)
    plt.title("Altitude profile")
    plt.plot(time, h/1000, marker=".", label="Altitude")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "altitudeNONINT" + ".png")

    plt.figure(4)
    plt.title("Velocity")
    plt.plot(time, v, marker=".", label="V")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "velocityNONINT" + ".png")

    plt.figure(5)
    plt.title("Mass")
    plt.plot(time, m, marker=".", label="Mass")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "massNONINT" + ".png")'''

    plt.figure(6)
    plt.title("Acceleration")
    plt.plot(time, ax, marker=".", label="Acc x")
    plt.plot(time, az, marker=".", label="Acc z")
    plt.plot(tres, ax_res, label="Acc x Integration")
    plt.plot(tres, az_res, label="Acc z Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Acceleration [m/s2]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "acceleration" + ".png")

    plt.figure(7)
    plt.title("Throttle profile")
    plt.plot(time, delta, ".", label="Delta")
    plt.plot(tres, deltares, label="Interp")
    plt.plot(time, tau, ".", label="Tau")
    plt.plot(tres, taures, label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" % ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "Throttle" + ".png")

    plt.figure(8)
    plt.title("Angle of attack profile")
    plt.plot(time, np.rad2deg(alfa), ".", label="Alpha")
    plt.plot(tres, np.rad2deg(alfares), label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "angAttack" + ".png")

    plt.figure(9)
    plt.title("Body Flap deflection profile")
    plt.plot(time, np.rad2deg(deltaf), ".", label="Delta f")
    plt.plot(tres, np.rad2deg(deltafres), label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "bdyFlap" + ".png")

    '''plt.figure(10)
    plt.title("Bank angle profile")
    plt.plot(time, np.rad2deg(mu), ".", label="Mu")
    plt.plot(tres, np.rad2deg(mures), label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "Roll" + ".png")'''

    '''
    plt.subplot(3,1,1)

    plt.step(time, delta, marker="o", label="Delta")
    plt.step(time, tau, marker="o", label="Tau")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" % ")
    plt.legend(loc="best")


    plt.subplot(3,1,2)
    plt.plot(time, np.rad2deg(alfa), marker="o", label="Alfa")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")


    plt.subplot(3,1,3)
    plt.plot(time, np.rad2deg(deltaf), marker="o", label="Deltaf")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")


    plt.subplot(2,2,4)
    plt.step(time, np.rad2deg(mu), where='post', marker=".", label="Mu")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    plt.suptitle("Controls profile")
    if flag_savefig:
        plt.savefig(savefig_file + "Commands" + ".png")

    '''

    plt.figure(11)
    plt.title("Trajectory angles")
    plt.plot(time, np.rad2deg(chi), marker=".", label="Chi")
    plt.plot(time, np.rad2deg(gamma), marker=".", label="Gamma")
    plt.plot(time, np.rad2deg(teta), marker=".", label="Theta")
    plt.plot(time, np.rad2deg(lam), marker=".", label="Lambda")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "Angles" + ".png")

    plt.figure(12)
    plt.title("Chi")
    plt.plot(time, np.rad2deg(chi), marker=".", label="Chi")
    plt.plot(tres, np.rad2deg(chires), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "chi" + ".png")

    plt.figure(13)
    plt.title("Gamma")
    plt.plot(time, np.rad2deg(gamma), marker=".", label="Gamma")
    plt.plot(tres, np.rad2deg(gammares), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "Gamma" + ".png")

    plt.figure(14)
    plt.title("Teta")
    plt.plot(time, np.rad2deg(teta), marker=".", label="Theta")
    plt.plot(tres, np.rad2deg(tetares), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "teta" + ".png")

    plt.figure(15)
    plt.title("Lambda")
    plt.plot(time, np.rad2deg(lam), marker=".", label="Lambda")
    plt.plot(tres, np.rad2deg(lamres), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "lam" + ".png")

    plt.figure(16)
    plt.title("Dynamic pressure profile")
    plt.plot(time, q / 1000, marker=".", label="Q")
    plt.plot(tres, qres / 1000, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" kPa ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "dynPress" + ".png")

    plt.figure(17)
    plt.title("Moment")
    plt.plot(time, MomTot / 1000, marker=".", label="Total Moment")
    plt.plot(tres, MomTotres / 1000, label="Integration")
    #plt.axhline(obj.k / 1000, 0, time[-1], color='r')
    #plt.axhline(-obj.k / 1000, 0, time[-1], color='r')
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" kNm ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "MomTot" + ".png")

    plt.figure(19)
    plt.title("Mach profile")
    plt.plot(time, M, marker=".", label="Mach")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "mach" + ".png")

    plt.figure(20)
    plt.title("Lift, Drag and Thrust profile")
    plt.plot(time, L, marker=".", label="Lift")
    plt.plot(time, D, marker=".", label="Drag")
    plt.plot(time, T, marker=".", label="Thrust")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "LDT" + ".png")

    figs = plt.figure(21)
    ax = figs.add_subplot(111, projection='3d')
    ax.plot(np.rad2deg(teta), np.rad2deg(lam), h / 1e3, color='b', label="3d Trajectory")
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_zlabel('Altitude [km]')
    ax.legend()
    if flag_savefig:
        plt.savefig(savefig_file + "traj" + ".png")

    plt.show(block=True)
    pool.close()
    pool.join()
    '''plt.close(0)
    plt.close(1)
    plt.close(2)
    # plt.close(3)
    # plt.close(4)
    # plt.close(5)
    plt.close(6)
    plt.close(7)
    plt.close(8)
    plt.close(9)
    plt.close(10)
    plt.close(11)
    # plt.close(12)
    # plt.close(13)
    # plt.close(14)
    # plt.close(15)
    plt.close(16)
    plt.close(17)
    # plt.close(18)
    plt.close(19)
    plt.close(20)
    plt.close(21)'''