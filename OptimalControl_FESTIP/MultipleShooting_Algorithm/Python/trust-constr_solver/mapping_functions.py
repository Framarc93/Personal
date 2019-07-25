import numpy as np

def to_new_int(t, a, b, c, d):
    ''' this function converts a value from an interval [a, b] to [c, d]
    t = value to be converted
    a = inf lim old interval
    b = sup lim old interval
    c = inf lim new interval
    d = sup lim new interval'''
    return c+((d-c)/(b-a))*(t-a)


def to_orig_int(ft, a, b, c, d):
    ''' this function converts a value from an interval [a, b] to [c, d]
        ft = value to be converted
        a = inf lim old interval
        b = sup lim old interval
        c = inf lim new interval
        d = sup lim new interval'''
    return a + (ft - c) * ((b-a)/(d-c))


def v_toNew(value):
    return to_new_int(value, 0.0, 1e4, 0.0, 1.0)


def v_toOrig(value):
    return to_orig_int(value, 0.0, 1e4, 0.0, 1.0)


def chi_toNew(value):
    return to_new_int(value, np.deg2rad(90.0), np.deg2rad(270.0), 0.0, 1.0)


def chi_toOrig(value):
    return to_orig_int(value, np.deg2rad(90.0), np.deg2rad(270.0), 0.0, 1.0)


def gamma_toNew(value):
    return to_new_int(value, np.deg2rad(-90), np.deg2rad(90.0), 0.0, 1.0)


def gamma_toOrig(value):
    return to_orig_int(value, np.deg2rad(-90), np.deg2rad(90.0), 0.0, 1.0)


def teta_toNew(value):
    return to_new_int(value, np.deg2rad(-90.0), 0.0, 0.0, 1.0)


def teta_toOrig(value):
    return to_orig_int(value, np.deg2rad(-90.0), 0.0, 0.0, 1.0)


def lam_toNew(value, obj):
    return to_new_int(value, -obj.incl, obj.incl, 0.0, 1.0)


def lam_toOrig(value, obj):
    return to_orig_int(value, -obj.incl, obj.incl, 0.0, 1.0)


def h_toNew(value):
    return to_new_int(value, -3e4, 3e5, 0.0, 1.0)


def h_toOrig(value):
    return to_orig_int(value, -3e4, 3e5, 0.0, 1.0)


def m_toNew(value, obj):
    return to_new_int(value, obj.m10, obj.M0, 0.0, 1.0)


def m_toOrig(value, obj):
    return to_orig_int(value, obj.m10, obj.M0, 0.0, 1.0)


def alfa_toNew(value):
    return to_new_int(value, np.deg2rad(-2.0), np.deg2rad(40.0), 0.0, 1.0)


def alfa_toOrig(value):
    return to_orig_int(value, np.deg2rad(-2.0), np.deg2rad(40.0), 0.0, 1.0)


def deltaf_toNew(value):
    return to_new_int(value, np.deg2rad(-20.0), np.deg2rad(30.0), 0.0, 1.0)


def deltaf_toOrig(value):
    return to_orig_int(value, np.deg2rad(-20.0), np.deg2rad(30.0), 0.0, 1.0)


def tau_toNew(value):
    return to_new_int(value, -1.0, 1.0, 0.0, 1.0)


def tau_toOrig(value):
    return to_orig_int(value, -1.0, 1.0, 0.0, 1.0)


def mu_toNew(value):
    return to_new_int(value, np.deg2rad(-90.0), np.deg2rad(90.0), 0.0, 1.0)


def mu_toOrig(value):
    return to_orig_int(value, np.deg2rad(-90.0), np.deg2rad(90.0), 0.0, 1.0)
