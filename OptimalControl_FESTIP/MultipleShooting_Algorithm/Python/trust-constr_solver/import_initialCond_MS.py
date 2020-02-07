import scipy.io as sio
from scipy.interpolate import PchipInterpolator
import numpy as np

def init_conds(t_stat, t_contr, source):
    if source == 'matlab':
        mat_contents = sio.loadmat('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/workspace_init_cond.mat')
        v = mat_contents['vres'].T[0]
        chi = mat_contents['chires'].T[0]
        gamma = mat_contents['gammares'].T[0]
        teta = mat_contents['tetares'].T[0]
        lam = mat_contents['lamres'].T[0]
        h = mat_contents['hres'].T[0]
        m = mat_contents['mres'].T[0]
        alfa = mat_contents['alfares'].T[0]
        delta = mat_contents['deltares'].T[0]
        t = mat_contents['t'][0]
    else:
        v = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_v.npy')
        chi = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_chi.npy')
        gamma = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_gamma.npy')
        teta = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_teta.npy')
        lam = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_lambda.npy')
        h = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_h.npy')
        m = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_m.npy')
        alfa = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_alfa.npy')
        delta = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_delta.npy')
        t = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_timeTot.npy')

    v_Int = PchipInterpolator(t, v)
    v_init = v_Int(t_stat)

    chi_Int = PchipInterpolator(t, chi)
    chi_init = chi_Int(t_stat)

    gamma_Int = PchipInterpolator(t, gamma)
    gamma_init = gamma_Int(t_stat)

    teta_Int = PchipInterpolator(t, teta)
    teta_init = teta_Int(t_stat)

    lam_Int = PchipInterpolator(t, lam)
    lam_init = lam_Int(t_stat)

    h_Int = PchipInterpolator(t, h)
    h_init = h_Int(t_stat)

    m_Int = PchipInterpolator(t, m)
    m_init = m_Int(t_stat)

    alfa_Int_post = PchipInterpolator(t, alfa)
    alfa_init = alfa_Int_post(t_contr)

    delta_Int_post = PchipInterpolator(t, delta)
    delta_init = delta_Int_post(t_contr)

    return [v_init, chi_init, gamma_init, teta_init, lam_init, h_init, m_init], [alfa_init, delta_init]


