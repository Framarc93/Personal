import scipy.io as sio
from scipy.interpolate import PchipInterpolator
import numpy as np

def init_conds(time_new, source):
    if source == 'matlab':
        mat_contents = sio.loadmat('/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/workspace_init_cond.mat')
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
        alfa_Int_post = PchipInterpolator(t, alfa)
        alfa_init = alfa_Int_post(time_new)

        delta_Int_post = PchipInterpolator(t, delta)
        delta_init = delta_Int_post(time_new)
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
        t_contr = np.linspace(0.0, t[-1], len(alfa))
        alfa_Int_post = PchipInterpolator(t_contr, alfa)
        alfa_init = alfa_Int_post(time_new)
        delta_Int_post = PchipInterpolator(t_contr, delta)
        delta_init = delta_Int_post(time_new)

    v_Int = PchipInterpolator(t, v)
    v_init = v_Int(time_new)

    chi_Int = PchipInterpolator(t, chi)
    chi_init = chi_Int(time_new)

    gamma_Int = PchipInterpolator(t, gamma)
    gamma_init = gamma_Int(time_new)

    teta_Int = PchipInterpolator(t, teta)
    teta_init = teta_Int(time_new)

    lam_Int = PchipInterpolator(t, lam)
    lam_init = lam_Int(time_new)

    h_Int = PchipInterpolator(t, h)
    h_init = h_Int(time_new)

    m_Int = PchipInterpolator(t, m)
    m_init = m_Int(time_new)


    return [v_init, chi_init, gamma_init, teta_init, lam_init, h_init, m_init], [alfa_init, delta_init]


