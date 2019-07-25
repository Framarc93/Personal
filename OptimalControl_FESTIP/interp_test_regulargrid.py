from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


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

cl = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/clfile.txt")
cd = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/cdfile.txt")
cm = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/cmfile.txt")
cl = np.asarray(cl)
cl = np.reshape(cl, (13, 17, 6))
cd = np.asarray(cd)
cm = np.asarray(cm)
mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0]
bodyFlap = [-20, -10, 0, 10, 20, 30]

interp_function = RegularGridInterpolator((mach, angAttack, bodyFlap), cl)
alfa = np.linspace(-2.0, 40, 200)
deltaf = np.linspace(-20, 30, 200)
coeff = [interp_function([1.8, alfa[i], deltaf[i]]) for i in range(200)]
coeff = np.reshape(coeff, (200,))


fig = plt.figure()
plt.plot(alfa, coeff)
plt.show()




def aeroForces(M, alfa, deltaf, cd, cl, cm):

    mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0]
    bodyFlap = [-20, -10, 0, 10, 20, 30]

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


    cL = coefCalc(cl, M, alfa, deltaf)
    cD = coefCalc(cd, M, alfa, deltaf)
    cM = coefCalc(cm, M, alfa, deltaf)


    return cL, cD, cM


