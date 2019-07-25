import numpy as np
import time
from scipy import interpolate

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

cl = fileReadOr("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/clfile.txt")
cd = fileReadOr("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cdfile.txt")
cm = fileReadOr("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cmfile.txt")
cl = np.asarray(cl)
cd = np.asarray(cd)
cm = np.asarray(cm)

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

start = time.time()

print("Coeff original function")
print(aeroForces(30, 40, -20, cd, cl, cm))

end = time.time()
print("Time elapsed: ", end - start)

def interp(M, alfa, deltaf, cd, cl, cm):
    mach = np.array((0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0))
    angAttack = np.array((-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0))
    bodyFlap = np.array((-20, -10, 0, 10, 20, 30))
    xx, yy = np.meshgrid(bodyFlap, angAttack)
    for i in range(len(mach)-1):
        if M > mach[i] and M <= mach[i+1]:

            Zd1 = cd[17 * i:17 * (i + 1), :] # the coefficients with the lower mach number are extracted
            Zl1 = cl[17 * i:17 * (i + 1), :]
            Zm1 = cm[17 * i:17 * (i + 1), :]

            Zd2 = cd[17 * (i+1):17 * (i + 2), :]  # the coefficients with the higher mach number are extracted
            Zl2 = cl[17 * (i+1):17 * (i + 2), :]
            Zm2 = cm[17 * (i+1):17 * (i + 2), :]


            fd1 = interpolate.interp2d(bodyFlap, angAttack, Zd1)
            fl1 = interpolate.interp2d(bodyFlap, angAttack, Zl1)
            fm1 = interpolate.interp2d(bodyFlap, angAttack, Zm1)

            fd2 = interpolate.interp2d(bodyFlap, angAttack, Zd2)
            fl2 = interpolate.interp2d(bodyFlap, angAttack, Zl2)
            fm2 = interpolate.interp2d(bodyFlap, angAttack, Zm2)

            cd1 = fd1(deltaf, alfa)
            cl1 = fl1(deltaf, alfa)
            cm1 = fm1(deltaf, alfa)

            cd2 = fd2(deltaf, alfa)
            cl2 = fl2(deltaf, alfa)
            cm2 = fm2(deltaf, alfa)

            Cd = np.interp(M, [mach[i], mach[i+1]], [cd1[0], cd2[0]])
            Cl = np.interp(M, [mach[i], mach[i + 1]], [cl1[0], cl2[0]])
            Cm = np.interp(M, [mach[i], mach[i + 1]], [cm1[0], cm2[0]])


        if M > mach[i+1]:
            Zd = cd[17 * i:17 * (i + 1), :]  # the coefficients with the lower mach number are extracted
            Zl = cl[17 * i:17 * (i + 1), :]
            Zm = cm[17 * i:17 * (i + 1), :]

            fd = interpolate.interp2d(bodyFlap, angAttack, Zd)
            fl = interpolate.interp2d(bodyFlap, angAttack, Zl)
            fm = interpolate.interp2d(bodyFlap, angAttack, Zm)

            Cd = fd(deltaf, alfa)
            Cl = fl(deltaf, alfa)
            Cm = fm(deltaf, alfa)

    return Cl, Cd, Cm

start = time.time()

print("Coeff interp")
print(interp(30, 40, -20, cd, cl, cm))

end = time.time()
print("Time elapsed: ", end - start)