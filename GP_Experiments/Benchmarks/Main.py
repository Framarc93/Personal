from Benchmarks import Benchmarks
from GP_selector import GP_choice
import time
from statistics import median
from scoop import futures
import multiprocessing
### choose method ###
'''    LIST OF BENCHMARKS
- keijzer 1: f = 0.3 * x * np.sin(2*np.pi*x)
        interval = (-1, 1, 21)
- keijzer 2: f = 0.3 * x * np.sin(2*np.pi*x)
        interval = (-2, 2, 21)
- keijzer 3: f = 0.3 * x * np.sin(2*np.pi*x)
        interval = (-3, 3, 21)       
- keijzer 4: f = x**3 + np.exp(-x) * np.cos(x) * np.sin(x) * (np.sin(x)**2 * np.cos(x) - 1)
        interval = (0, 10, 11)
- keijzer 5: f = (30 * x * z)/((x - 10) * y**2)
        interval1 = (-1, 1, 1000)
        interval2 = interval1
        interval3 = (1, 2, 1000)
- keijzer 6: f = [sum(i) for i in range(x)]
        interval = (1, 50, 50)
- keijzer 7: f = np.log(x)
        interval = (1, 100, 100)
- keijzer 8: f = np.sqrt(x)
        interval = (0, 100, 101)
- keijzer 10: f = x**y
        interval1 = np.linspace(0, 1, 100)
        interval2 = interval1
- keijzer 11: f = x * y + np.sin((x - 1) * (y - 1))
        interval1 = np.linspace(-3, 3, 20)
        interval2 = interval1
- keijzer 13: f = 6 * np.sin(x) * np.cos(y)
        interval1 = np.linspace(-3, 3, 20)
        interval2 = interval1
- keijzer 14: f = 8 / (2 + x**2 + y**2)
        interval1 = np.linspace(-3, 3, 20)
        interval2 = np.linspace(-3, 3, 20)
- keijzer 15: f = x**3/5 + y**3/2 - y - x
        interval1 = np.linspace(-3, 3, 20)
        interval2 = np.linspace(-3, 3, 20)
- korns 1: f = 1.57 + 24.3 * x
        interval = np.linspace(-50, 50, 10000)
- korns 2: f = 0.23 + 14.2 * ((x + y) / 3 * z)
        interval1 = np.linspace(-50, 50, 10000)
        interval2 = interval1
        interval3 = interval1
- korns 3: f = -5.41 + 4.9 * ((z - x + y/w) / (3 * w))
        interval1 = np.linspace(-50, 50, 10000)
        interval2 = interval1
        interval3 = interval1
- korns 4: f = -2.3 + 0.13 * np.sin(x)
        interval1 = np.linspace(-50, 50, 10000)
        interval2 = interval1
        interval3 = interval1
'''


if __name__ == "__main__":
    GP = ["GP", "MGGP", "MGGP+SNGP"]
    #GP = ["MGGP+SNGP"]
    ### choose benchmark ###

    benchmark = "korns 3"
    Ngen = 10
    nEph = 3
    Nevals = 10
    best_fit_gp = []
    best_ind_gp = []
    best_fit_mggp = []
    best_ind_mggp = []
    best_fit_sngp = []
    best_ind_sngp = []
    best_fit_mggpsngp = []
    best_ind_mggpsngp = []
    teval_gp = []
    teval_mggp = []
    teval_sngp = []
    teval_mggpsngp = []

    for ev in range(Nevals):
        for gp in GP:
            eval_fun, interval, terminals = Benchmarks(benchmark)

            start = time.time()
            bf, bi = GP_choice(gp, eval_fun, interval, Ngen, nEph, terminals, ev)
            end = time.time()

            teval = end - start
            if gp == "GP":
                best_fit_gp.append(bf)
                best_ind_gp.append(bi)
                teval_gp.append(teval)
            elif gp == "MGGP":
                best_fit_mggp.append(bf)
                best_ind_mggp.append(bi)
                teval_mggp.append(teval)
            elif gp == "SNGP":
                best_fit_sngp.append(bf)
                best_ind_sngp.append(bi)
                teval_sngp.append(teval)
            elif gp == "MGGP+SNGP":
                best_fit_mggpsngp.append(bf)
                best_ind_mggpsngp.append(bi)
                teval_mggpsngp.append(teval)

    median_fit_gp = median(best_fit_gp)
    median_time_gp = median(teval_gp)
    median_fit_mggp = median(best_fit_mggp)
    median_time_mggp = median(teval_mggp)
    #median_fit_sngp = median(best_fit_sngp)
    #median_time_sngp = median(teval_sngp)
    median_fit_mggpsngp = median(best_fit_mggpsngp)
    median_time_mggpsngp = median(teval_mggpsngp)
    print("\n")
    print("---------------------------    GP    ---------------------------------")
    print("Median evaluation time of GP {} s".format(median_time_gp))
    print("Median fitness value of GP {} ".format(median_fit_gp))
    print("-------------------------    MGGP    ---------------------------------")
    print("Median evaluation time of MGGP {} s".format(median_time_mggp))
    print("Median fitness value of MGGP {} ".format(median_fit_mggp))
    #print("Median evaluation time of SNGP {} s".format(median_time_sngp))
    #print("Median fitness value of SNGP {} ".format(median_fit_sngp))
    print("----------------------    MGGP + SNGP    ------------------------------")
    print("Median evaluation time of MGGP + SNGP {} s".format(median_time_mggpsngp))
    print("Median fitness value of MGGP + SNGP {} ".format(median_fit_mggpsngp))
    print("-----------------------------------------------------------------------")



