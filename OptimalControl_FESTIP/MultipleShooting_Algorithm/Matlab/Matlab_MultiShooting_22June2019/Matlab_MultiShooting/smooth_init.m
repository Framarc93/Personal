function [v_new, chi_new, gamma_new, teta_new, lam_new, h_new, m_new, alfa_new, delta_new, deltaf_new, tau_new, mu_new, tf] = smooth_init(leg, contp)
%res = load('resMod_afterMPAIDEA');

% v = res.vres';
% chi= res.chires';
% gamma = res.gammares';
% teta = res.tetares';
% lam = res.lamres';
% h = res.hres';
% m = res.mres';
% alfa = res.alfares;
% delta = res.deltares;
% deltaf = res.deltafres;
% tau = res.taures;
% mu = res.mures;
% time = res.timeTotal;

v = readNPY("v.npy");
chi= readNPY("chi.npy");
gamma = readNPY("gamma.npy");
teta = readNPY("teta.npy");
lam = readNPY("lam.npy");
h = readNPY("h.npy");
m = readNPY("m.npy");
alfa = readNPY("alfa.npy");
delta = readNPY("delta.npy");
deltaf = readNPY("deltaf.npy");
tau = readNPY("tau.npy");
mu = readNPY("mu.npy");
timeStates = readNPY("timeTot.npy");
timeCont = readNPY("timeCol.npy");

tf = timeStates(end);

time_cont = linspace(0, tf, leg*contp);
time_stat = linspace(0, tf, leg+1);

v_new = pchip(timeStates, v, time_stat);
chi_new = pchip(timeStates, chi, time_stat);
gamma_new = pchip(timeStates, gamma, time_stat);
teta_new = pchip(timeStates, teta, time_stat);
lam_new = pchip(timeStates, lam, time_stat);
h_new = pchip(timeStates, h, time_stat);
m_new = pchip(timeStates, m, time_stat);

alfa_new = pchip(timeCont, alfa, time_cont);
delta_new = pchip(timeCont, delta, time_cont);
deltaf_new = pchip(timeCont, deltaf, time_cont);
tau_new = pchip(timeCont, tau, time_cont);
mu_new = pchip(timeCont, mu, time_cont);

% figure(1)
% plot(time_stat, v_new)
% figure(2)
% plot(time_stat, chi_new)
% figure(3)
% plot(time_stat, gamma_new)
% figure(4)
% plot(time_stat, teta_new)
% figure(5)
% plot(time_stat, lam_new)
% figure(6)
% plot(time_stat, h_new)
% figure(7)
% plot(time_stat, m_new)
% figure(8)
% plot(time_cont, alfa_new)
% figure(9)
% plot(time_cont, delta_new)
% figure(10)
% plot(time_cont, deltaf_new)
% figure(11)
% plot(time_cont, tau_new)
% figure(12)
% plot(time_cont, mu_new)


end

