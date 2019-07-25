function out = Plot_MS(var, prob, obj, file)

time = zeros(1);
timeTotal = zeros(0);
alfaCP = zeros(prob.Nleg, prob.NContPoints);
deltaCP = zeros(prob.Nleg, prob.NContPoints);
deltafCP = zeros(prob.Nleg, prob.NContPoints);
tauCP = zeros(prob.Nleg, prob.NContPoints);
muCP = zeros(prob.Nleg, prob.NContPoints);
vtot = [];
chitot = [];
gammatot = [];
tetatot = [];
lamtot = [];
htot = [];
mtot = [];
alfatot = [];
deltatot = [];
deltaftot = [];
tautot = [];
mutot = [];
ttot = [];

timestart = 0;
res = fopen("res_MS_Matlab.txt", "w");
tCtot = zeros((0));
varD = var.*(prob.UBV - prob.LBV) + prob.LBV;
for i=1:prob.Nleg
    alfa = zeros((prob.NContPoints));
    delta = zeros((prob.NContPoints));
    deltaf = zeros((prob.NContPoints));
    tau = zeros((prob.NContPoints));
    mu = zeros((prob.NContPoints));
    controls = extract_controls(varD(prob.varStates+1+(prob.Ncontrols*prob.NContPoints)*(i-1):prob.varStates+i*prob.Ncontrols*prob.NContPoints), prob.NContPoints, prob.Ncontrols);
    states = varD(i*prob.Nstates-(prob.Nstates-1):i*prob.Nstates);  % orig intervals
    timeend = timestart + varD(prob.varTot + i);
    timeTotal = linspace(timestart, timeend, prob.Nint);
    time = [time, timeend];
    tC = linspace(timestart, timeend, prob.NContPoints);
    tCtot = [tCtot, tC];
    
    [vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, ...
    taures, mures, alfa_I,  delta_I, deltaf_I, tau_I, mu_I] = SingleShooting(states, controls, timestart, timeend, prob, obj, file);
    timestart = timeend;
    alfaCP(i,:)=controls(1,:);
    deltaCP(i,:) = controls(2,:);
    deltafCP(i,:) = controls(3,:);
    tauCP(i,:) = controls(4,:);
    muCP(i,:) = controls(5,:);
%     vres = obj.States((i-1) * prob.Nint+1:(i) * prob.Nint, 1);
%     chires = obj.States(i * prob.Nint:(i + 1) * prob.Nint, 2);
%     gammares = obj.States(i * prob.Nint:(i + 1) * prob.Nint, 3);
%     tetares = obj.States(i * prob.Nint:(i + 1) * prob.Nint, 4);  % teta to [-90,0]
%     lamres = obj.States(i * prob.Nint:(i + 1) * prob.Nint, 5);  % lam to [-incl, incl]
%     hres = obj.States(i * prob.Nint:(i + 1) * prob.Nint, 6);
%     mres = obj.States(i * prob.Nint:(i + 1) * prob.Nint, 7);
% 
%     alfares = obj.Controls(i * prob.Nint:(i + 1) * prob.Nint, 1);
%     deltares = obj.Controls(i * prob.Nint:(i + 1) * prob.Nint, 2);
%     deltafres = obj.Controls(i * prob.Nint:(i + 1) * prob.Nint, 3);
%     taures = obj.Controls(i * prob.Nint:(i + 1) * prob.Nint, 4);  % tau to (-1,1]
%     mures = obj.Controls(i * prob.Nint:(i + 1) * prob.Nint, 5);
    
    

   [Press, rho, c] = isaMulti(hres', obj);

M = vres' ./ c;
varnum=prob.Nint;

[L, D, MomA] = aeroForcesMulti(M, alfares, deltafres, file.cd, file.cl, file.cm, vres', rho, mres', obj, varnum);

[T, Deps, isp, MomT] = thrustMulti(Press, mres', file.presv, file.spimpv, deltares, taures, obj, varnum);

MomTot = MomA + MomT;


% dynamic pressure

q = 0.5 * rho .* (vres' .^ 2);

% accelerations

ax = (T .* cos(Deps) - D .* cos(alfares) + L .* sin(alfares)) ./ mres';
az = (T .* sin(Deps) + D .* sin(alfares) + L .* cos(alfares)) ./ mres';

r1 = hres(end) + obj.Re;
Dv1 = sqrt(obj.GMe / r1) * (sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1);
Dv2 = sqrt(obj.GMe / obj.r2) * (1 - sqrt((2 * r1) / (r1 + obj.r2)));
mf = mres(end) / exp((Dv1 + Dv2) / (obj.g0*isp(end)));


vtot = [vtot, vres'];
chitot = [chitot, chires'];
gammatot = [gammatot, gammares'];
tetatot = [tetatot, tetares'];
lamtot = [lamtot, lamres'];
htot = [htot, hres'];
mtot = [mtot, mres'];
alfatot = [alfatot, alfares];
deltatot = [deltatot, deltares];
deltaftot = [deltaftot, deltafres];
tautot = [tautot, taures];
mutot = [mutot, mures];
ttot = [ttot, tres];

%     g0 = obj.g0;
%     eps = Deps + alfares;
%     g = []
%     for alt=hres(1):hres(end)
%         if alt == 0
%             g.append(g0)
%         else
%             g = [g, (obj.g0 * (obj.Re / (obj.Re + alt)) ^ 2)]
%         end
%     end
%     
   
   
%     fprintf(res, "Number of leg: " + str(Nleg) + "\n" + "Max number Optimization iterations: " + str(maxIterator) + "\n" ...
%         + "Number of NLP iterations: " + str(maxiter) + "\n" + "Leg Number:" + str(i) + "\n" + "v: " + str( ...
%         vres) + "\n" + "Chi: " + str(rad2deg(chires)) ...
%         + "\n" + "Gamma: " + str(rad2deg(gammares)) + "\n" + "Teta: " + str( ...
%         rad2deg(tetares)) + "\n" + "Lambda: " ...
%         + str(rad2deg(lamres)) + "\n" + "Height: " + str(hres) + "\n" + "Mass: " + str( ...
%         mres) + "\n" + "mf: " + str(mf) + "\n" ...
%         + "Objective Function: " + str(-mf / obj.M0) + "\n" + "Alfa: " ...
%         + str(rad2deg(alfares)) + "\n" + "Delta: " + str(deltares) + "\n" + "Delta f: " + str( ...
%         rad2deg(deltafres)) + "\n" ...
%         + "Tau: " + str(taures) + "\n" + "Eps: " + str(rad2deg(eps)) + "\n" + "Lift: " ...
%         + str(L) + "\n" + "Drag: " + str(D) + "\n" + "Thrust: " + str(T) + "\n" + "Spimp: " + str( ...
%         isp) + "\n" + "c: " ...
%         + str(c) + "\n" + "Mach: " + str(M) + "\n" + "Time vector: " + str(timeTotal) + "\n" + "Press: " + str( ...
%         Press) + "\n" + "Dens: " + str(rho) + "\n" + "Time elapsed during optimization: " + tformat);

%     downrange = - (vres ^ 2) / g * sin(2 * chires);

    figure(100000)
    hold on
    title("Velocity");
    plot(timeTotal, vres)
    ylabel("m/s");
    xlabel("time [s]");
    grid on


    figure(1)
    hold on
    title("Heading");
    plot(timeTotal, rad2deg(chires))
    ylabel("deg");
    xlabel("time [s]");
    grid on


    figure(2)
    hold on
    title("Flight path angle");
    plot(timeTotal, rad2deg(gammares))
    ylabel("deg");
    xlabel("time [s]");
    grid on

    figure(3)
    hold on
    title("Longitude");
    plot(timeTotal, rad2deg(tetares))
    ylabel("deg");
    xlabel("time [s]");
    grid on

    figure(4)
    hold on
    title("Latitude");
    plot(timeTotal, rad2deg(lamres))
    ylabel("deg");
    xlabel("time [s]");
    grid on


    figure(5)
    hold on
    title("Flight angles");
    plot(timeTotal, rad2deg(chires), 'g')
    plot(timeTotal, rad2deg(gammares), 'b')
    plot(timeTotal, rad2deg(tetares), 'r')
    plot(timeTotal, rad2deg(lamres), 'k')
    ylabel("deg");
    xlabel("time [s]");
    legend("Chi", "Gamma", "Theta", "Lambda");
    grid on


    figure(6)
    hold on
    title("Altitude");
    plot(timeTotal, hres / 1000)
    ylabel("km");
    xlabel("time [s]");
    grid on

    figure(7)
    hold on
    title("Mass");
    plot(timeTotal, mres)
    ylabel("kg");
    xlabel("time [s]");
    grid on

    figure(8)
    hold on
    title("Angle of attack");
    plot(tC, rad2deg(alfaCP(i, :)), 'ro')
    plot(timeTotal, rad2deg(alfares));
    ylabel("deg");
    xlabel("time [s]");
    legend(["Control points"]);
    grid on


    figure(9)
    hold on
    title("Throttles");
    plot(timeTotal, deltares * 100, 'r')
    plot(timeTotal, taures * 100, 'k')
    plot(tC, deltaCP(i, :) * 100, 'ro')
    plot(tC, tauCP(i, :) * 100, 'ro')
    ylabel("%");
    xlabel("time [s]");
    legend("Delta", "Tau", "Control points");
    grid on

    figure(10)
    hold on
    title("Body Flap deflection");
    plot(tC, rad2deg(deltafCP(i, :)), "ro")
    plot(timeTotal, rad2deg(deltafres));
    ylabel("deg");
    xlabel("time [s]");
    legend(["Control points"]);
    grid on

    figure(11)
    hold on
    title("Bank angle profile");
    plot(tC, rad2deg(muCP(i, :)), "ro")
    plot(timeTotal, rad2deg(mures))
    ylabel("deg");
    xlabel("time [s]");
    legend(["Control points"]);
    grid on


    figure(12)
    hold on
    title("Dynamic Pressure");
    plot(timeTotal, q / 1000)
    ylabel("kPa");
    xlabel("time [s]");
    grid on


    figure(13)
    hold on
    title("Accelerations");
    plot(timeTotal, ax, 'b')
    plot(timeTotal, az, 'r')
    ylabel("m/s^2");
    xlabel("time [s]");
    legend(["ax", "az"]);
    grid on

%     figure(14)
%     title("Downrange");
%     plot(downrange / 1000, hres / 1000)
%     ylabel("km");
%     xlabel("km");
   

    figure(15)
    hold on
    title("Forces");
    plot(timeTotal, T / 1000, 'r')
    plot(timeTotal, L / 1000, 'b')
    plot(timeTotal, D / 1000, 'k')
    ylabel("kN");
    xlabel("time [s]");
    legend("Thrust", "Lift", "Drag");
    grid on

    figure(16)
    hold on
    title("Mach");
    plot(timeTotal, M)
    xlabel("time [s]");
    grid on
    
    figure(17)
    hold on
    title("Total pitching Moment");
    plot(timeTotal, MomTot / 1000, 'k')
    ylabel("kNm");
    xlabel("time [s]");
    grid on

%     disp("m before Ho : ", mres(end))
%     disp("mf          : ", mf(end))
%     disp("altitude Hohmann starts: ", hres(end))
%     disp("final time  : ", time)

% fclose(res);
end

% writeNPY(vtot, 'v.npy')
% writeNPY(chitot, 'chi.npy')
% writeNPY(gammatot, "gamma.npy")
% writeNPY(tetatot, "teta.npy")
% writeNPY(lamtot, "lam.npy")
% writeNPY(htot, "h.npy")
% writeNPY(mtot, "m.npy")
% writeNPY(alfatot, "alfa.npy")
% writeNPY(deltatot, "delta.npy")
% writeNPY(deltaftot, "deltaf.npy")
% writeNPY(tautot, "tau.npy")
% writeNPY(mutot, "mu.npy")
% writeNPY(ttot, "time.npy")
% save('v.mat', 'vtot')
% save('chi.mat', 'chitot')
% save('gamma.mat', 'gammatot')
% save('teta.mat', 'tetatot')
% save('lam.mat', 'lamtot')
% save('h.mat', 'htot')
% save('m.mat', 'mtot')
% save('alfa.mat', 'alfatot')
% save('delta.mat', 'deltatot')
% save('deltaf.mat', 'deltaftot')
% save('tau.mat', 'tautot')
% save('mu.mat', 'mutot')
% save('t.mat', 'ttot')
% close(0);
% close(1);
% close(2);
% close(3);
% close(4);
% close(5);
% close(6);
% close(7);
% close(8);
% close(9);
% close(10);
% close(11);
% close(12);
% close(13);
% close(14);
% close(15);
% close(16);
% close(17);
end
        


   