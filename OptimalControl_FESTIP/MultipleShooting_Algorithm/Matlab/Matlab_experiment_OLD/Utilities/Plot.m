function Plot(var, prob, obj, file)
global States Controls globTime
time = zeros(1);
timeTotal = [];
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

timestart = 0;

tCtot = [];

for i=1:prob.Nleg
    timeend = timestart + var(i + prob.varTot) * prob.unit_t;
    timeTotal = linspace(timestart, timeend, prob.Nint);
    time = [time, timeend];
    tC = linspace(timestart, timeend, prob.NContPoints);
    tCtot = [tCtot, tC];
    controlsCp = extract_controls(var(prob.varStates+1+(prob.Ncontrols*prob.NContPoints)*(i-1):prob.varStates+i*prob.Ncontrols*prob.NContPoints), prob.NContPoints, prob.Ncontrols);
    for j=1:prob.NContPoints
            controlsCp(:,j) = convControlstoOrig(controlsCp(:,j), prob.Ncontrols);
    end
    timestart = timeend;

    vres = States((i-1) * prob.Nint+1:i * prob.Nint, 1);
    chires = States((i-1) * prob.Nint+1:i* prob.Nint, 2);
    gammares = States((i-1) * prob.Nint+1:i * prob.Nint, 3);
    tetares = States((i-1) * prob.Nint+1:i* prob.Nint, 4);  % teta to [-90,0]
    lamres = States((i-1) * prob.Nint+1:i * prob.Nint, 5);  % lam to [-incl, incl]
    hres = States((i-1) * prob.Nint+1:i * prob.Nint, 6);
    mres = States((i-1) * prob.Nint+1:i * prob.Nint, 7);

    alfares = Controls((i-1) * prob.Nint+1:i * prob.Nint, 1);
    deltares = Controls((i-1) * prob.Nint+1:i * prob.Nint, 2);
    deltafres = Controls((i-1) * prob.Nint+1:i * prob.Nint, 3);
    taures = Controls((i-1) * prob.Nint+1:i * prob.Nint, 4);  % tau to (-1,1]
    mures = Controls((i-1) * prob.Nint+1:i * prob.Nint, 5);
    
    vtot = [vtot, vres'];
    chitot = [chitot, chires'];
    gammatot = [gammatot, gammares'];
    tetatot = [tetatot, tetares'];
    lamtot = [lamtot, lamres'];
    htot = [htot, hres'];
    mtot = [mtot, mres'];
    alfatot = [alfatot, alfares'];
    deltatot = [deltatot, deltares'];
    deltaftot = [deltaftot, deltafres'];
    tautot = [tautot, taures'];
    mutot = [mutot, mures'];
    
    rep = prob.Nint;

    [Press, rho, c] = isaMulti(hres, obj);
    
    M = vres' ./ c;

    [L, D, MomA] = aeroForcesMulti(M, alfares, deltafres, file.cd, file.cl, file.cm, vres, rho, mres, obj, rep);
    

    [T, Deps, isp, MomT] = thrustMulti(Press, mres, file.presv, file.spimpv, deltares, taures, obj, rep);

    MomTot = MomA + MomT;

    r1 = hres' + obj.Re;
    Dv1 = sqrt(obj.GMe ./ r1) .* (sqrt((2 * obj.r2) ./ (r1 + obj.r2)) - 1);
    Dv2 = sqrt(obj.GMe ./ obj.r2) .* (1 - sqrt((2 * r1) ./ (r1 + obj.r2)));
    mf = mres' ./ exp((Dv1 + Dv2) ./ (obj.g0*isp));

    g0 = obj.g0;
    eps = Deps + alfares';
    g = [];
    for j=1:size(hres)
        if hres(j) == 0
            g = [g, g0];
        else
            g = [g, (obj.g0 * (obj.Re ./ (obj.Re + hres(j))) .^ 2)];
        end
    end
    
   
    % dynamic pressure

    q = 0.5 .* rho .* (vres' .^ 2);

    % accelerations

    ax = (T .* cos(Deps) - D .* cos(alfares') + L .* sin(alfares')) ./ mres';
    az = (T .* sin(Deps) + D .* sin(alfares') + L .* cos(alfares')) ./ mres';

%     fprintf(res, "Number of leg: " + num2str(prob.Nleg) + "\n" + "Leg Number:" + num2str(i) + "\n" + "v: " + num2str( ...
%         vres) + "\n" + "Chi: " + num2str(rad2deg(chires)) ...
%         + "\n" + "Gamma: " + num2str(rad2deg(gammares)) + "\n" + "Teta: " + num2str( ...
%         rad2deg(tetares)) + "\n" + "Lambda: " ...
%         + num2str(rad2deg(lamres)) + "\n" + "Height: " + num2str(hres) + "\n" + "Mass: " + num2str( ...
%         mres) + "\n" + "mf: " + num2str(mf) + "\n" ...
%         + "Objective Function: " + num2str(-mf / obj.M0) + "\n" + "Alfa: " ...
%         + num2str(rad2deg(alfares)) + "\n" + "Delta: " + num2str(deltares) + "\n" + "Delta f: " + num2str( ...
%         rad2deg(deltafres)) + "\n" ...
%         + "Tau: " + num2str(taures) + "\n" + "Eps: " + num2str(rad2deg(eps)) + "\n" + "Lift: " ...
%         + num2str(L) + "\n" + "Drag: " + num2str(D) + "\n" + "Thrust: " + num2str(T) + "\n" + "Spimp: " + num2str( ...
%         isp) + "\n" + "c: " ...
%         + num2str(c) + "\n" + "Mach: " + num2str(M) + "\n" + "Time vector: " + num2str(timeTotal) + "\n" + "Press: " + num2str( ...
%         Press) + "\n" + "Dens: " + num2str(rho));
    
% fprintf(fid, [ header1 ' ' header2 'r\n']);
% fprintf(fid, '%f %f r\n', [A B]');
    downrange = - (vres' .^ 2) ./ g .* sin(2 .* gammares');

    figure(100)
    hold on
    title("Velocity");
    plot(timeTotal, vres)
    ylabel("m/s");
    xlabel("time [s]");
    
    figure(1)
    hold on
    title("Flight path angle");
    plot(timeTotal, rad2deg(chires))
    ylabel("deg");
    xlabel("time [s]");
    
    figure(2)
    hold on
    title("Angle of climb");
    plot(timeTotal, rad2deg(gammares))
    ylabel("deg");
    xlabel("time [s]");
 
    figure(3)
    hold on
    title("Longitude");
    plot(timeTotal, rad2deg(tetares))
    ylabel("deg");
    xlabel("time [s]");

    figure(4)
    hold on
    title("Latitude");
    plot(timeTotal, rad2deg(lamres))
    ylabel("deg");
    xlabel("time [s]");

    figure(5)
    hold on
    title("Flight angles");
    plot(timeTotal, rad2deg(chires), "g")
    plot(timeTotal, rad2deg(gammares), "b")
    plot(timeTotal, rad2deg(tetares), "r")
    plot(timeTotal, rad2deg(lamres), "k")
    ylabel("deg");
    xlabel("time [s]");


    figure(6)
    hold on
    title("Altitude");
    plot(timeTotal, hres / 1000)
    ylabel("km");
    xlabel("time [s]");
 
    figure(7)
    hold on
    title("Mass");
    plot(timeTotal, mres)
    ylabel("kg");
    xlabel("time [s]");

    figure(8)
    hold on
    title("Angle of attack");
    plot(tC, rad2deg(alfaCP(i, :)), 'ro')
    plot(timeTotal, rad2deg(alfares));
    ylabel("deg");
    xlabel("time [s]");


    figure(9)
    hold on
    title("Throttles");
    plot(timeTotal, deltares * 100, 'r')
    plot(timeTotal, taures * 100, 'k')
    plot(tC, deltaCP(i, :) * 100, 'ro')
    plot(tC, tauCP(i, :) * 100, 'ro')
    ylabel("%");
    xlabel("time [s]");

    figure(10)
    hold on
    title("Body Flap deflection");
    plot(tC, rad2deg(deltafCP(i, :)), "ro")
    plot(timeTotal, rad2deg(deltafres));
    ylabel("deg");
    xlabel("time [s]");

    figure(11)
    hold on
    title("Bank angle profile");
    plot(tC, rad2deg(muCP(i, :)), "ro")
    plot(timeTotal, rad2deg(mures))
    ylabel("deg");
    xlabel("time [s]");


    figure(12)
    hold on
    title("Dynamic Pressure");
    plot(timeTotal, q / 1000)
    ylabel("kPa");
    xlabel("time [s]");


    figure(13)
    hold on
    title("Accelerations");
    plot(timeTotal, ax, 'b')
    plot(timeTotal, az, 'r')
    ylabel("m/s^2");
    xlabel("time [s]");


    figure(14)
    hold on
    title("Downrange");
    plot(downrange / 1000, hres / 1000)
    ylabel("km");
    xlabel("km");


    figure(15)
    hold on
    title("Forces");
    plot(timeTotal, T / 1000, 'r')
    plot(timeTotal, L / 1000, 'b')
    plot(timeTotal, D / 1000, 'k')
    ylabel("kN");
    xlabel("time [s]");


    figure(16)
    hold on
    title("Mach");
    plot(timeTotal, M)
    xlabel("time [s]");


    figure(17)
    title("Total pitching Moment");
    plot(timeTotal, MomTot / 1000, 'k')
    ylabel("kNm");
    xlabel("time [s]");

end

disp("m before Ho : ")
disp(mres(end))
disp("mf          : ")
disp(mf(end))
disp("altitude Hohmann starts: ")
disp(hres(end))
disp("final time  : ")
disp(time)
end
        


   