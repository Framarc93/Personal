function out = Plot_MS(var, prob, obj, file, states_init)

timestart = 0.0;

varD = var.*(prob.UBV - prob.LBV) + prob.LBV;

i = 0;
while i <= prob.Nleg-1
    if i == 0
        states = states_init;
        states(2) = varD(1);
        controls = extract_controls(varD(prob.varStates+1+(prob.Ncontrols*prob.NContPointsLeg1)*i:prob.varStates+(i+1)*prob.Ncontrols*prob.NContPointsLeg1), prob.NContPointsLeg1, prob.Ncontrols);
    else
        states = varD(1+i*prob.Nstates-(prob.Nstates-1):i*prob.Nstates+1);  
        controls = extract_controls(varD(prob.varStates+1+prob.NContPointsLeg1*prob.Ncontrols+(prob.Ncontrols*prob.NContPoints)*(i-1):prob.varStates+prob.NContPointsLeg1*prob.Ncontrols+i*prob.Ncontrols*prob.NContPoints), prob.NContPoints, prob.Ncontrols);
    end
    alfaCP = controls(1,:);
    deltaCP = controls(2,:);
    deltafCP = controls(3,:);
    tauCP = controls(4,:);
    timeend = timestart + varD(prob.varTot + i+1);
  
    tC = linspace(timestart, timeend, length(alfaCP));
    [vres, chires, gammares, tetares, lamres, hres, mres, t, alfares, deltares, deltafres, taures] = SingleShooting(states, controls, timestart, timeend, prob, obj, file);
    %taures = zeros(1, length(vres))';
    [Press, rho, c] = isaMulti(hres, obj);

    M = vres' ./ c;
    varnum=prob.Nint;

    [L, D, MomA] = aeroForcesMulti(M, alfares, deltafres, file.cd, file.cl, file.cm, vres, rho, mres, obj, varnum);

    [T, Deps, isp, MomT] = thrustMulti(Press, mres, file.presv, file.spimpv, deltares, taures, obj, varnum);

    MomTot = MomA + MomT;


    % dynamic pressure

    q = 0.5 * rho .* (vres' .^ 2);

    % accelerations

    ax = (T .* cos(Deps) - D .* cos(alfares') + L .* sin(alfares')) ./ mres';
    az = (T .* sin(Deps) + D .* sin(alfares') + L .* cos(alfares')) ./ mres';

    r1 = hres(end) + obj.Re;
    Dv1 = sqrt(obj.GMe / r1) * (sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1);
    Dv2 = sqrt(obj.GMe / obj.r2) * (1 - sqrt((2 * r1) / (r1 + obj.r2)));
    mf = mres(end) / exp((Dv1 + Dv2) / (obj.g0*isp(end)));


    figure(100000)
    hold on
    title("Velocity");
    plot(t, vres)
    ylabel("m/s");
    xlabel("time [s]");
    grid on


    figure(1)
    hold on
    title("Heading");
    plot(t, rad2deg(chires))
    ylabel("deg");
    xlabel("time [s]");
    grid on


    figure(2)
    hold on
    title("Flight path angle");
    plot(t, rad2deg(gammares))
    ylabel("deg");
    xlabel("time [s]");
    grid on

    figure(3)
    hold on
    title("Longitude");
    plot(t, rad2deg(tetares))
    ylabel("deg");
    xlabel("time [s]");
    grid on

    figure(4)
    hold on
    title("Latitude");
    plot(t, rad2deg(lamres))
    ylabel("deg");
    xlabel("time [s]");
    grid on


    figure(5)
    hold on
    title("Flight angles");
    plot(t, rad2deg(chires), 'g')
    plot(t, rad2deg(gammares), 'b')
    plot(t, rad2deg(tetares), 'r')
    plot(t, rad2deg(lamres), 'k')
    ylabel("deg");
    xlabel("time [s]");
    legend("Chi", "Gamma", "Theta", "Lambda");
    grid on


    figure(6)
    hold on
    title("Altitude");
    plot(t, hres / 1000)
    ylabel("km");
    xlabel("time [s]");
    grid on

    figure(7)
    hold on
    title("Mass");
    plot(t, mres)
    ylabel("kg");
    xlabel("time [s]");
    grid on

    figure(8)
    hold on
    title("Angle of attack");
    plot(tC, rad2deg(alfaCP), 'ro')
    plot(t, rad2deg(alfares));
    ylabel("deg");
    xlabel("time [s]");
    legend(["Control points"]);
    grid on


    figure(9)
    hold on
    title("Throttles");
    plot(t, deltares * 100, 'r')
    plot(t, taures * 100, 'k')
    plot(tC, deltaCP * 100, 'ro')
    plot(tC, tauCP * 100, 'ro')
    ylabel("%");
    xlabel("time [s]");
    legend("Delta", "Tau", "Control points");
    grid on

    figure(10)
    hold on
    title("Body Flap deflection");
    plot(tC, rad2deg(deltafCP), "ro")
    plot(t, rad2deg(deltafres));
    ylabel("deg");
    xlabel("time [s]");
    legend(["Control points"]);
    grid on

    %figure(11)
    %hold on
    %     title("Bank angle profile");
    %     plot(tC, rad2deg(muCP(i, :)), "ro")
    %     plot(timeTotal, rad2deg(mures))
    %     ylabel("deg");
    %     xlabel("time [s]");
    %     legend(["Control points"]);
    %     grid on


    figure(12)
    hold on
    title("Dynamic Pressure");
    plot(t, q / 1000)
    ylabel("kPa");
    xlabel("time [s]");
    grid on


    figure(13)
    hold on
    title("Accelerations");
    plot(t, ax, 'b')
    plot(t, az, 'r')
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
    plot(t, T / 1000, 'r')
    plot(t, L / 1000, 'b')
    plot(t, D / 1000, 'k')
    ylabel("kN");
    xlabel("time [s]");
    legend("Thrust", "Lift", "Drag");
    grid on

    figure(16)
    hold on
    title("Mach");
    plot(t, M)
    xlabel("time [s]");
    grid on

    figure(17)
    hold on
    title("Total pitching Moment");
    plot(t, MomTot / 1000, 'k')
    ylabel("kNm");
    xlabel("time [s]");
    grid on
    timestart = timeend;
    i = i + 1;
end
        


   