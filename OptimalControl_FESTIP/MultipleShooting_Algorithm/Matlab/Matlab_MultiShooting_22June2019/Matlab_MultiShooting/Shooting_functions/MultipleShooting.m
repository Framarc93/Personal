function [ineq_Cond, eq_Cond, objective] = MultipleShooting(var, obj, prob, file)
%in this function the states and controls are scaled%
%this function takes the data from the optimization variable, so the angles enters in radians%
global varOld costOld eqOld ineqOld States Controls globTime
global Pissue

vineq = zeros(1, prob.Nint); % states and controls defined as row vectors
chiineq = zeros(1, prob.Nint);
gammaineq = zeros(1, prob.Nint);
tetaineq = zeros(1, prob.Nint);
lamineq = zeros(1, prob.Nint);
hineq = zeros(1, prob.Nint);
mineq = zeros(1, prob.Nint);
alfaineq = zeros(1, prob.Nint);
deltaineq = zeros(1, prob.Nint);
deltafineq = zeros(1, prob.Nint);
tauineq = zeros(1, prob.Nint);
muineq = zeros(1, prob.Nint);

timestart = 0.0;

states_atNode = zeros(1,prob.Nleg*prob.Nstates);
globTime = zeros(1);
i=1;
states_after = zeros(prob.Nint*prob.Nleg, prob.Nstates);
controls_after = zeros(prob.Nint*prob.Nleg, prob.Ncontrols);
Pissue =0;
varD = var.*(prob.UBV - prob.LBV) + prob.LBV;
tVec=[];
while i <= prob.Nleg 
    controls = extract_controls(varD(prob.varStates+1+(prob.Ncontrols*prob.NContPoints)*(i-1):prob.varStates+i*prob.Ncontrols*prob.NContPoints), prob.NContPoints, prob.Ncontrols);
    states = varD(i*prob.Nstates-(prob.Nstates-1):i*prob.Nstates);  % orig intervals
    
    timeend = timestart + varD(prob.varTot + i);

    globTime = [globTime, timeend];

    [vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, ...
    taures, mures, alfa_I,  delta_I, deltaf_I, tau_I, mu_I] = SingleShooting(states, controls, timestart, timeend, prob, obj, file);
    
    states_atNode((i-1)*prob.Nstates+1:i*prob.Nstates) = [vres(end), chires(end), gammares(end), tetares(end), ...
                 lamres(end), hres(end), mres(end)]; % new intervals
    %res quantities are unscaled%

    %these are row vectors%
    vineq(1, :) = vres;  % vinterp(time_ineq)
    chiineq(1, :) = chires;  % chiinterp(time_ineq)
    gammaineq(1, :) = gammares;  % gammainterp(time_ineq)
    tetaineq(1, :) = tetares;  % tetainterp(time_ineq)
    lamineq(1, :) = lamres;  % laminterp(time_ineq)
    hineq(1, :) = hres;  % hinterp(time_ineq)
    mineq(1, :) = mres;  % minterp(time_ineq)
    alfaineq(1, :) = alfares;  % alfa_I(time_ineq)
    deltaineq(1, :) = deltares;  % delta_I(time_ineq)
    deltafineq(1, :) = deltafres;  % deltaf_I(time_ineq)
    tauineq(1, :) = taures;  % tau_I(time_ineq)
    muineq(1, :) = mures;

    
    %here controls and states matrices are defined%
    if i == 1
        states_after(i:prob.Nint,:) = [vineq', chiineq', gammaineq', tetaineq', lamineq', hineq', mineq'];
        controls_after(i:prob.Nint,:) = [alfaineq', deltaineq', deltafineq', tauineq', muineq'];
    else
        states_after((i-1)*prob.Nint + 1:i*prob.Nint,:) = [vineq', chiineq', gammaineq', tetaineq', lamineq', hineq', mineq'];
        controls_after((i-1)*prob.Nint + 1:i*prob.Nint,:) = [alfaineq', deltaineq', deltafineq', tauineq', muineq'];
    end
    tVec=[tVec; tres'];
    i = i + 1;
    timestart = timeend;
end
   
h = states_after(end, 6);
if isnan(h)
h=0;
end
m = states_after(end, 7);
delta = controls_after(end, 2);
tau = controls_after(end, 4);

[Press, rho, c] = isa_FESTIP(h, obj);

[T, Deps, isp, MomT] = prop_FESTIP(Press, m, file.presv, file.spimpv, delta, tau, obj);

r1 = h + obj.Re;
Dv1 = sqrt(obj.GMe / r1) * (sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1);
Dv2 = sqrt(obj.GMe / obj.r2) * (1 - sqrt((2 * r1) / (r1 + obj.r2)));
mf = m / exp((Dv1 + Dv2) / (obj.g0 * isp));

States = states_after;
Controls = controls_after;

ineq_Cond = inequalityAll(states_after, controls_after, size(states_after), obj, file);% evaluation of inequality constraints


eq_Cond = equality(varD, states_atNode, obj, prob, file);  % evaluation of equality constraints
objective = - mf / obj.M0;  % evaluation of objective function
if Pissue==1
   objective=objective+1;
   ineq_Cond=ineq_Cond+1e3;
   eq_Cond=eq_Cond*1e3;
end
costOld = objective;
eqOld = eq_Cond;
ineqOld = ineq_Cond;
varOld = var;

end

