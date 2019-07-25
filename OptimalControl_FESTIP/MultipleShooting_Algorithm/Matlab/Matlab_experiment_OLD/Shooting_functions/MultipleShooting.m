function [ineq_Cond, eq_Cond, objective] = MultipleShooting(var, obj, prob, file)
%in this function the states and controls are scaled%
%this function takes the data from the optimization variable, so the angles enters in radians%
global varOld costOld eqOld ineqOld States Controls globTime

timestart = 0.0;
states_atNode = zeros(1,prob.varStates);
globTime = zeros(1);
i=1;
states_after = zeros(prob.Nint*prob.Nleg, prob.Nstates);
controls_after = zeros(prob.Nint*prob.Nleg, prob.Ncontrols);

while i <= prob.Nleg 
    controls = extract_controls(var(prob.varStates+1+(prob.Ncontrols*prob.NContPoints)*(i-1):prob.varStates+i*prob.Ncontrols*prob.NContPoints), prob.NContPoints, prob.Ncontrols);
    states = convStatestoOrig(var(i*prob.Nstates-(prob.Nstates-1):i*prob.Nstates), prob.Nstates, obj, 1);  % orig intervals
    for j=1:prob.NContPoints
            controls(:,j) = convControlstoOrig(controls(:,j), prob.Ncontrols);
    end
    timeend = timestart + var(prob.varTot + i) * prob.unit_t;
    globTime = [globTime, timeend];

    [vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, ...
    taures, mures, alfa_I,  delta_I, deltaf_I, tau_I, mu_I] = SingleShooting(states, controls, timestart, timeend, prob, obj, file);
    
    states_atNode((i-1)*prob.Nstates+1:i*prob.Nstates) = [vres(end), chires(end), gammares(end), tetares(end), ...
                 lamres(end), hres(end), mres(end)]; % new intervals
    %res quantities are unscaled%

    %here controls and states matrices are defined%
    if i == 1
        states_after(i:prob.Nint,:) = [vres, chires, gammares, tetares, lamres, hres, mres];
        controls_after(i:prob.Nint,:) = [alfares', deltares', deltafres', taures', mures'];
    else
        states_after((i-1)*prob.Nint + 1:i*prob.Nint,:) = [vres, chires, gammares, tetares, lamres, hres, mres];
        controls_after((i-1)*prob.Nint + 1:i*prob.Nint,:) = [alfares', deltares', deltafres', taures', mures'];
    end
    i = i + 1;
    timestart = timeend;
end
   
h = states_after(end, 6);
if isnan(h)
    h(isnan(h)) = 1000;
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

ineq_Cond = inequalityAll(states_after, controls_after, prob.Nint*prob.Nleg, obj, file);% evaluation of inequality constraints


eq_Cond = equality(var, states_atNode, obj, prob, file);  % evaluation of equality constraints
objective = - mf / obj.M0;  % evaluation of objective function

costOld = objective;
eqOld = eq_Cond;
ineqOld = ineq_Cond;
varOld = var;

end

