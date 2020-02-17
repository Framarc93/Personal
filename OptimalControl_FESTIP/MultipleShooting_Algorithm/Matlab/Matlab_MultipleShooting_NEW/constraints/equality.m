function eq_cond = equality(var, conj, final_states, final_controls, obj, prob, file, states_init, cont_init)

hFinal = final_states(end, 6);
lamFinal = final_states(end, 4);
gammaFinal = final_states(end, 3);
%deltaFinal =final_controls(end, 2);
[vtAbs, chiass] = vass(final_states(end,:), final_controls(end,:), obj.omega, obj, file);
 
vvv = sqrt(obj.GMe / (obj.Re + hFinal));
 
if cos(obj.incl) / cos(lamFinal) > 1
     chifin = pi;
else
    if cos(obj.incl) / cos(lamFinal) < - 1
        chifin = 0.0;
    else
        chifin = 0.5 * pi + asin(cos(obj.incl) / cos(lamFinal));
    end
end

divv = [obj.vmax, obj.chimax, obj.gammamax, obj.tetamax, obj.lammax, obj.hmax, obj.M0];

%divc = [obj.alfamax, obj.deltamax, obj.deltafmax, obj.taumax];

eq_cond = zeros(0);
eq_cond = [eq_cond, (var(2:prob.varStates) - conj)./repmat(divv, 1, prob.Nleg-1)];  % knotting conditions 
eq_cond = [eq_cond, var(prob.varStates + 1)];  % init cond on alfa
eq_cond = [eq_cond, var(prob.varStates + 2) - 1.0];  % init cond on delta
eq_cond = [eq_cond, var(prob.varStates + 3)];  % init cond on deltaf
eq_cond = [eq_cond, var(prob.varStates + 4)];  % init cond on tau
i=0;
while i < prob.Nleg-1
    if i == 0
        eq_cond = [eq_cond, var(prob.varStates+prob.Ncontrols*prob.NContPointsLeg1-(prob.Ncontrols-1):prob.varStates+prob.Ncontrols*prob.NContPointsLeg1) ...
         - var(prob.varStates+prob.Ncontrols*prob.NContPointsLeg1+1:prob.varStates+prob.Ncontrols*prob.NContPointsLeg1+prob.Ncontrols)];
    else
        eq_cond = [eq_cond, var(prob.varStates+prob.NContPointsLeg1*prob.Ncontrols+i*prob.Ncontrols*prob.NContPoints-(prob.Ncontrols-1):prob.varStates+prob.NContPointsLeg1*prob.Ncontrols+i*prob.Ncontrols*prob.NContPoints) ...
         - var(prob.varStates+prob.NContPointsLeg1*prob.Ncontrols+i*prob.Ncontrols*prob.NContPoints+1:prob.varStates+prob.NContPointsLeg1*prob.Ncontrols+i*prob.Ncontrols*prob.NContPoints+prob.Ncontrols)];
    end
    i = i + 1;

end 
eq_cond = [eq_cond, (vtAbs-vvv)/obj.vmax];
eq_cond = [eq_cond, (chifin - chiass)/obj.chimax];
eq_cond = [eq_cond, gammaFinal/obj.gammamax]; % final condition on gamma

end

