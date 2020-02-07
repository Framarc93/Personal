function eq_cond = equality(var, conj, final_states, final_controls, obj, prob, file, states_init, cont_init)

hFinal = final_states(end - 1);
lamFinal = final_states(end - 2);
gammaFinal = final_states(end - 4);

[vtAbs, chiass] = vass(final_states, final_controls, obj.omega, obj, file);
 
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
eq_cond = zeros(0);
eq_cond = [eq_cond, (var(2:prob.varStates) - conj)./repmat(divv, 1, prob.Nleg-1)];  % knotting conditions 
eq_cond = [eq_cond, (var(prob.varStates + prob.Ncontrols) - 1.0)/obj.deltamax];  % init cond on delta
eq_cond = [eq_cond, (vvv - vtAbs)/obj.vmax];
eq_cond = [eq_cond, (chifin - chiass)/obj.chimax];
eq_cond = [eq_cond, gammaFinal/obj.gammamax]; % final condition on gamma

end

